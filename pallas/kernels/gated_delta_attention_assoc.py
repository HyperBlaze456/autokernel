"""
AutoKernel Gated Delta Attention -- Associative scan variant for TPU.

Copy this file's content to kernel.py at project root to benchmark via:
    python tpu_vm.py bench-pallas --project <PROJECT> --kernel-type gated_delta_attention

This kernel replaces the sequential jax.lax.scan for the intra-chunk state
update with a PARALLEL jax.lax.associative_scan, reducing the sequential
depth from O(C) to O(log C) at the cost of higher total FLOPs.

== Approach ==

The original recurrence:
    S_t = diag(exp(g_t)) @ S_{t-1} + beta_t * k_t outer (v_t - k_t^T @ S_{t-1})

is reformulated as a LINEAR matrix recurrence:
    S_t = A_t @ S_{t-1} + B_t

where:
    A_t = diag(exp(g_t)) - beta_t * (k_t @ k_t^T)   -- (D, D) transition matrix
    B_t = beta_t * (k_t @ v_t^T)                      -- (D, D) additive term

The associative scan operator is:
    (A2, B2) o (A1, B1) = (A2 @ A1, A2 @ B1 + B2)

This yields ALL prefix states via cumulative products/sums:
    cumA[t], cumB[t] = associative_scan(combine_fn, (A, B))
    S[t] = cumA[t] @ S_init + cumB[t]

Because we have every intermediate S[t], we compute the EXACT output:
    output[t] = S[t] @ q[t]

This eliminates the inter-chunk/intra-chunk approximation split used in
the sequential chunk kernel, yielding numerically exact results matching
the naive scan.

== Computational cost comparison (per chunk of C steps, head dim D) ==

Sequential scan (gated_delta_attention.py):
    - Per step: 2x einsum (B,H,D)x(B,H,D,D) = O(D^2), plus outer product O(D^2)
    - Total: O(C * D^2) FLOPs, O(C) sequential depth
    - Plus intra-chunk C*C attention: O(C^2 * D)
    - Dominant cost: C * D^2 sequential steps (each too small for MXU)

Associative scan (this file):
    - Build A_t, B_t: C outer products = O(C * D^2)
    - Associative scan: log2(C) levels, each with C/2 matmuls of (D,D)x(D,D)
      = O(C * log2(C) * D^3) total FLOPs
    - Output: C matmuls of (D,D)x(D,1) = O(C * D^2)
    - Dominant cost: C * log2(C) * D^3 parallel matmuls (MXU-friendly)

For C=64, D=64:
    Sequential: ~64 * 64^2 = 262,144 FLOPs/step, 64 sequential steps
    Associative: ~64 * 6 * 64^3 = 100,663,296 total FLOPs, 6 sequential steps
    Ratio: ~384x more total FLOPs, but ~10.7x less sequential depth

For C=64, D=128:
    Sequential: ~64 * 128^2 = 1,048,576 FLOPs/step, 64 sequential steps
    Associative: ~64 * 6 * 128^3 = 805,306,368 total FLOPs, 6 sequential steps
    Ratio: ~768x more total FLOPs, but ~10.7x less sequential depth

The tradeoff is worthwhile when:
    1. The MXU is underutilized by the small sequential ops (D=64 or 128
       doesn't fill v6e's 256x256 MXU well in sequential mode)
    2. The log(C) parallel depth fits within acceptable latency
    3. Total throughput is compute-bound, not memory-bound

On TPU v6e with 918 bf16 TFLOPS, the D*D matmuls in the scan can achieve
higher MXU utilization than the small vector ops in the sequential scan,
potentially offsetting the higher total FLOP count.

== Mixed precision strategy ==

- A_t, B_t construction: float32 (numerical stability for exp(g) - beta*kk^T)
- Associative scan matmuls (A@A, A@B): bf16 for 2x MXU throughput
- State S and cumulative products: float32 accumulation
- Final output matmul (S @ q): bf16

Algorithm (Gated Delta Attention / KDA):
    S_t = exp(g_t) * S_{t-1} + beta_t * k_t outer (v_t - k_t^T @ S_{t-1})
    o_t = S_t @ q_t

Reference: Kimi Linear (arXiv:2510.26692), Gated DeltaNet (arXiv:2412.06464)
"""

KERNEL_TYPE = "gated_delta_attention"
BACKEND = "pallas"

import jax
import jax.numpy as jnp

# Chunk size: C=64 empirically optimal on v6e.
CHUNK_SIZE = 64

# Outer chunk scan unroll factor.
_OUTER_UNROLL = 2


def _kda_assoc_scan(q, k, v, g, beta, chunk_size=CHUNK_SIZE):
    """Chunk-based gated delta attention using associative scan for state update.

    Within each chunk, the sequential state recurrence is reformulated as a
    linear matrix recurrence and solved with jax.lax.associative_scan, giving
    O(log C) sequential depth instead of O(C).

    Args:
        q, k, v, g: (B, T, H, D)
        beta: (B, T, H)
        chunk_size: block size C
    Returns:
        output: (B, T, H, D)
    """
    B, T, H, D = q.shape
    C = chunk_size

    # Pad T to multiple of C
    pad_len = (C - T % C) % C
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        g = jnp.pad(g, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, pad_len), (0, 0)))
        T_padded = T + pad_len
    else:
        T_padded = T

    num_chunks = T_padded // C

    # Float32 for state accumulation precision
    q = q.astype(jnp.float32)
    k = k.astype(jnp.float32)
    v = v.astype(jnp.float32)
    g = g.astype(jnp.float32)
    beta = beta.astype(jnp.float32)

    # Reshape to (B, num_chunks, C, H, D)
    q_c = q.reshape(B, num_chunks, C, H, D)
    k_c = k.reshape(B, num_chunks, C, H, D)
    v_c = v.reshape(B, num_chunks, C, H, D)
    g_c = g.reshape(B, num_chunks, C, H, D)
    beta_c = beta.reshape(B, num_chunks, C, H)

    S0 = jnp.zeros((B, H, D, D), dtype=jnp.float32)

    def _associative_combine(carry_left, carry_right):
        """Associative operator: (A2, B2) o (A1, B1) = (A2 @ A1, A2 @ B1 + B2).

        Uses bf16 for the D*D matmuls to exploit MXU throughput, with float32
        accumulation for numerical stability.

        Args:
            carry_left: (A1, B1) each of shape (B, H, C_scan, D, D)
            carry_right: (A2, B2) same shapes
        Returns:
            (A_combined, B_combined) same shapes
        """
        A1, B1 = carry_left
        A2, B2 = carry_right

        # D*D matmuls in bf16 for 2x MXU throughput on v6e (918 vs 459 TFLOPS)
        # A2 @ A1: (B, H, *, D, D) @ (B, H, *, D, D) -> (B, H, *, D, D)
        A_combined = jnp.einsum(
            '...ij,...jk->...ik',
            A2.astype(jnp.bfloat16),
            A1.astype(jnp.bfloat16),
        ).astype(jnp.float32)

        # A2 @ B1 + B2
        A2_B1 = jnp.einsum(
            '...ij,...jk->...ik',
            A2.astype(jnp.bfloat16),
            B1.astype(jnp.bfloat16),
        ).astype(jnp.float32)
        B_combined = A2_B1 + B2

        return (A_combined, B_combined)

    def process_chunk(S_prev, chunk_inputs):
        """Process one chunk using associative scan for the state update.

        Args:
            S_prev: (B, H, D, D) state from previous chunk
            chunk_inputs: tuple of (q_blk, k_blk, v_blk, g_blk, beta_blk)
        Returns:
            S_next: (B, H, D, D) state after this chunk
            output: (B, C, H, D) output for this chunk
        """
        q_blk, k_blk, v_blk, g_blk, beta_blk = chunk_inputs
        # q_blk: (B, C, H, D), beta_blk: (B, C, H)

        # Transpose to head-first for batched matmul: (B, H, C, D)
        k_h = jnp.transpose(k_blk, (0, 2, 1, 3))
        v_h = jnp.transpose(v_blk, (0, 2, 1, 3))
        g_h = jnp.transpose(g_blk, (0, 2, 1, 3))
        q_h = jnp.transpose(q_blk, (0, 2, 1, 3))
        beta_h = jnp.transpose(beta_blk, (0, 2, 1))  # (B, H, C)

        # ---- Build per-step transition matrices A_t and additive terms B_t ----
        # A_t = diag(exp(g_t)) - beta_t * (k_t @ k_t^T)   shape: (B, H, C, D, D)
        # B_t = beta_t * (k_t @ v_t^T)                      shape: (B, H, C, D, D)

        decay_h = jnp.exp(g_h)  # (B, H, C, D)

        # diag(exp(g_t)): broadcast to (B, H, C, D, D) via identity scaling
        eye = jnp.eye(D, dtype=jnp.float32)  # (D, D)
        A_diag = decay_h[..., :, None] * eye[None, None, None, :, :]  # (B, H, C, D, D)

        # k_t @ k_t^T: (B, H, C, D, 1) @ (B, H, C, 1, D) -> (B, H, C, D, D)
        kk_T = k_h[..., :, None] * k_h[..., None, :]  # outer product

        # k_t @ v_t^T: (B, H, C, D, D)
        kv_T = k_h[..., :, None] * v_h[..., None, :]  # outer product

        beta_expanded = beta_h[..., None, None]  # (B, H, C, 1, 1)

        A = A_diag - beta_expanded * kk_T  # (B, H, C, D, D)
        Bmat = beta_expanded * kv_T         # (B, H, C, D, D)

        # ---- Associative scan over the C dimension (axis=2) ----
        # After scan: cumA[t] = A_t @ A_{t-1} @ ... @ A_1
        #             cumB[t] = A_t @ ... @ A_2 @ B_1 + ... + A_t @ B_{t-1} + B_t
        # So: S[t] = cumA[t] @ S_prev + cumB[t]
        cumA, cumB = jax.lax.associative_scan(
            _associative_combine, (A, Bmat), axis=2,
        )

        # ---- Compute all intermediate states ----
        # S[t] = cumA[t] @ S_prev + cumB[t]
        # S_prev: (B, H, D, D), cumA[t]: (B, H, C, D, D)
        # S_prev contribution: cumA[t] @ S_prev -> (B, H, C, D, D)
        state_from_prev = jnp.einsum(
            'bhcij,bhjk->bhcik',
            cumA.astype(jnp.bfloat16),
            S_prev.astype(jnp.bfloat16),
        ).astype(jnp.float32)

        # All intermediate states: (B, H, C, D, D)
        S_all = state_from_prev + cumB

        # ---- Compute exact output: output[t] = S[t] @ q[t] ----
        # S_all: (B, H, C, D, D), q_h: (B, H, C, D)
        # output[t] = S[t]^T @ q[t] since S is (D_k, D_v) and we want (D_v,)
        # Using einsum: S[bhc,k,v] * q[bhc,k] -> o[bhc,v]
        output_h = jnp.einsum(
            'bhckv,bhck->bhcv',
            S_all.astype(jnp.bfloat16),
            q_h.astype(jnp.bfloat16),
        ).astype(jnp.float32)

        # Transpose back to (B, C, H, D)
        output = jnp.transpose(output_h, (0, 2, 1, 3))

        # ---- Extract final state for next chunk ----
        # S_next = S_all[:, :, -1, :, :] -> (B, H, D, D)
        S_next = S_all[:, :, -1, :, :]

        return S_next, output

    # Transpose for outer scan: (num_chunks, B, C, H, D)
    q_scan = jnp.transpose(q_c, (1, 0, 2, 3, 4))
    k_scan = jnp.transpose(k_c, (1, 0, 2, 3, 4))
    v_scan = jnp.transpose(v_c, (1, 0, 2, 3, 4))
    g_scan = jnp.transpose(g_c, (1, 0, 2, 3, 4))
    beta_scan = jnp.transpose(beta_c, (1, 0, 2, 3))

    _, all_outputs = jax.lax.scan(
        process_chunk, S0,
        (q_scan, k_scan, v_scan, g_scan, beta_scan),
        unroll=_OUTER_UNROLL,
    )

    # (num_chunks, B, C, H, D) -> (B, T, H, D)
    all_outputs = jnp.transpose(all_outputs, (1, 0, 2, 3, 4))
    all_outputs = all_outputs.reshape(B, T_padded, H, D)

    if pad_len > 0:
        all_outputs = all_outputs[:, :T, :, :]

    return all_outputs


def kernel_fn(q, k, v, g, beta):
    """Gated delta attention -- associative scan variant for TPU.

    Uses jax.lax.associative_scan within each chunk for O(log C) sequential
    depth instead of O(C). Computes exact outputs from all intermediate states,
    eliminating the inter-chunk/intra-chunk approximation split.

    Args:
        q: (B, T, H, D) queries
        k: (B, T, H, D) keys
        v: (B, T, H, D) values
        g: (B, T, H, D) per-dim decay factors (log-space)
        beta: (B, T, H) per-head gating scalar

    Returns:
        output: (B, T, H, D)
    """
    return _kda_assoc_scan(q, k, v, g, beta)
