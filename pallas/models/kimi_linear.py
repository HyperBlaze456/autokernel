"""
Minimal Kimi Linear model for AutoKernel TPU profiling.

Self-contained JAX implementation of the Kimi Linear architecture with
Gated Delta Attention (KDA) -- no external dependencies beyond JAX.

The model uses a hybrid architecture with KDA linear attention layers
and optional full attention layers (3:1 KDA-to-full ratio in the paper).

Reference: "Kimi Linear: An Expressive, Efficient Attention Architecture"
           (arXiv:2510.26692, Moonshot AI)

Usage:
    # Copy to kernel.py with appropriate wrapper for benchmarking
    # Or use directly for model-level profiling
"""

import functools
import math
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class KimiLinearConfig(NamedTuple):
    """Configuration for a Kimi Linear model."""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    intermediate_size: int = 2048
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 2048
    # KDA-specific
    kda_ratio: int = 3          # Every kda_ratio+1 layers, one is full attention
    short_conv_kernel_size: int = 4


# Small config for testing (fits any accelerator)
KIMI_LINEAR_SMALL = KimiLinearConfig(
    vocab_size=32000, hidden_size=768, num_layers=12, num_heads=12,
    head_dim=64, intermediate_size=2048, max_seq_len=2048,
)

# Medium config (~1B params)
KIMI_LINEAR_1B = KimiLinearConfig(
    vocab_size=32000, hidden_size=2048, num_layers=24, num_heads=16,
    head_dim=128, intermediate_size=5504, max_seq_len=4096,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)."""
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def silu(x: jnp.ndarray) -> jnp.ndarray:
    """SiLU / Swish activation."""
    return x * jax.nn.sigmoid(x)


# ---------------------------------------------------------------------------
# Gated Delta Attention (KDA) -- core mechanism
# ---------------------------------------------------------------------------

def gated_delta_attention_naive(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Naive sequential gated delta attention using lax.scan.

    Args:
        q: (B, T, H, D)  queries
        k: (B, T, H, D)  keys
        v: (B, T, H, D)  values
        g: (B, T, H, D)  per-dim decay factors (log-space)
        beta: (B, T, H)  per-head gating scalars
        initial_state: (B, H, D, D) optional initial recurrent state

    Returns:
        output: (B, T, H, D)
        final_state: (B, H, D, D)
    """
    B, T, H, D = q.shape

    if initial_state is None:
        S0 = jnp.zeros((B, H, D, D), dtype=jnp.float32)
    else:
        S0 = initial_state.astype(jnp.float32)

    # Pack inputs for scan: (T, B, ...)
    q_t = jnp.transpose(q, (1, 0, 2, 3)).astype(jnp.float32)
    k_t = jnp.transpose(k, (1, 0, 2, 3)).astype(jnp.float32)
    v_t = jnp.transpose(v, (1, 0, 2, 3)).astype(jnp.float32)
    g_t = jnp.transpose(g, (1, 0, 2, 3)).astype(jnp.float32)
    beta_t = jnp.transpose(beta, (1, 0, 2)).astype(jnp.float32)  # (T, B, H)

    def scan_step(S, inputs):
        q_i, k_i, v_i, g_i, beta_i = inputs

        # Exponential decay
        decay = jnp.exp(g_i)  # (B, H, D)
        S = S * decay[:, :, :, None]

        # Delta rule: retrieval = k^T @ S
        retrieval = jnp.einsum('bhk,bhkv->bhv', k_i, S)
        delta = v_i - retrieval

        # Rank-1 update: S += beta * k outer delta
        update = jnp.einsum('bhk,bhv->bhkv', k_i, delta) * beta_i[:, :, None, None]
        S = S + update

        # Output: o = S @ q
        o_i = jnp.einsum('bhkv,bhk->bhv', S, q_i)

        return S, o_i

    final_state, outputs = jax.lax.scan(
        scan_step,
        S0,
        (q_t, k_t, v_t, g_t, beta_t),
    )

    # outputs: (T, B, H, D) -> (B, T, H, D)
    outputs = jnp.transpose(outputs, (1, 0, 2, 3))
    return outputs.astype(q.dtype), final_state


def gated_delta_attention_chunk(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    chunk_size: int = 64,
    initial_state: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Chunk-based gated delta attention for better MXU utilization.

    Splits the sequence into chunks of size C, computes intra-chunk
    interactions via materialized C x C attention, and propagates
    inter-chunk state via recurrence over T/C steps.

    This reduces sequential steps from T to T/C and uses matrix operations
    that saturate the MXU within each chunk.

    Args:
        q: (B, T, H, D)
        k: (B, T, H, D)
        v: (B, T, H, D)
        g: (B, T, H, D)  log-space decay
        beta: (B, T, H)
        chunk_size: block size C
        initial_state: (B, H, D, D) optional

    Returns:
        output: (B, T, H, D)
        final_state: (B, H, D, D)
    """
    B, T, H, D = q.shape
    C = chunk_size
    assert T % C == 0, f"seq_len {T} must be divisible by chunk_size {C}"
    num_chunks = T // C

    if initial_state is None:
        S = jnp.zeros((B, H, D, D), dtype=jnp.float32)
    else:
        S = initial_state.astype(jnp.float32)

    # Reshape to (B, num_chunks, C, H, D)
    q_c = q.reshape(B, num_chunks, C, H, D).astype(jnp.float32)
    k_c = k.reshape(B, num_chunks, C, H, D).astype(jnp.float32)
    v_c = v.reshape(B, num_chunks, C, H, D).astype(jnp.float32)
    g_c = g.reshape(B, num_chunks, C, H, D).astype(jnp.float32)
    beta_c = beta.reshape(B, num_chunks, C, H).astype(jnp.float32)

    def process_chunk(S, chunk_inputs):
        q_blk, k_blk, v_blk, g_blk, beta_blk = chunk_inputs
        # q_blk: (B, C, H, D), etc.

        # --- Cumulative decay within chunk ---
        # g_cum[i] = sum(g[0..i]) for prefix-sum of log-decays
        g_cum = jnp.cumsum(g_blk, axis=1)  # (B, C, H, D)

        # Decay from start of chunk to each position
        # decay_from_start[i] = exp(sum(g[0..i]))
        decay_from_start = jnp.exp(g_cum)  # (B, C, H, D)

        # Total chunk decay = exp(sum(g[0..C-1]))
        total_chunk_decay = decay_from_start[:, -1, :, :]  # (B, H, D)

        # --- Inter-chunk: contribution from previous state S ---
        # Decay the previous state to each position within chunk
        # For position i: S_decayed = exp(g_cum[i]) * S
        # inter_output[i] = (exp(g_cum[i])[:,:,None] * S) @ q[i]
        # = sum_v( exp(g_cum[i])[k] * S[k,v] * q[i][k] )
        inter_output = jnp.einsum(
            'bchk,bhkv,bchk->bchv',
            decay_from_start, S,
            q_blk,
        )  # (B, C, H, D)

        # --- Intra-chunk: pairwise interactions within chunk ---
        # For positions i, j where j <= i:
        # contribution of j to i = exp(g_cum[i] - g_cum[j]) * beta[j] * (k[j]^T q[i]) * delta[j]
        # But delta depends on S which evolves... for the intra-chunk part,
        # we use the simpler formulation:
        #
        # Intra attention scores: A[i,j] = q[i]^T @ k[j] * exp(g_cum[i] - g_cum[j]) * beta[j]
        # Causal mask: A[i,j] = 0 for j > i

        # Relative decay: exp(g_cum[i] - g_cum[j]) per head-dim
        # For efficiency, compute via: exp(g_cum[i]) / exp(g_cum[j])
        # But this is element-wise in D, so we need to handle it carefully.

        # Simplified: use scalar decay (sum over D dimension of log-decays)
        # This is an approximation that enables efficient matmul.
        # The exact chunk approach uses the WY representation from FLA.
        g_scalar_cum = jnp.sum(g_cum, axis=-1) / D  # (B, C, H) - averaged log-decay

        # Pairwise relative decay: (B, H, C, C)
        # rel_decay[i,j] = exp(g_scalar_cum[i] - g_scalar_cum[j])
        rel_decay = jnp.exp(
            g_scalar_cum[:, :, :, None] - g_scalar_cum[:, None, :, :]
        )  # (B, C_i, H, C_j) -- but we need (B, H, C, C)
        # Rearrange: (B, C, H) -> for einsum, transpose
        g_sc = jnp.transpose(g_scalar_cum, (0, 2, 1))  # (B, H, C)
        rel_decay = jnp.exp(g_sc[:, :, :, None] - g_sc[:, :, None, :])  # (B, H, C_i, C_j)

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))  # (C, C)
        # Zero out the diagonal (position i doesn't attend to itself via intra)
        # Actually, include diagonal for self-update
        rel_decay = rel_decay * causal_mask[None, None, :, :]

        # QK attention within chunk: (B, H, C, C)
        # q_bhcd: (B, H, C, D), k_bhcd: (B, H, C, D)
        q_h = jnp.transpose(q_blk, (0, 2, 1, 3))  # (B, H, C, D)
        k_h = jnp.transpose(k_blk, (0, 2, 1, 3))  # (B, H, C, D)
        v_h = jnp.transpose(v_blk, (0, 2, 1, 3))  # (B, H, C, D)
        beta_h = jnp.transpose(beta_blk, (0, 2, 1))  # (B, H, C)

        # Attention scores
        attn = jnp.einsum('bhid,bhjd->bhij', q_h, k_h)  # (B, H, C, C)
        attn = attn * rel_decay * beta_h[:, :, None, :]  # scale by beta at source position

        # Intra output: weighted sum of values
        intra_output = jnp.einsum('bhij,bhjd->bhid', attn, v_h)  # (B, H, C, D)
        intra_output = jnp.transpose(intra_output, (0, 2, 1, 3))  # (B, C, H, D)

        # --- Combine ---
        output = inter_output + intra_output

        # --- Update state for next chunk ---
        # Process chunk sequentially to get exact final state
        # (This is O(C) sequential but C is small, e.g. 64)
        def state_step(S_inner, idx):
            i = idx
            k_i = k_blk[:, i, :, :]      # (B, H, D)
            v_i = v_blk[:, i, :, :]      # (B, H, D)
            g_i = g_blk[:, i, :, :]      # (B, H, D)
            b_i = beta_blk[:, i, :]      # (B, H)

            decay = jnp.exp(g_i)
            S_inner = S_inner * decay[:, :, :, None]
            retrieval = jnp.einsum('bhk,bhkv->bhv', k_i, S_inner)
            delta = v_i - retrieval
            update = jnp.einsum('bhk,bhv->bhkv', k_i, delta) * b_i[:, :, None, None]
            S_inner = S_inner + update
            return S_inner, None

        S_next, _ = jax.lax.scan(state_step, S, jnp.arange(C))

        return S_next, output

    # Transpose chunks for scan: (num_chunks, B, C, H, D)
    q_scan = jnp.transpose(q_c, (1, 0, 2, 3, 4))
    k_scan = jnp.transpose(k_c, (1, 0, 2, 3, 4))
    v_scan = jnp.transpose(v_c, (1, 0, 2, 3, 4))
    g_scan = jnp.transpose(g_c, (1, 0, 2, 3, 4))
    beta_scan = jnp.transpose(beta_c, (1, 0, 2, 3))

    final_state, all_outputs = jax.lax.scan(
        process_chunk,
        S,
        (q_scan, k_scan, v_scan, g_scan, beta_scan),
    )

    # all_outputs: (num_chunks, B, C, H, D) -> (B, T, H, D)
    all_outputs = jnp.transpose(all_outputs, (1, 0, 2, 3, 4))
    all_outputs = all_outputs.reshape(B, T, H, D)

    return all_outputs.astype(q.dtype), final_state


# ---------------------------------------------------------------------------
# KDA Layer (full layer with projections)
# ---------------------------------------------------------------------------

class KDALayerParams(NamedTuple):
    """Parameters for a single KDA attention layer."""
    q_proj_w: jnp.ndarray    # (hidden, num_heads * head_dim)
    k_proj_w: jnp.ndarray    # (hidden, num_heads * head_dim)
    v_proj_w: jnp.ndarray    # (hidden, num_heads * head_dim)
    o_proj_w: jnp.ndarray    # (num_heads * head_dim, hidden)
    gate_a_w: jnp.ndarray    # (hidden, head_dim)
    gate_b_w: jnp.ndarray    # (head_dim, num_heads * head_dim)
    beta_proj_w: jnp.ndarray # (hidden, num_heads)
    a_log: jnp.ndarray       # (num_heads,)
    dt_bias: jnp.ndarray     # (num_heads * head_dim,)
    norm_w: jnp.ndarray      # (head_dim,)


def init_kda_layer(key, config: KimiLinearConfig) -> KDALayerParams:
    """Initialize KDA layer parameters."""
    h = config.hidden_size
    d = config.head_dim
    nh = config.num_heads
    proj = nh * d

    keys = jax.random.split(key, 10)
    scale = 1.0 / math.sqrt(h)

    return KDALayerParams(
        q_proj_w=jax.random.normal(keys[0], (h, proj)) * scale,
        k_proj_w=jax.random.normal(keys[1], (h, proj)) * scale,
        v_proj_w=jax.random.normal(keys[2], (h, proj)) * scale,
        o_proj_w=jax.random.normal(keys[3], (proj, h)) * scale,
        gate_a_w=jax.random.normal(keys[4], (h, d)) * scale,
        gate_b_w=jax.random.normal(keys[5], (d, proj)) * scale,
        beta_proj_w=jax.random.normal(keys[6], (h, nh)) * scale,
        a_log=jnp.log(jax.random.uniform(keys[7], (nh,), minval=1.0, maxval=16.0)),
        dt_bias=jnp.zeros((proj,)),
        norm_w=jnp.ones((d,)),
    )


def kda_layer_forward(
    params: KDALayerParams,
    x: jnp.ndarray,
    config: KimiLinearConfig,
    state: Optional[jnp.ndarray] = None,
    use_chunk: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass for a single KDA layer.

    Args:
        params: KDA layer parameters
        x: (B, T, H_model) input hidden states
        config: model config
        state: optional recurrent state (B, num_heads, head_dim, head_dim)
        use_chunk: if True use chunk-based, else naive scan

    Returns:
        output: (B, T, H_model)
        new_state: (B, num_heads, head_dim, head_dim)
    """
    B, T, _ = x.shape
    nh = config.num_heads
    d = config.head_dim

    # Projections
    q = x @ params.q_proj_w                          # (B, T, nh*d)
    k = x @ params.k_proj_w
    v = x @ params.v_proj_w

    # Reshape to (B, T, H, D)
    q = q.reshape(B, T, nh, d)
    k = k.reshape(B, T, nh, d)
    v = v.reshape(B, T, nh, d)

    # Gating: g = softplus(gate_b(silu(gate_a(x))) + dt_bias) * (-a_log.exp())
    g_hidden = silu(x @ params.gate_a_w)             # (B, T, d)
    g_proj = g_hidden @ params.gate_b_w              # (B, T, nh*d)
    g_proj = g_proj + params.dt_bias[None, None, :]
    g_proj = jax.nn.softplus(g_proj)                  # ensure positive
    g_proj = g_proj.reshape(B, T, nh, d)
    # Apply learned decay rate
    a = -jnp.exp(params.a_log)                        # (nh,) negative
    g = g_proj * a[None, None, :, None]                # (B, T, H, D) negative log-decay

    # Beta: sigmoid gating per head
    beta = jax.nn.sigmoid(x @ params.beta_proj_w)     # (B, T, nh)

    # Run attention
    if use_chunk and T >= 64:
        chunk_size = min(64, T)
        o, new_state = gated_delta_attention_chunk(q, k, v, g, beta, chunk_size, state)
    else:
        o, new_state = gated_delta_attention_naive(q, k, v, g, beta, state)

    # Output projection
    o = o.reshape(B, T, nh * d)
    o = o @ params.o_proj_w

    return o, new_state
