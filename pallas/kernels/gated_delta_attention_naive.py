"""
AutoKernel Pallas Gated Delta Attention -- Naive scan() baseline.

Copy this file's content to kernel.py at project root to benchmark via:
    python tpu_vm.py bench-pallas --project <PROJECT> --kernel-type gated_delta_attention

This is the NAIVE BASELINE using jax.lax.scan() for sequential recurrence.
It is correct but slow -- each timestep is processed sequentially, resulting
in small per-step operations that don't saturate the TPU MXU.

The optimized chunk-based version is in gated_delta_attention.py.

Algorithm (Gated Delta Attention / KDA):
    For each timestep t:
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t outer (v_t - k_t^T @ S_{t-1})
        o_t = S_t @ q_t

    Where:
        S: (H, D, D) recurrent state per head
        g: per-dim exponential decay factors (log-space)
        beta: per-head scalar gating
        "k_t outer delta" is the delta rule update

Reference: Kimi Linear (arXiv:2510.26692), Gated DeltaNet (arXiv:2412.06464)

Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass
"""

KERNEL_TYPE = "gated_delta_attention"
BACKEND = "pallas"

import jax
import jax.numpy as jnp


def kernel_fn(q, k, v, g, beta):
    """Naive gated delta attention via jax.lax.scan.

    Args:
        q: (B, T, H, D) queries
        k: (B, T, H, D) keys
        v: (B, T, H, D) values
        g: (B, T, H, D) per-dim decay (log-space, typically negative)
        beta: (B, T, H) per-head gating scalar in [0, 1]

    Returns:
        output: (B, T, H, D)
    """
    B, T, H, D = q.shape

    S0 = jnp.zeros((B, H, D, D), dtype=jnp.float32)

    # Transpose to (T, B, ...) for scan
    q_t = jnp.transpose(q, (1, 0, 2, 3)).astype(jnp.float32)
    k_t = jnp.transpose(k, (1, 0, 2, 3)).astype(jnp.float32)
    v_t = jnp.transpose(v, (1, 0, 2, 3)).astype(jnp.float32)
    g_t = jnp.transpose(g, (1, 0, 2, 3)).astype(jnp.float32)
    beta_t = jnp.transpose(beta, (1, 0, 2)).astype(jnp.float32)

    def scan_step(S, inputs):
        q_i, k_i, v_i, g_i, beta_i = inputs

        # Exponential decay
        decay = jnp.exp(g_i)  # (B, H, D)
        S = S * decay[:, :, :, None]

        # Delta rule: retrieval = k^T @ S -> (B, H, D_v)
        retrieval = jnp.einsum('bhk,bhkv->bhv', k_i, S)
        delta = v_i - retrieval

        # Rank-1 update: S += beta * k outer delta
        update = jnp.einsum('bhk,bhv->bhkv', k_i, delta) * beta_i[:, :, None, None]
        S = S + update

        # Output: o = S @ q
        o_i = jnp.einsum('bhkv,bhk->bhv', S, q_i)

        return S, o_i

    _, outputs = jax.lax.scan(scan_step, S0, (q_t, k_t, v_t, g_t, beta_t))

    # (T, B, H, D) -> (B, T, H, D)
    return jnp.transpose(outputs, (1, 0, 2, 3)).astype(q.dtype)
