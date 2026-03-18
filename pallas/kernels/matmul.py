"""
AutoKernel Pallas Matmul Kernel -- Starter implementation for TPU.

Copy this file's content to kernel.py at project root to benchmark via:
    python tpu_vm.py bench-pallas --project <PROJECT>

This file contains TWO implementations:
  1. _baseline_kernel_fn: jax.jit wrapper around jnp.matmul (for loop validation)
  2. _pallas_kernel_fn: Pallas pallas_call with tiled MXU kernel (primary)

The default kernel_fn points to the Pallas implementation. To switch to the
baseline for loop validation, change the assignment at the bottom of this file.

Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass
"""

KERNEL_TYPE = "matmul"
BACKEND = "pallas"

import jax
import jax.numpy as jnp
import functools

# ---------------------------------------------------------------------------
# Baseline: jax.jit wrapper (validates the bench loop without Pallas)
# ---------------------------------------------------------------------------


@jax.jit
def _baseline_kernel_fn(A, B):
    """Simple jnp.matmul baseline. Use this to validate the experiment loop."""
    return jnp.matmul(A, B)


# ---------------------------------------------------------------------------
# Pallas: tiled matmul using pallas_call
# ---------------------------------------------------------------------------

try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    _PALLAS_AVAILABLE = True
except ImportError:
    _PALLAS_AVAILABLE = False


def _matmul_kernel_body(a_ref, b_ref, o_ref):
    """Pallas kernel body: compute C tile = A tile @ B tile."""
    o_ref[...] = jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)


def _pallas_kernel_fn(A, B):
    """Tiled matmul via jax.experimental.pallas.pallas_call.

    Uses 128x128 tiles aligned to TPU MXU dimensions. The grid iterates
    over output tiles; each kernel invocation computes one (BLOCK_M, BLOCK_N)
    tile by accumulating over the K dimension.

    For simplicity this starter kernel does NOT tile the K dimension --
    it reads the full K extent per tile. This is correct but suboptimal
    for large K; the agent should add K-tiling as a first optimization.
    """
    if not _PALLAS_AVAILABLE:
        return _baseline_kernel_fn(A, B)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Shape mismatch: A is {A.shape}, B is {B.shape}"

    # Block sizes aligned to TPU MXU (128x128)
    BLOCK_M = min(128, M)
    BLOCK_N = min(128, N)
    BLOCK_K = K  # Full K -- no K-tiling in starter kernel

    grid_m = M // BLOCK_M
    grid_n = N // BLOCK_N

    # Ensure shapes are divisible (pad if needed)
    assert M % BLOCK_M == 0, f"M={M} not divisible by BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} not divisible by BLOCK_N={BLOCK_N}"

    out = pl.pallas_call(
        _matmul_kernel_body,
        out_shape=jax.ShapeDtypeStruct((BLOCK_M, BLOCK_N), A.dtype),
        grid=(grid_m, grid_n),
        in_specs=[
            pl.BlockSpec((BLOCK_M, BLOCK_K), lambda i, j: (i, 0)),
            pl.BlockSpec((BLOCK_K, BLOCK_N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
    )(A, B)
    return out


# ---------------------------------------------------------------------------
# Default kernel_fn -- change this to _baseline_kernel_fn for loop validation
# ---------------------------------------------------------------------------

kernel_fn = _pallas_kernel_fn
