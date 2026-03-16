#!/usr/bin/env python3
"""
pallas/bench.py -- AutoKernel JAX/Pallas benchmark harness.

Analogous to bench.py for CUDA/Triton, but targets JAX/Pallas kernels on TPU
(and CPU/GPU as fallback).

Handles:
  1. TPU hardware detection and roofline modelling (via pallas.profiler)
  2. Correctness verification (compare kernel_fn against JAX reference)
  3. Performance benchmarking (warm/cold timing with block_until_ready)
  4. Structured, greppable output matching the CUDA bench format

Usage:
  python pallas/bench.py                    # benchmark kernel.py using its KERNEL_TYPE
  python pallas/bench.py --kernel matmul    # force kernel type
  python pallas/bench.py --quick            # smoke test only, bench large size
  python pallas/bench.py --sizes large      # benchmark only 'large' size
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy JAX import -- module remains importable without JAX installed
# ---------------------------------------------------------------------------

_jax = None
_jnp = None
_jax_nn = None


def _import_jax():
    global _jax, _jnp, _jax_nn
    if _jax is None:
        try:
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore
            import jax.nn as jax_nn  # type: ignore
            _jax = jax
            _jnp = jnp
            _jax_nn = jax_nn
        except ImportError as exc:
            raise RuntimeError(
                "JAX is not installed. Install jax before using pallas/bench.py."
            ) from exc
    return _jax, _jnp, _jax_nn


# ---------------------------------------------------------------------------
# TPU hardware detection (delegates to pallas.profiler)
# ---------------------------------------------------------------------------

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# Allow running as a script from any cwd by ensuring the package root is on path
_PROJECT_ROOT = SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pallas.profiler import TPUSpec, get_tpu_spec, compute_roofline  # noqa: E402


def detect_tpu() -> Optional[TPUSpec]:
    """Attempt to detect the current TPU and return its spec.

    Returns None if running on CPU/GPU or if TPU generation is unknown.
    """
    try:
        spec = get_tpu_spec()  # auto-detect
        return spec
    except (KeyError, RuntimeError):
        return None


# =========================================================================
# Input generators (deterministic via JAX PRNGKey)
# =========================================================================

def _key(seed: int = 42):
    jax, jnp, _ = _import_jax()
    return jax.random.PRNGKey(seed)


def gen_matmul_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2 = jax.random.split(_key(seed))
    M, N, K = size["M"], size["N"], size["K"]
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype)
    return {"A": A, "B": B}


def gen_softmax_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    rows, cols = size["rows"], size["cols"]
    x = jax.random.normal(_key(seed), (rows, cols), dtype=dtype)
    return {"x": x}


def gen_layernorm_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2, k3 = jax.random.split(_key(seed), 3)
    batch, dim = size["batch"], size["dim"]
    x = jax.random.normal(k1, (batch, dim), dtype=dtype)
    weight = jnp.ones((dim,), dtype=dtype)
    bias = jnp.zeros((dim,), dtype=dtype)
    return {"x": x, "weight": weight, "bias": bias}


def gen_flash_attention_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2, k3 = jax.random.split(_key(seed), 3)
    batch, heads, seq_len, head_dim = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
    shape = (batch, heads, seq_len, head_dim)
    Q = jax.random.normal(k1, shape, dtype=dtype)
    K = jax.random.normal(k2, shape, dtype=dtype)
    V = jax.random.normal(k3, shape, dtype=dtype)
    return {"Q": Q, "K": K, "V": V}


def gen_fused_mlp_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2, k3, k4 = jax.random.split(_key(seed), 4)
    batch, dim, hidden = size["batch"], size["dim"], size["hidden"]
    x = jax.random.normal(k1, (batch, dim), dtype=dtype)
    w_gate = jax.random.normal(k2, (hidden, dim), dtype=dtype) * 0.02
    w_up = jax.random.normal(k3, (hidden, dim), dtype=dtype) * 0.02
    w_down = jax.random.normal(k4, (dim, hidden), dtype=dtype) * 0.02
    return {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def gen_cross_entropy_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2 = jax.random.split(_key(seed))
    batch, vocab = size["batch"], size["vocab"]
    logits = jax.random.normal(k1, (batch, vocab), dtype=dtype)
    targets = jax.random.randint(k2, (batch,), 0, vocab)
    return {"logits": logits, "targets": targets}


def gen_rotary_embedding_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2, k3 = jax.random.split(_key(seed), 3)
    batch, heads, seq_len, head_dim = size["batch"], size["heads"], size["seq_len"], size["head_dim"]
    half_dim = head_dim // 2
    x = jax.random.normal(k1, (batch, heads, seq_len, head_dim), dtype=dtype)
    cos = jax.random.normal(k2, (seq_len, half_dim), dtype=dtype)
    sin = jax.random.normal(k3, (seq_len, half_dim), dtype=dtype)
    return {"x": x, "cos": cos, "sin": sin}


def gen_rmsnorm_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    k1, k2 = jax.random.split(_key(seed))
    M, N = size["M"], size["N"]
    x = jax.random.normal(k1, (M, N), dtype=dtype)
    weight = jax.random.normal(k2, (N,), dtype=dtype)
    return {"x": x, "weight": weight}


def gen_reduce_inputs(size: dict, dtype: Any, seed: int = 42) -> dict:
    jax, jnp, _ = _import_jax()
    M, N = size["M"], size["N"]
    x = jax.random.normal(_key(seed), (M, N), dtype=dtype)
    return {"x": x}


# =========================================================================
# JAX reference implementations
# =========================================================================

def _ref_matmul(inputs: dict) -> Any:
    _, jnp, _ = _import_jax()
    return jnp.matmul(inputs["A"], inputs["B"])


def _ref_softmax(inputs: dict) -> Any:
    _, _, jax_nn = _import_jax()
    return jax_nn.softmax(inputs["x"], axis=-1)


def _ref_layernorm(inputs: dict) -> Any:
    jax, jnp, _ = _import_jax()
    x, weight, bias = inputs["x"], inputs["weight"], inputs["bias"]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
    return x_norm * weight + bias


def _ref_flash_attention(inputs: dict) -> Any:
    _, jnp, jax_nn = _import_jax()
    Q, K, V = inputs["Q"], inputs["K"], inputs["V"]
    head_dim = Q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    # Q: (batch, heads, seq, head_dim)
    # scores: (batch, heads, seq, seq)
    scores = jnp.einsum("bhsd,bhtd->bhst", Q, K) * scale
    weights = jax_nn.softmax(scores, axis=-1)
    return jnp.einsum("bhst,bhtd->bhsd", weights, V)


def _ref_fused_mlp(inputs: dict) -> Any:
    jax, jnp, jax_nn = _import_jax()
    x, w_gate, w_up, w_down = inputs["x"], inputs["w_gate"], inputs["w_up"], inputs["w_down"]
    gate = jax_nn.silu(x @ w_gate.T)
    up = x @ w_up.T
    hidden = gate * up
    return hidden @ w_down.T


def _ref_cross_entropy(inputs: dict) -> Any:
    jax, jnp, jax_nn = _import_jax()
    logits, targets = inputs["logits"], inputs["targets"]
    log_probs = jax_nn.log_softmax(logits, axis=-1)
    batch = logits.shape[0]
    # gather log-probs at target indices
    nll = -log_probs[jnp.arange(batch), targets]
    return jnp.mean(nll)


def _ref_rotary_embedding(inputs: dict) -> Any:
    _, jnp, _ = _import_jax()
    x, cos, sin = inputs["x"], inputs["cos"], inputs["sin"]
    # x: (batch, heads, seq, head_dim)
    # cos/sin: (seq, half_dim)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # broadcast cos/sin to (1, 1, seq, half_dim)
    cos = cos[jnp.newaxis, jnp.newaxis, :, :]
    sin = sin[jnp.newaxis, jnp.newaxis, :, :]
    rotated = jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rotated


def _ref_rmsnorm(inputs: dict) -> Any:
    _, jnp, _ = _import_jax()
    x, weight = inputs["x"], inputs["weight"]
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-6)
    return (x / rms) * weight


def _ref_reduce(inputs: dict) -> Any:
    _, jnp, _ = _import_jax()
    return jnp.sum(inputs["x"], axis=-1)


# =========================================================================
# Kernel configs
# =========================================================================

def _dtype_bytes(dtype: Any) -> int:
    """Return element byte-width for a JAX dtype."""
    _, jnp, _ = _import_jax()
    return jnp.dtype(dtype).itemsize


KERNEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "matmul": {
        "test_sizes": [
            ("small",   {"M": 512,  "N": 512,  "K": 512}),
            ("medium",  {"M": 1024, "N": 1024, "K": 1024}),
            ("large",   {"M": 2048, "N": 2048, "K": 2048}),
            ("llm_mlp", {"M": 4096, "N": 11008, "K": 4096}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
            "float32":  {"atol": 1e-4, "rtol": 1e-4},
        },
        "flops_fn": lambda s: 2 * s["M"] * s["N"] * s["K"],
        "bytes_fn": lambda s, dt: (s["M"] * s["K"] + s["K"] * s["N"] + s["M"] * s["N"]) * _dtype_bytes(dt),
        "input_generator": gen_matmul_inputs,
        "reference_fn": _ref_matmul,
    },
    "softmax": {
        "test_sizes": [
            ("small",  {"rows": 256,  "cols": 512}),
            ("medium", {"rows": 1024, "cols": 1024}),
            ("large",  {"rows": 4096, "cols": 4096}),
            ("vocab",  {"rows": 4096, "cols": 32000}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
            "float32":  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 5 * s["rows"] * s["cols"],
        "bytes_fn": lambda s, dt: 2 * s["rows"] * s["cols"] * _dtype_bytes(dt),
        "input_generator": gen_softmax_inputs,
        "reference_fn": _ref_softmax,
    },
    "layernorm": {
        "test_sizes": [
            ("small",  {"batch": 256,  "dim": 512}),
            ("medium", {"batch": 1024, "dim": 1024}),
            ("large",  {"batch": 4096, "dim": 2048}),
            ("llm_7b", {"batch": 4096, "dim": 4096}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
            "float32":  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 8 * s["batch"] * s["dim"],
        "bytes_fn": lambda s, dt: (2 * s["batch"] * s["dim"] + 2 * s["dim"]) * _dtype_bytes(dt),
        "input_generator": gen_layernorm_inputs,
        "reference_fn": _ref_layernorm,
    },
    "flash_attention": {
        "test_sizes": [
            ("small",  {"batch": 2, "heads": 8,  "seq_len": 256,  "head_dim": 64}),
            ("medium", {"batch": 2, "heads": 16, "seq_len": 512,  "head_dim": 64}),
            ("large",  {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 64}),
            ("llm_7b", {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
        ],
        "test_dtypes": ["bfloat16"],
        "tolerances": {
            "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
        },
        "flops_fn": lambda s: 4 * s["batch"] * s["heads"] * (s["seq_len"] ** 2) * s["head_dim"],
        "bytes_fn": lambda s, dt: 4 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * _dtype_bytes(dt),
        "input_generator": gen_flash_attention_inputs,
        "reference_fn": _ref_flash_attention,
    },
    "fused_mlp": {
        "test_sizes": [
            ("small",  {"batch": 256,  "dim": 512,  "hidden": 1024}),
            ("medium", {"batch": 1024, "dim": 1024, "hidden": 2048}),
            ("large",  {"batch": 2048, "dim": 2048, "hidden": 5504}),
            ("llm_7b", {"batch": 2048, "dim": 4096, "hidden": 11008}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
            "float32":  {"atol": 1e-4, "rtol": 1e-4},
        },
        "flops_fn": lambda s: 2 * s["batch"] * s["dim"] * s["hidden"] * 3,
        "bytes_fn": lambda s, dt: (s["batch"] * s["dim"] + s["hidden"] * s["dim"] * 3 + s["batch"] * s["dim"]) * _dtype_bytes(dt),
        "input_generator": gen_fused_mlp_inputs,
        "reference_fn": _ref_fused_mlp,
    },
    "cross_entropy": {
        "test_sizes": [
            ("small",  {"batch": 256,  "vocab": 1024}),
            ("medium", {"batch": 1024, "vocab": 4096}),
            ("large",  {"batch": 4096, "vocab": 32000}),
            ("gpt2",   {"batch": 4096, "vocab": 50257}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-2, "rtol": 2e-2},
            "float32":  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 4 * s["batch"] * s["vocab"],
        "bytes_fn": lambda s, dt: (s["batch"] * s["vocab"] + s["batch"]) * _dtype_bytes(dt),
        "input_generator": gen_cross_entropy_inputs,
        "reference_fn": _ref_cross_entropy,
    },
    "rotary_embedding": {
        "test_sizes": [
            ("small",  {"batch": 2, "heads": 8,  "seq_len": 256,  "head_dim": 64}),
            ("medium", {"batch": 2, "heads": 16, "seq_len": 512,  "head_dim": 64}),
            ("large",  {"batch": 2, "heads": 32, "seq_len": 1024, "head_dim": 128}),
            ("llm_7b", {"batch": 1, "heads": 32, "seq_len": 2048, "head_dim": 128}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 2e-3, "rtol": 2e-3},
            "float32":  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: 6 * s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"],
        "bytes_fn": lambda s, dt: (s["batch"] * s["heads"] * s["seq_len"] * s["head_dim"] * 2 +
                                    s["seq_len"] * s["head_dim"]) * _dtype_bytes(dt),
        "input_generator": gen_rotary_embedding_inputs,
        "reference_fn": _ref_rotary_embedding,
    },
    "rmsnorm": {
        "test_sizes": [
            ("small",  {"M": 1024, "N": 768}),
            ("medium", {"M": 4096, "N": 1024}),
            ("large",  {"M": 4096, "N": 4096}),
            ("llama",  {"M": 2048, "N": 4096}),
        ],
        "test_dtypes": ["bfloat16"],
        "tolerances": {
            "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
        },
        "flops_fn": lambda s: 6 * s["M"] * s["N"],
        "bytes_fn": lambda s, dt: (2 * s["M"] * s["N"] + s["N"]) * _dtype_bytes(dt),
        "input_generator": gen_rmsnorm_inputs,
        "reference_fn": _ref_rmsnorm,
    },
    "reduce": {
        "test_sizes": [
            ("small",  {"M": 1024, "N": 1024}),
            ("medium", {"M": 4096, "N": 4096}),
            ("large",  {"M": 8192, "N": 8192}),
            ("wide",   {"M": 1024, "N": 32768}),
        ],
        "test_dtypes": ["bfloat16", "float32"],
        "tolerances": {
            "bfloat16": {"atol": 1e-2, "rtol": 1e-2},
            "float32":  {"atol": 1e-5, "rtol": 1e-5},
        },
        "flops_fn": lambda s: s["M"] * s["N"],
        "bytes_fn": lambda s, dt: (s["M"] * s["N"] + s["M"]) * _dtype_bytes(dt),
        "input_generator": gen_reduce_inputs,
        "reference_fn": _ref_reduce,
    },
}


# =========================================================================
# Correctness verification
# =========================================================================

def _has_nan_inf(arr: Any) -> bool:
    jax, jnp, _ = _import_jax()
    arr = jnp.asarray(arr)
    return bool(jnp.any(jnp.isnan(arr)) or jnp.any(jnp.isinf(arr)))


def _allclose(out: Any, ref: Any, atol: float, rtol: float) -> Tuple[bool, float, float]:
    """Return (match, max_abs_error, mean_abs_error)."""
    jax, jnp, _ = _import_jax()
    out_f = jnp.asarray(out, dtype=jnp.float32)
    ref_f = jnp.asarray(ref, dtype=jnp.float32)
    abs_diff = jnp.abs(out_f - ref_f)
    max_err = float(jnp.max(abs_diff))
    mean_err = float(jnp.mean(abs_diff))
    match = bool(jnp.all(abs_diff <= atol + rtol * jnp.abs(ref_f)))
    return match, max_err, mean_err


def run_correctness(kernel_fn: Callable, config: dict, quick: bool = False) -> dict:
    """Smoke test + shape sweep correctness check.

    Returns a dict with 'correctness': 'PASS' or 'FAIL' and stage details.
    In quick mode only runs smoke test (first size, first dtype).
    """
    jax, jnp, _ = _import_jax()

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    sizes = config["test_sizes"]
    dtypes = config["test_dtypes"]
    tols = config["tolerances"]

    results: Dict[str, Any] = {
        "smoke_test": "SKIP",
        "shape_sweep": "SKIP",
        "correctness": "FAIL",
    }
    all_pass = True

    # ------------------------------------------------------------------
    # Stage 1: Smoke test
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Smoke Test ---")
    dtype_str = dtypes[0]
    try:
        dtype = jnp.dtype(dtype_str)
        label, sz = sizes[0]
        inputs = gen_fn(sz, dtype, seed=42)
        ref_out = ref_fn(inputs)
        kernel_out = kernel_fn(**inputs)
        if hasattr(kernel_out, "block_until_ready"):
            kernel_out.block_until_ready()

        if _has_nan_inf(kernel_out):
            results["smoke_test"] = "FAIL"
            all_pass = False
            print("  FAIL: NaN/Inf in output")
        else:
            tol = tols.get(dtype_str, {"atol": 1e-2, "rtol": 1e-2})
            match, max_err, _ = _allclose(kernel_out, ref_out, **tol)
            if match:
                results["smoke_test"] = "PASS"
                print(f"  PASS (max_abs_error={max_err:.6e})")
            else:
                results["smoke_test"] = "FAIL"
                all_pass = False
                print(f"  FAIL: max_abs_error={max_err:.6e} exceeds tol(atol={tol['atol']}, rtol={tol['rtol']})")
    except Exception as exc:
        results["smoke_test"] = "FAIL"
        all_pass = False
        print(f"  FAIL: CRASH ({type(exc).__name__}: {exc})")

    if results["smoke_test"] == "FAIL":
        results["correctness"] = "FAIL"
        print(f"\ncorrectness: FAIL (smoke test failed)")
        return results

    if quick:
        results["shape_sweep"] = "SKIP (quick mode)"
        results["correctness"] = "PASS"
        print(f"\ncorrectness: PASS (quick mode: shape sweep skipped)")
        return results

    # ------------------------------------------------------------------
    # Stage 2: Shape sweep
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Shape Sweep ---")
    sweep_pass = True
    sweep_count = 0
    sweep_fail = 0
    worst_err = 0.0
    worst_case = ""

    for label, sz in sizes:
        for dtype_str in dtypes:
            sweep_count += 1
            try:
                dtype = jnp.dtype(dtype_str)
                inputs = gen_fn(sz, dtype, seed=42)
                ref_out = ref_fn(inputs)
                kernel_out = kernel_fn(**inputs)
                if hasattr(kernel_out, "block_until_ready"):
                    kernel_out.block_until_ready()

                if _has_nan_inf(kernel_out):
                    sweep_pass = False
                    sweep_fail += 1
                    print(f"  FAIL: {label}/{dtype_str} -> NaN/Inf")
                    continue

                tol = tols.get(dtype_str, {"atol": 1e-2, "rtol": 1e-2})
                match, max_err, _ = _allclose(kernel_out, ref_out, **tol)

                if max_err > worst_err:
                    worst_err = max_err
                    worst_case = f"{label}/{dtype_str}"

                if match:
                    print(f"  PASS: {label} {dtype_str} (max_err={max_err:.2e})")
                else:
                    sweep_pass = False
                    sweep_fail += 1
                    print(f"  FAIL: {label} {dtype_str} -> max_err={max_err:.2e}")
            except Exception as exc:
                sweep_pass = False
                sweep_fail += 1
                print(f"  FAIL: {label}/{dtype_str} -> {type(exc).__name__}: {exc}")

    if sweep_pass:
        results["shape_sweep"] = f"PASS ({sweep_count} configs, worst_err={worst_err:.2e} at {worst_case})"
        print(f"  shape_sweep: PASS ({sweep_count} configs)")
    else:
        results["shape_sweep"] = f"FAIL ({sweep_fail}/{sweep_count} failed)"
        all_pass = False
        print(f"  shape_sweep: FAIL ({sweep_fail}/{sweep_count} failed)")

    results["correctness"] = "PASS" if all_pass else "FAIL"
    print(f"\ncorrectness: {results['correctness']}")
    return results


# =========================================================================
# Performance benchmarking
# =========================================================================

def _bench_fn(fn: Callable, warmup: int = 5, rep: int = 20) -> float:
    """Benchmark fn() and return median latency in milliseconds.

    Uses JAX's block_until_ready for accurate device-side timing.
    """
    jax, jnp, _ = _import_jax()

    # Warmup
    for _ in range(warmup):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    # Measure
    times: List[float] = []
    for _ in range(rep):
        t0 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]  # median


def run_performance(
    kernel_fn: Callable,
    config: dict,
    tpu_spec: Optional[TPUSpec],
    sizes_filter: str = "all",
) -> dict:
    """Run performance benchmarks. Returns primary result and all-sizes list."""
    jax, jnp, _ = _import_jax()

    gen_fn = config["input_generator"]
    ref_fn = config["reference_fn"]
    flops_fn = config["flops_fn"]
    bytes_fn = config["bytes_fn"]
    dtypes = config["test_dtypes"]
    sizes = config["test_sizes"]

    # Select which sizes to bench
    if sizes_filter == "all":
        bench_sizes = sizes
    else:
        bench_sizes = [(l, s) for l, s in sizes if l == sizes_filter]
        if not bench_sizes:
            bench_sizes = [(l, s) for l, s in sizes if l == "large"] or [sizes[-1]]

    # Primary size for summary output
    primary_label = next((l for l, _ in sizes if l == "large"), sizes[-1][0])

    dtype_str = dtypes[0]
    dtype = jnp.dtype(dtype_str)

    all_results: List[Dict[str, Any]] = []
    primary_result: Optional[Dict[str, Any]] = None

    for label, sz in bench_sizes:
        print(f"\n  Benchmarking: {label} ...")
        try:
            flops = flops_fn(sz)
            nbytes = bytes_fn(sz, dtype_str)

            inputs = gen_fn(sz, dtype, seed=42)

            kernel_ms = _bench_fn(lambda: kernel_fn(**inputs))
            ref_ms = _bench_fn(lambda: ref_fn(inputs))

            kernel_us = kernel_ms * 1000.0
            ref_us = ref_ms * 1000.0
            throughput_tflops = flops / (kernel_ms / 1000.0) / 1e12 if kernel_ms > 0 else 0.0
            bandwidth_gb_s = nbytes / (kernel_ms / 1000.0) / 1e9 if kernel_ms > 0 else 0.0
            ref_tflops = flops / (ref_ms / 1000.0) / 1e12 if ref_ms > 0 else 0.0
            speedup = ref_ms / kernel_ms if kernel_ms > 0 else 0.0

            # Roofline
            if tpu_spec is not None:
                roofline = compute_roofline(flops, nbytes, tpu_spec, measured_latency_ms=kernel_ms)
                bottleneck = roofline["bottleneck"].replace("_", "-")
                pct_peak_compute = roofline.get("pct_peak_compute", 0.0)
                pct_peak_bandwidth = roofline.get("pct_peak_bandwidth", 0.0)
                arithmetic_intensity = roofline["arithmetic_intensity"]
                ridge_point = roofline["ridge_point"]
            else:
                # Fallback: no TPU spec available
                arithmetic_intensity = flops / nbytes if nbytes > 0 else 0.0
                ridge_point = 0.0
                bottleneck = "unknown"
                pct_peak_compute = 0.0
                pct_peak_bandwidth = 0.0

            entry: Dict[str, Any] = {
                "label": label,
                "size": sz,
                "dtype": dtype_str,
                "flops": flops,
                "bytes": nbytes,
                "kernel_latency_us": kernel_us,
                "ref_latency_us": ref_us,
                "throughput_tflops": throughput_tflops,
                "bandwidth_gb_s": bandwidth_gb_s,
                "ref_throughput_tflops": ref_tflops,
                "pct_peak_compute": pct_peak_compute,
                "pct_peak_bandwidth": pct_peak_bandwidth,
                "arithmetic_intensity": arithmetic_intensity,
                "ridge_point": ridge_point,
                "bottleneck": bottleneck,
                "speedup_vs_ref": speedup,
            }
            all_results.append(entry)

            if label == primary_label:
                primary_result = entry

            print(
                f"    kernel: {kernel_us:.2f} us | ref: {ref_us:.2f} us | "
                f"speedup: {speedup:.3f}x | {throughput_tflops:.3f} TFLOPS | "
                f"{pct_peak_compute:.1f}% peak"
            )

        except Exception as exc:
            print(f"    ERROR: {label} -> {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if primary_result is None and all_results:
        primary_result = all_results[-1]

    return {"primary": primary_result, "all": all_results}


# =========================================================================
# CLI main
# =========================================================================

def main() -> int:
    t_start = time.time()

    parser = argparse.ArgumentParser(description="AutoKernel JAX/Pallas benchmark harness")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Kernel type to benchmark (default: read KERNEL_TYPE from kernel.py)")
    parser.add_argument("--sizes", type=str, default="all",
                        help="Which sizes to benchmark: small|medium|large|all (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: smoke test only, bench large size")
    parser.add_argument("--tpu-gen", type=str, default=None,
                        help="Override TPU generation (e.g. v4, v5e, v6e)")
    args = parser.parse_args()

    print("=" * 60)
    print("AutoKernel JAX/Pallas Benchmark Harness")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Import JAX early so errors surface immediately
    # ------------------------------------------------------------------
    try:
        jax, jnp, _ = _import_jax()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        return 1

    # ------------------------------------------------------------------
    # Load kernel module
    # ------------------------------------------------------------------
    kernel_fn = None
    kernel_type = args.kernel

    try:
        cwd = os.getcwd()
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        script_dir = str(pathlib.Path(__file__).resolve().parent.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        kernel_module = importlib.import_module("kernel")
        kernel_fn = kernel_module.kernel_fn

        if kernel_type is None:
            kernel_type = getattr(kernel_module, "KERNEL_TYPE", None)
            if kernel_type is None:
                print("ERROR: kernel.py has no KERNEL_TYPE attribute and --kernel not specified")
                print("\ncorrectness: FAIL")
                print("throughput_tflops: 0.000")
                return 1

        print(f"kernel_type: {kernel_type}")
        print("kernel_module: kernel.py loaded successfully")

    except SyntaxError as exc:
        print(f"\nERROR: kernel.py has a syntax error:\n  {exc}")
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        return 1
    except Exception as exc:
        print(f"\nERROR: Failed to import kernel.py:\n  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        return 1

    if kernel_type not in KERNEL_CONFIGS:
        print(f"\nERROR: Unknown kernel type '{kernel_type}'")
        print(f"  Available: {', '.join(KERNEL_CONFIGS.keys())}")
        print("\ncorrectness: FAIL")
        print("throughput_tflops: 0.000")
        return 1

    config = KERNEL_CONFIGS[kernel_type]

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------
    if args.tpu_gen:
        try:
            from pallas.profiler import get_tpu_spec
            tpu_spec = get_tpu_spec(args.tpu_gen)
        except KeyError as exc:
            print(f"WARNING: {exc}")
            tpu_spec = None
    else:
        tpu_spec = detect_tpu()

    print(f"\n=== HARDWARE INFO ===")
    device_meta = {}
    try:
        devices = jax.devices()
        platform = str(devices[0].platform) if devices else "unknown"
        device_kind = str(getattr(devices[0], "device_kind", "unknown")) if devices else "unknown"
        device_count = len(devices)
        jax_version = str(getattr(jax, "__version__", "unknown"))
        print(f"platform: {platform}")
        print(f"device_kind: {device_kind}")
        print(f"device_count: {device_count}")
        print(f"jax_version: {jax_version}")
    except Exception:
        print("platform: unavailable")

    if tpu_spec is not None:
        print(f"tpu_generation: {tpu_spec.generation}")
        print(f"tpu_name: {tpu_spec.name}")
        print(f"tpu_peak_tflops_bf16: {tpu_spec.peak_tflops_bf16}")
        print(f"tpu_hbm_bandwidth_gb_s: {tpu_spec.hbm_bandwidth_gb_s}")
        print(f"tpu_vmem_mib_per_core: {tpu_spec.vmem_mib_per_core}")
    else:
        print("tpu_spec: not available (running on CPU/GPU or unknown TPU generation)")

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------
    print(f"\n=== CORRECTNESS ===")
    try:
        correctness_results = run_correctness(kernel_fn, config, quick=args.quick)
    except Exception as exc:
        print(f"\nFATAL: Correctness testing crashed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        correctness_results = {"correctness": "FAIL", "smoke_test": "CRASH", "shape_sweep": "CRASH"}

    print(f"\n--- Correctness Summary ---")
    print(f"smoke_test: {correctness_results.get('smoke_test', 'N/A')}")
    print(f"shape_sweep: {correctness_results.get('shape_sweep', 'N/A')}")
    print(f"correctness: {correctness_results['correctness']}")

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------
    _perf_sizes = config["test_sizes"]
    _primary_label = next((l for l, _ in _perf_sizes if l == "large"), _perf_sizes[-1][0])
    _primary_size = next((s for l, s in _perf_sizes if l == _primary_label), _perf_sizes[-1][1])
    _dtype_str = config["test_dtypes"][0]
    _size_params = ", ".join(f"{k}={v}" for k, v in _primary_size.items())
    print(f"\n=== PERFORMANCE ({_primary_label}: {_size_params}, dtype={_dtype_str}) ===")

    perf_results: Dict[str, Any] = {"primary": None, "all": []}
    try:
        sizes_filter = "large" if args.quick else args.sizes
        perf_results = run_performance(kernel_fn, config, tpu_spec, sizes_filter=sizes_filter)
    except Exception as exc:
        print(f"\nFATAL: Performance benchmarking crashed: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    primary = perf_results.get("primary")
    if primary is not None:
        print(f"\n--- Performance Summary (primary: {primary['label']}) ---")
        print(f"latency_us: {primary['kernel_latency_us']:.1f}")
        print(f"throughput_tflops: {primary['throughput_tflops']:.3f}")
        print(f"bandwidth_gb_s: {primary['bandwidth_gb_s']:.1f}")
        print(f"pct_peak_compute: {primary['pct_peak_compute']:.1f}")
        print(f"pct_peak_bandwidth: {primary['pct_peak_bandwidth']:.1f}")
        print(f"arithmetic_intensity: {primary['arithmetic_intensity']:.2f}")
        print(f"bottleneck: {primary['bottleneck']}")
        print(f"flops: {primary['flops']}")
        print(f"bytes: {primary['bytes']}")

        # Estimate peak VRAM via array sizes (no CUDA memory tracker available)
        peak_vram_mb = primary["bytes"] / (1024 * 1024)
        print(f"peak_vram_mb: {peak_vram_mb:.0f}")

        print(f"\n=== COMPARISON VS JAX REFERENCE ===")
        print(f"ref_latency_us: {primary['ref_latency_us']:.1f}")
        print(f"kernel_latency_us: {primary['kernel_latency_us']:.1f}")
        print(f"speedup_vs_ref: {primary['speedup_vs_ref']:.3f}x")
        print(f"ref_tflops: {primary['ref_throughput_tflops']:.3f}")
        print(f"kernel_tflops: {primary['throughput_tflops']:.3f}")
    else:
        print(f"\nlatency_us: 0.0")
        print(f"throughput_tflops: 0.000")
        print(f"bandwidth_gb_s: 0.0")
        print(f"pct_peak_compute: 0.0")
        print(f"pct_peak_bandwidth: 0.0")
        print(f"bottleneck: unknown")
        print(f"peak_vram_mb: 0")

    # Size sweep table
    all_perf = perf_results.get("all", [])
    if len(all_perf) > 1:
        print(f"\n=== SIZE SWEEP ===")
        print(f"{'size':<12} {'kernel_us':>12} {'ref_us':>12} {'speedup':>10} {'tflops':>10} {'%peak':>8}")
        print("-" * 66)
        for entry in all_perf:
            print(
                f"{entry['label']:<12} {entry['kernel_latency_us']:>12.2f} "
                f"{entry['ref_latency_us']:>12.2f} {entry['speedup_vs_ref']:>9.3f}x "
                f"{entry['throughput_tflops']:>10.3f} {entry['pct_peak_compute']:>7.1f}%"
            )

    elapsed = time.time() - t_start
    print(f"\ntotal_bench_time_s: {elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
