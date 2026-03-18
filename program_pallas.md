# AutoKernel Pallas/TPU -- Autonomous Kernel Optimization Agent

You are an autonomous TPU kernel optimization researcher using JAX/Pallas.
You write Pallas kernels, benchmark them on a remote TPU VM, analyze results,
and iterate until performance targets are met.

---

## Overview

The workflow is a tight edit-sync-bench-read loop:

```
1. Edit kernel.py locally (Pallas kernel code)
2. Run: python tpu_vm.py bench-pallas --project <P> --skip-setup --skip-sync
3. Read structured results (correctness, TFLOPS, % peak)
4. Iterate: fix correctness issues or apply optimization techniques
```

The benchmark harness (`pallas/bench.py`) handles correctness verification,
performance measurement, and roofline analysis. You only modify `kernel.py`.

---

## Phase 1: Setup (One-Time)

### 1.1 Create the TPU VM

```bash
python tpu_vm.py create --project <PROJECT>
```

### 1.2 Bootstrap the environment

```bash
python tpu_vm.py setup --project <PROJECT>
```

This installs JAX[tpu], creates the Python venv, and clones the repo.

### 1.3 First benchmark run (full sync)

```bash
python tpu_vm.py bench-pallas --project <PROJECT> --skip-setup
```

This syncs the entire repo and runs `pallas/bench.py` remotely.
Subsequent runs use `--skip-sync` to only sync `kernel.py`.

---

## Phase 2: The Experiment Loop

### 2.1 Edit kernel.py

Write your Pallas kernel at the project root in `kernel.py`. The file MUST contain:

```python
KERNEL_TYPE = "matmul"   # Must match a key in pallas/bench.py KERNEL_CONFIGS
BACKEND = "pallas"       # REQUIRED -- identifies this as a Pallas kernel

def kernel_fn(**kwargs):
    """Entry point called by pallas/bench.py."""
    ...
```

**BACKEND = "pallas" is mandatory.** Without it, backend detection fails.

Starter kernels are available in `pallas/kernels/`. Copy one to `kernel.py`:
```bash
cp pallas/kernels/matmul.py kernel.py
```

### 2.2 Run the benchmark

```bash
# First run after setup (syncs full repo):
python tpu_vm.py bench-pallas --project <P> --skip-setup

# Subsequent runs (syncs only kernel.py -- fast iteration):
python tpu_vm.py bench-pallas --project <P> --skip-setup --skip-sync

# Benchmark a specific kernel type:
python tpu_vm.py bench-pallas --project <P> --skip-setup --skip-sync --kernel-type softmax

# Quick mode (smoke test + large size only):
python tpu_vm.py bench-pallas --project <P> --skip-setup --skip-sync --bench-args "--quick"
```

**Note:** `kernel.py` is ALWAYS synced to the TPU, even with `--skip-sync`.
The `--skip-sync` flag only skips the full repo sync.

### 2.3 Read the results

The bench output includes structured `key: value` lines:

| Key | Type | Meaning |
|-----|------|---------|
| `correctness` | PASS/FAIL | Must be PASS before looking at performance |
| `throughput_tflops` | float | Primary optimization target |
| `latency_us` | float | Kernel latency in microseconds |
| `pct_peak_compute` | float | % of theoretical TPU peak TFLOPS |
| `pct_peak_bandwidth` | float | % of peak HBM bandwidth |
| `bandwidth_gb_s` | float | Achieved memory bandwidth |
| `bottleneck` | string | "compute" or "memory-bound" |
| `speedup_vs_ref` | float | Speedup vs JAX reference (jnp.matmul etc.) |
| `peak_vram_mb` | float | Estimated memory usage |

### 2.4 Decision logic

```
if correctness == FAIL:
    Fix the kernel (shapes, dtypes, algorithm bugs)
elif pct_peak_compute < target:
    Apply optimization techniques (see below)
elif speedup_vs_ref < 1.0:
    Kernel is slower than JAX reference -- investigate
else:
    Record result and move to next kernel or size
```

---

## Phase 3: Pallas Kernel Writing Guide

### 3.1 Basic structure

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def kernel_body(x_ref, o_ref):
    """Pallas kernel body. Operates on block-sized tiles."""
    o_ref[...] = some_computation(x_ref[...])

def kernel_fn(x):
    return pl.pallas_call(
        kernel_body,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(num_blocks,),
        in_specs=[pl.BlockSpec(block_shape, index_map)],
        out_specs=pl.BlockSpec(block_shape, index_map),
    )(x)
```

### 3.2 Key concepts

- **BlockSpec**: Defines how data is tiled and mapped to kernel invocations
- **Grid**: Number of kernel invocations (like CUDA grid)
- **index_map**: Function `(block_idx...) -> (start_indices...)` mapping grid position to data offset
- **Refs**: `x_ref[...]` reads/writes the current tile (like shared memory in CUDA)

### 3.3 Supported kernel types

The benchmark harness supports 9 kernel types (see `pallas/bench.py KERNEL_CONFIGS`):

| Kernel | Signature | Sizes |
|--------|-----------|-------|
| matmul | `kernel_fn(A, B)` | M,N,K up to 4096x11008x4096 |
| softmax | `kernel_fn(x)` | rows,cols up to 4096x32000 |
| layernorm | `kernel_fn(x, weight, bias)` | batch,dim up to 4096x4096 |
| flash_attention | `kernel_fn(Q, K, V)` | batch,heads,seq,dim up to 1x32x2048x128 |
| fused_mlp | `kernel_fn(x, w_gate, w_up, w_down)` | batch,dim,hidden |
| cross_entropy | `kernel_fn(logits, targets)` | batch,vocab up to 4096x50257 |
| rotary_embedding | `kernel_fn(x, cos, sin)` | batch,heads,seq,dim |
| rmsnorm | `kernel_fn(x, weight)` | M,N up to 4096x4096 |
| reduce | `kernel_fn(x)` | M,N up to 8192x8192 |

---

## TPU-Specific Optimization Techniques

### 1. MXU Alignment

The TPU Matrix Multiply Unit (MXU) operates on fixed tile sizes:
- **v4**: 128x128 tiles
- **v5e/v5p**: 128x128 tiles
- **v6e (Trillium)**: 256x256 tiles (can also use 128x128)

**Always align block sizes to MXU dimensions.** Misaligned blocks waste
compute cycles on padding. For matmul, use `BLOCK_M = BLOCK_N = 128`
(or 256 on v6e).

```python
# Good: aligned to MXU
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 128

# Bad: wasted compute on padding
BLOCK_M, BLOCK_N, BLOCK_K = 100, 100, 100
```

### 2. VMEM Tiling

TPU cores have limited VMEM (vector memory, similar to GPU shared memory):
- **v4**: 32 MiB per core
- **v5e**: 32 MiB per core
- **v6e**: 64 MiB per core

Your kernel tiles must fit in VMEM. For a matmul with BLOCK_M=128,
BLOCK_N=128, BLOCK_K=128 in bfloat16:
- A tile: 128 * 128 * 2 = 32 KiB
- B tile: 128 * 128 * 2 = 32 KiB
- C tile: 128 * 128 * 4 = 64 KiB (float32 accumulator)
- Total: 128 KiB << 32 MiB (fits easily)

If you hit VMEM overflow, reduce block sizes or use K-tiling to stream
through the K dimension.

### 3. K-Dimension Tiling (for Matmul)

The starter kernel reads the full K dimension per tile. For large K,
tile the K dimension and accumulate:

```python
def matmul_body(a_ref, b_ref, o_ref, acc_ref):
    acc_ref[...] += jnp.dot(a_ref[...], b_ref[...],
                            preferred_element_type=jnp.float32)

# Grid over (M_tiles, N_tiles, K_tiles)
# Use carry for accumulation across K tiles
```

### 4. Megacore Utilization

v5e and v6e TPUs have "megacore" -- two MXUs per core. To utilize both:
- Use `dimension_semantics=["parallel", "parallel", ...]` in pallas_call
- The first grid dimension marked "parallel" is split across MXUs
- Rule of thumb: put the largest grid dimension first

### 5. bfloat16 as Native Dtype

TPU MXUs natively compute in bfloat16. Using float32 inputs forces
implicit casts and can halve throughput. Always:
- Accept bfloat16 inputs when possible
- Use `preferred_element_type=jnp.float32` in dot products for accuracy
- Cast outputs back to input dtype at the end

### 6. Pipeline Parallelism (Advanced)

For memory-bound kernels, overlap compute with memory access using
`pltpu.emit_pipeline`:

```python
from jax.experimental.pallas import tpu as pltpu

# Pipeline prefetches next tiles while computing current tiles
pltpu.emit_pipeline(...)
```

### 7. Minimize Host-Device Transfers

- Keep data on-device between kernel calls
- Use `jax.block_until_ready()` only for timing, not between operations
- Profile with `pallas/profiler.py` to identify transfer bottlenecks

---

## Hardware Specs Reference

From `pallas/profiler.py TPU_SPECS`:

| Gen | Name | Peak TFLOPS (bf16) | HBM BW (GB/s) | VMEM/Core |
|-----|------|--------------------|----------------|-----------|
| v4 | TPU v4 | 275 | 1200 | 32 MiB |
| v5e | TPU v5e | 197 | 1600 | 32 MiB |
| v5p | TPU v5p | 459 | 2400 | 64 MiB |
| v6e | Trillium | 918 | 2400 | 64 MiB |

**Ridge point** = Peak TFLOPS / Peak BW. Operations with arithmetic
intensity below the ridge point are memory-bound; above are compute-bound.

---

## Move-On Criteria

Stop optimizing a kernel when any of these are true:

1. **Near peak**: `pct_peak_compute >= 90%` -- you're close to hardware limit
2. **Strong speedup**: `speedup_vs_ref >= 2.0x` -- significant win over JAX
3. **Plateau**: 5 consecutive experiments with no improvement
4. **Time budget**: 2 hours spent on a single kernel

---

## Common Pitfalls

1. **Shape not divisible by block size**: Pallas requires exact divisibility.
   Pad inputs or choose block sizes that divide the problem dimensions.

2. **VMEM overflow**: Tiles too large for VMEM. Reduce block sizes or add
   K-tiling to reduce working set.

3. **dtype mismatch**: bench.py tests both bfloat16 and float32 for most
   kernels. Make sure your kernel handles both.

4. **Missing BACKEND variable**: kernel.py MUST have `BACKEND = "pallas"`.

5. **Stale kernel on TPU**: If results don't change after editing, run
   without `--skip-sync` to force full repo sync.

6. **Import errors on TPU**: The TPU VM only has JAX installed, not PyTorch.
   Never import torch/triton in a Pallas kernel.

---

## File Reference

| File | Purpose |
|------|---------|
| `kernel.py` | The single file you modify (at project root) |
| `pallas/bench.py` | Benchmark harness -- DO NOT modify |
| `pallas/profiler.py` | TPU specs, roofline analysis |
| `pallas/kernels/` | Starter kernel implementations (copy to kernel.py) |
| `tpu_vm.py` | TPU VM lifecycle + `bench-pallas` action |
| `program_pallas.md` | This file -- agent instructions |
| `orchestrate.py` | Multi-kernel orchestrator (CUDA-focused, future TPU support) |
