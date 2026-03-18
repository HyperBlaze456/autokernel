"""Final report: JIT'd naive scan vs JIT'd chunk kernel at modern LLM scale on TPU v6e.

Explicitly compares _kda_naive (sequential scan) vs _kda_chunk_v3c (chunked)
by forcing both paths regardless of D, to show where chunking helps vs hurts.
Also includes D=64 large-scale configs where chunk kernel dominates.
"""
import time
import json
import jax
import jax.numpy as jnp
from pallas.bench import gen_gated_delta_attention_inputs

from kernel import _kda_naive, _kda_chunk_v3c, _adaptive_chunk_size

print(f"Devices: {jax.device_count()}")
print(f"Device type: {jax.devices()[0].device_kind}")
print()


def bench_jit(fn, warmup=10, rep=30):
    """Benchmark a JIT-compiled function, return median ms."""
    jit_fn = jax.jit(fn)
    for _ in range(warmup):
        o = jit_fn()
        o.block_until_ready()
    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        o = jit_fn()
        o.block_until_ready()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2], o


def run_config(label, sz):
    B, H, T, D = sz["batch"], sz["heads"], sz["seq_len"], sz["head_dim"]
    C = _adaptive_chunk_size(T)
    inputs = gen_gated_delta_attention_inputs(sz, jnp.bfloat16)

    # Cast inputs to fp32 for both paths
    fp32 = {k: v.astype(jnp.float32) for k, v in inputs.items()}

    # JIT'd naive scan (sequential baseline)
    naive_ms, naive_out = bench_jit(lambda: _kda_naive(**fp32))

    # JIT'd chunk kernel (our optimized approach)
    chunk_ms, chunk_out = bench_jit(lambda: _kda_chunk_v3c(**fp32, chunk_size=C))

    # Correctness: chunk vs naive
    err = float(jnp.max(jnp.abs(chunk_out - naive_out)))
    correct = "PASS" if err < 0.1 else "FAIL"
    speedup = naive_ms / chunk_ms if chunk_ms > 0 else 0

    # Memory estimate (MB)
    elems = B * T * H * D
    input_mb = elems * 4 * 5 / 1e6  # q,k,v,g,beta approx
    state_mb = B * H * D * D * 4 / 1e6

    return {
        "label": label, "B": B, "H": H, "T": T, "D": D, "C": C,
        "naive_ms": naive_ms, "chunk_ms": chunk_ms, "speedup": speedup,
        "err": err, "correct": correct,
        "input_mb": round(input_mb, 1), "state_mb": round(state_mb, 1),
    }


# ============================================================
# Test matrix
# ============================================================
configs = [
    # === D=64: Chunk kernel dominates (small matmuls need batching) ===
    # Standard multi-head with many small heads (e.g., model_dim=4096 with H=64, D=64)
    ("D=64  H=32 T=2048",   {"batch": 1, "heads": 32, "seq_len": 2048,  "head_dim": 64}),
    ("D=64  H=32 T=4096",   {"batch": 1, "heads": 32, "seq_len": 4096,  "head_dim": 64}),
    ("D=64  H=32 T=8192",   {"batch": 1, "heads": 32, "seq_len": 8192,  "head_dim": 64}),
    ("D=64  H=32 T=16384",  {"batch": 1, "heads": 32, "seq_len": 16384, "head_dim": 64}),
    ("D=64  H=64 T=4096",   {"batch": 1, "heads": 64, "seq_len": 4096,  "head_dim": 64}),
    ("D=64  H=64 T=8192",   {"batch": 1, "heads": 64, "seq_len": 8192,  "head_dim": 64}),
    ("D=64  B=4  T=4096",   {"batch": 4, "heads": 32, "seq_len": 4096,  "head_dim": 64}),
    ("D=64  B=8  T=2048",   {"batch": 8, "heads": 32, "seq_len": 2048,  "head_dim": 64}),

    # === D=128: 7B/13B/70B-class models ===
    ("D=128 7B   T=2048",   {"batch": 1, "heads": 32, "seq_len": 2048,  "head_dim": 128}),
    ("D=128 7B   T=4096",   {"batch": 1, "heads": 32, "seq_len": 4096,  "head_dim": 128}),
    ("D=128 7B   T=8192",   {"batch": 1, "heads": 32, "seq_len": 8192,  "head_dim": 128}),
    ("D=128 7B   T=16384",  {"batch": 1, "heads": 32, "seq_len": 16384, "head_dim": 128}),
    ("D=128 13B  T=8192",   {"batch": 1, "heads": 40, "seq_len": 8192,  "head_dim": 128}),
    ("D=128 70B  T=4096",   {"batch": 1, "heads": 64, "seq_len": 4096,  "head_dim": 128}),
    ("D=128 70B  T=8192",   {"batch": 1, "heads": 64, "seq_len": 8192,  "head_dim": 128}),
    ("D=128 B=4  T=4096",   {"batch": 4, "heads": 32, "seq_len": 4096,  "head_dim": 128}),

    # === D=256: Newer architectures ===
    ("D=256 H=16 T=2048",   {"batch": 1, "heads": 16, "seq_len": 2048,  "head_dim": 256}),
    ("D=256 H=16 T=8192",   {"batch": 1, "heads": 16, "seq_len": 8192,  "head_dim": 256}),

    # === Stress: 32K sequence ===
    ("D=64  H=16 T=32768",  {"batch": 1, "heads": 16, "seq_len": 32768, "head_dim": 64}),
    ("D=128 H=16 T=32768",  {"batch": 1, "heads": 16, "seq_len": 32768, "head_dim": 128}),
]

results = []

print(f"{'Label':<22} {'B':>2} {'H':>3} {'T':>6} {'D':>4} {'C':>4} | {'Naive':>9} {'Chunk':>9} {'Speedup':>8} | {'MaxErr':>10} {'OK':>4}")
print("=" * 105)

prev_d = None
for label, sz in configs:
    d = sz["head_dim"]
    if prev_d is not None and d != prev_d:
        print("-" * 105)
    prev_d = d

    try:
        r = run_config(label, sz)
        results.append(r)
        winner = "<--chunk" if r["speedup"] > 1.05 else ("<--naive" if r["speedup"] < 0.95 else "  ~same")
        print(f"{label:<22} {r['B']:>2} {r['H']:>3} {r['T']:>6} {r['D']:>4} {r['C']:>4} | "
              f"{r['naive_ms']:>8.2f}ms {r['chunk_ms']:>8.2f}ms {r['speedup']:>7.2f}x | "
              f"{r['err']:>10.6f} {r['correct']:>4} {winner}")
    except Exception as e:
        msg = str(e).split('\n')[0][:60]
        print(f"{label:<22} {sz['batch']:>2} {sz['heads']:>3} {sz['seq_len']:>6} {sz['head_dim']:>4} | OOM: {msg}")
        results.append({"label": label, "error": msg})

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
d64 = [r for r in results if isinstance(r.get("D"), int) and r["D"] == 64 and "speedup" in r]
d128 = [r for r in results if isinstance(r.get("D"), int) and r["D"] == 128 and "speedup" in r]
d256 = [r for r in results if isinstance(r.get("D"), int) and r["D"] == 256 and "speedup" in r]

if d64:
    avg = sum(r["speedup"] for r in d64) / len(d64)
    best = max(d64, key=lambda r: r["speedup"])
    print(f"D=64:  avg speedup {avg:.2f}x, best {best['speedup']:.2f}x ({best['label']})")
if d128:
    avg = sum(r["speedup"] for r in d128) / len(d128)
    best = max(d128, key=lambda r: r["speedup"])
    print(f"D=128: avg speedup {avg:.2f}x, best {best['speedup']:.2f}x ({best['label']})")
if d256:
    avg = sum(r["speedup"] for r in d256) / len(d256)
    print(f"D=256: avg speedup {avg:.2f}x")

print(f"\nAll correct: {all(r.get('correct') == 'PASS' for r in results if 'correct' in r)}")
print("\n" + json.dumps(results, indent=2))
