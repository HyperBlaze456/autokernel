#!/usr/bin/env python3
"""
AutoKernel JAX profiler scaffold.

This is a profiler-first entrypoint for JAX/XLA/Pallas workflows, designed to
produce agent-friendly profiling artifacts and summaries instead of assuming the
user should inspect raw traces manually.

Current scope:
- programmatic or server/manual capture modes
- cold vs warm timing summaries
- remote TPU tunnel instructions
- heuristic diagnosis for deciding whether to stay in JAX/XLA or escalate to Pallas
- lazy JAX import so the module remains importable on non-JAX environments
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pathlib
import statistics
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LOGDIR = SCRIPT_DIR / "workspace" / "jax_profile"
DEFAULT_TRACE_SUBDIR = "trace"
DEFAULT_XPROF_SUBDIR = "xprof"
DEFAULT_PROFILER_PORT = 9999
DEFAULT_PERFETTO_PORT = 9001
DEFAULT_XPROF_PORT = 8791
DEFAULT_WARMUP_ITERS = 3
DEFAULT_MEASURE_ITERS = 10


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_timings(cold_latency_ms: float, warm_latencies_ms: Iterable[float]) -> Dict[str, Any]:
    warm = [float(v) for v in warm_latencies_ms]
    warm_mean = float(statistics.mean(warm)) if warm else 0.0
    warm_p50 = _percentile(warm, 50) if warm else 0.0
    compile_overhead = max(0.0, float(cold_latency_ms) - warm_p50)
    return {
        "cold_latency_ms": round(float(cold_latency_ms), 3),
        "warm_latency_ms_mean": round(warm_mean, 3),
        "warm_latency_ms_min": round(min(warm), 3) if warm else 0.0,
        "warm_latency_ms_max": round(max(warm), 3) if warm else 0.0,
        "warm_latency_ms_p50": round(warm_p50, 3),
        "warm_latency_ms_p90": round(_percentile(warm, 90), 3) if warm else 0.0,
        "measure_iters": len(warm),
        "compile_overhead_ms": round(compile_overhead, 3),
        "compile_fraction": round((compile_overhead / cold_latency_ms) if cold_latency_ms > 0 else 0.0, 4),
    }


def build_remote_instructions(
    host: str,
    user: Optional[str] = None,
    perfetto_port: int = DEFAULT_PERFETTO_PORT,
    xprof_port: int = DEFAULT_XPROF_PORT,
    profiler_port: int = DEFAULT_PROFILER_PORT,
) -> Dict[str, str]:
    target = f"{user}@{host}" if user else host
    return {
        "perfetto_tunnel": f"ssh -L {perfetto_port}:127.0.0.1:{perfetto_port} {target}",
        "xprof_tunnel": f"ssh -L {xprof_port}:127.0.0.1:{xprof_port} {target}",
        "collect_profile": f"python -m jax.collect_profile {profiler_port} 3000 --log_dir={DEFAULT_LOGDIR / DEFAULT_XPROF_SUBDIR} --no_perfetto_link",
        "xprof_url": f"http://localhost:{xprof_port}/",
    }


def heuristic_diagnosis(summary: Dict[str, Any]) -> Dict[str, Any]:
    compile_fraction = float(summary.get("compile_fraction") or 0.0)
    compile_overhead = float(summary.get("compile_overhead_ms") or 0.0)
    warm_mean = float(summary.get("warm_latency_ms_mean") or 0.0)
    regions = summary.get("annotated_regions") or []
    lower_names = [str(r.get("name", "")).lower() for r in regions]
    joined_names = " ".join(lower_names)

    if compile_fraction >= 0.65 or (compile_overhead > 0 and compile_overhead >= 3 * max(warm_mean, 1.0)):
        return {
            "primary_bottleneck": "compile-bound",
            "secondary_bottlenecks": ["needs-shape-stability-review"],
            "pallas_recommended": False,
            "rationale": "Compile overhead dominates cold execution; custom kernels are unlikely to be the first lever.",
            "next_actions": [
                "Try an XLA/JAX rewrite before introducing Pallas.",
                "Check shape polymorphism, recompilation triggers, and static argument usage.",
                "Re-profile steady-state execution after reducing recompiles.",
            ],
        }

    if any(tok in joined_names for tok in ["gather", "slice", "index", "prefetch", "build_kv", "build_kv_slice"]):
        return {
            "primary_bottleneck": "xla-unfriendly-indexing",
            "secondary_bottlenecks": ["memory-data-movement"],
            "pallas_recommended": False,
            "rationale": "Annotated regions suggest indexing/gather-style work dominates, which often responds better to an XLA-friendly rewrite first.",
            "next_actions": [
                "First rewrite indexing and data movement to be more vectorized and XLA-friendly.",
                "Reduce large dynamic index materialization and re-profile.",
                "Only escalate to Pallas if the warm trace still shows a dominant device-side bottleneck.",
            ],
        }

    if warm_mean > 0 and regions:
        top = max(regions, key=lambda r: float(r.get("pct", 0.0)))
        top_name = str(top.get("name", "unknown"))
        top_pct = float(top.get("pct", 0.0))
        if top_pct >= 55.0:
            return {
                "primary_bottleneck": "device-hotspot",
                "secondary_bottlenecks": [top_name],
                "pallas_recommended": True,
                "rationale": f"A single annotated region ({top_name}) dominates warm execution, making it a strong kernelization candidate.",
                "next_actions": [
                    f"Inspect the {top_name} region in Perfetto/XProf and confirm compute vs memory behavior.",
                    "Benchmark an XLA-friendly rewrite against the current baseline.",
                    "If the hotspot remains dominant, prototype a Pallas kernel for that region.",
                ],
            }

    return {
        "primary_bottleneck": "unclear",
        "secondary_bottlenecks": [],
        "pallas_recommended": False,
        "rationale": "Not enough structured region data is available yet to justify custom kernel work.",
        "next_actions": [
            "Add TraceAnnotation regions around suspected hotspots.",
            "Capture a warm execution trace with Perfetto/XProf.",
            "Compare baseline JAX against an XLA-friendly rewrite before attempting Pallas.",
        ],
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AutoKernel JAX profiler scaffold for TPU/XLA/Pallas research workflows."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", type=str, default=None, help="Python script to import and profile.")
    src.add_argument("--module", type=str, default=None, help="Python module to import and profile.")
    parser.add_argument("--function", type=str, required=True, help="Function name to profile.")
    parser.add_argument("--input-shape", type=str, required=True, help="Comma-separated shape metadata for bookkeeping.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="dtype metadata for the run.")
    parser.add_argument("--capture", choices=["programmatic", "server"], default="programmatic")
    parser.add_argument("--mode", choices=["perfetto", "xprof", "both"], default="both")
    parser.add_argument("--logdir", type=pathlib.Path, default=DEFAULT_LOGDIR)
    parser.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    parser.add_argument("--measure-iters", type=int, default=DEFAULT_MEASURE_ITERS)
    parser.add_argument("--profiler-port", type=int, default=DEFAULT_PROFILER_PORT)
    parser.add_argument("--perfetto-port", type=int, default=DEFAULT_PERFETTO_PORT)
    parser.add_argument("--xprof-port", type=int, default=DEFAULT_XPROF_PORT)
    parser.add_argument("--remote-host", type=str, default=None)
    parser.add_argument("--remote-user", type=str, default=None)
    parser.add_argument("--perfetto-link", action="store_true", default=False, help="Request Perfetto link creation when supported.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Emit metadata/instructions without executing the target function.")
    return parser.parse_args(argv)


def _load_callable(args: argparse.Namespace) -> Callable[..., Any]:
    if args.script:
        script_path = pathlib.Path(args.script).resolve()
        sys.path.insert(0, str(script_path.parent))
        module = importlib.import_module(script_path.stem)
    else:
        module = importlib.import_module(args.module)
    fn = getattr(module, args.function, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Could not find callable '{args.function}'")
    return fn


def _import_jax():
    try:
        import jax  # type: ignore
        return jax
    except ImportError as exc:
        raise RuntimeError(
            "JAX is not installed. Install the optional JAX profiling dependencies before using profile_jax.py."
        ) from exc


def _run_timed_callable(fn: Callable[[], Any], measure_iters: int, warmup_iters: int) -> Dict[str, Any]:
    jax = _import_jax()

    # 1. Cold call FIRST — the very first invocation includes JIT compilation overhead.
    #    This must run before any warmup to capture the true compile cost.
    t0 = time.perf_counter()
    cold = fn()
    if hasattr(cold, "block_until_ready"):
        cold.block_until_ready()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Warmup — additional calls to reach steady state (caches, memory pools, etc.)
    for _ in range(max(0, warmup_iters)):
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()

    # 3. Measure steady-state latencies
    warm_latencies = []
    for _ in range(max(1, measure_iters)):
        t1 = time.perf_counter()
        out = fn()
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        warm_latencies.append((time.perf_counter() - t1) * 1000.0)

    # touch jax so the import is not optimized away and to emphasize runtime dependency
    _ = jax.devices()
    return summarize_timings(cold_ms, warm_latencies)


def _collect_trace_artifacts(trace_dir: pathlib.Path) -> List[str]:
    """Scan *trace_dir* for profile artifacts produced by JAX's profiler.

    Returns a list of file paths (strings) suitable for CI upload or manual
    inspection in Perfetto / TensorBoard.
    """
    suffixes = {".xplane.pb", ".trace.json.gz", ".perfetto-trace", ".pb"}
    artifacts: List[str] = []
    if trace_dir.is_dir():
        for p in sorted(trace_dir.rglob("*")):
            if p.is_file() and any(p.name.endswith(s) for s in suffixes):
                artifacts.append(str(p))
    return artifacts


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    args.logdir.mkdir(parents=True, exist_ok=True)
    trace_dir = args.logdir / DEFAULT_TRACE_SUBDIR
    xprof_dir = args.logdir / DEFAULT_XPROF_SUBDIR
    trace_dir.mkdir(parents=True, exist_ok=True)
    xprof_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "runtime": "jax",
        "capture": args.capture,
        "mode": args.mode,
        "input_shape": [int(x.strip()) for x in args.input_shape.split(",") if x.strip()],
        "dtype": args.dtype,
        "logdir": str(args.logdir),
        "trace_dir": str(trace_dir),
        "xprof_dir": str(xprof_dir),
        "annotated_regions": [],
    }

    if args.remote_host:
        result["remote_instructions"] = build_remote_instructions(
            host=args.remote_host,
            user=args.remote_user,
            perfetto_port=args.perfetto_port,
            xprof_port=args.xprof_port,
            profiler_port=args.profiler_port,
        )

    if args.dry_run:
        result.update(summarize_timings(0.0, []))
        result["diagnosis"] = heuristic_diagnosis(result)
        return result

    fn = _load_callable(args)

    if args.capture == "server":
        result.update(summarize_timings(0.0, []))
        result["notes"] = [
            f"Start JAX profiler server via jax.profiler.start_server({args.profiler_port}) in the target process.",
            f"Then capture with: python -m jax.collect_profile {args.profiler_port} 3000 --log_dir={xprof_dir} --no_perfetto_link",
        ]
        result["diagnosis"] = heuristic_diagnosis(result)
        return result

    jax = _import_jax()
    profiler = jax.profiler

    if args.mode in {"perfetto", "both", "xprof"}:
        options = None
        try:
            options = profiler.ProfileOptions()
            options.python_tracer_level = 0
            options.host_tracer_level = 2
            options.device_tracer_level = 1
        except Exception:
            options = None

        profiler.start_trace(str(xprof_dir), create_perfetto_link=bool(args.perfetto_link), profiler_options=options)
        try:
            timings = _run_timed_callable(lambda: fn(), args.measure_iters, args.warmup_iters)
        finally:
            profiler.stop_trace()
        result.update(timings)

        # Collect trace artifacts for CI/remote consumption (non-blocking).
        artifacts = _collect_trace_artifacts(xprof_dir)
        if artifacts:
            result["trace_artifacts"] = artifacts

    result["diagnosis"] = heuristic_diagnosis(result)
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = run_profile(args)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    metadata_path = args.logdir / "benchmark_summary.json"
    diagnosis_path = args.logdir / "diagnosis.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(diagnosis_path, "w", encoding="utf-8") as f:
        json.dump(result.get("diagnosis", {}), f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved summary to {metadata_path}")
    print(f"Saved diagnosis to {diagnosis_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
