#!/usr/bin/env python3
"""
AutoKernel JAX profiler scaffold.

This is a profiler-first entrypoint for JAX/XLA/Pallas workflows, designed to
produce agent-friendly profiling artifacts and summaries instead of assuming the
user should inspect raw traces manually.

Current scope:
- programmatic or server/manual capture modes
- cold vs warm timing summaries
- TraceRegion / RegionTracker for annotated region instrumentation
- device and platform metadata collection
- run-to-run comparison for agent-driven optimisation loops
- remote TPU tunnel instructions
- heuristic diagnosis for deciding whether to stay in JAX/XLA or escalate to Pallas
- TPU hardware specs (v4, v6e) with roofline analysis and MXU utilization estimation
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LOGDIR = SCRIPT_DIR / "workspace" / "jax_profile"
DEFAULT_TRACE_SUBDIR = "trace"
DEFAULT_XPROF_SUBDIR = "xprof"
DEFAULT_PROFILER_PORT = 9999
DEFAULT_PERFETTO_PORT = 9001
DEFAULT_XPROF_PORT = 8791
DEFAULT_WARMUP_ITERS = 3
DEFAULT_MEASURE_ITERS = 10


# =========================================================================
# TPU Hardware Specifications
# =========================================================================

@dataclass
class TPUSpec:
    """Hardware specification for a TPU generation."""

    name: str
    generation: str
    peak_tflops_bf16: float
    peak_tflops_fp32: float
    hbm_bandwidth_gb_s: float
    vmem_mib_per_core: float
    smem_kib_per_core: float
    mxu_shape: Tuple[int, int]
    cores_per_chip: int


_KNOWN_TPUS: Dict[str, TPUSpec] = {
    "v4": TPUSpec(
        name="TPU v4",
        generation="v4",
        peak_tflops_bf16=275.0,
        peak_tflops_fp32=137.5,
        hbm_bandwidth_gb_s=1200.0,
        vmem_mib_per_core=32.0,
        smem_kib_per_core=16.0,
        mxu_shape=(128, 128),
        cores_per_chip=2,
    ),
    "v5e": TPUSpec(
        name="TPU v5e",
        generation="v5e",
        peak_tflops_bf16=197.0,
        peak_tflops_fp32=98.5,
        hbm_bandwidth_gb_s=800.0,
        vmem_mib_per_core=128.0,
        smem_kib_per_core=16.0,
        mxu_shape=(128, 128),
        cores_per_chip=1,
    ),
    "v5p": TPUSpec(
        name="TPU v5p",
        generation="v5p",
        peak_tflops_bf16=459.0,
        peak_tflops_fp32=229.5,
        hbm_bandwidth_gb_s=2765.0,
        vmem_mib_per_core=64.0,
        smem_kib_per_core=32.0,
        mxu_shape=(128, 128),
        cores_per_chip=2,
    ),
    "v6e": TPUSpec(
        name="TPU v6e (Trillium)",
        generation="v6e",
        peak_tflops_bf16=918.0,
        peak_tflops_fp32=459.0,
        hbm_bandwidth_gb_s=1640.0,
        vmem_mib_per_core=64.0,
        smem_kib_per_core=64.0,
        mxu_shape=(256, 256),
        cores_per_chip=1,
    ),
}


def get_tpu_spec(generation: Optional[str] = None) -> TPUSpec:
    """Look up a TPU spec by generation string (e.g. ``'v4'``, ``'v6e'``).

    If *generation* is ``None``, attempt auto-detection via JAX device
    metadata.  Raises :class:`KeyError` if the generation is unknown.
    """
    if generation is not None:
        gen = generation.lower().replace("tpu", "").strip()
        if gen in _KNOWN_TPUS:
            return _KNOWN_TPUS[gen]
        raise KeyError(
            f"Unknown TPU generation '{generation}'. Available: {list(_KNOWN_TPUS.keys())}"
        )

    # Auto-detect from JAX
    try:
        jax = _import_jax()
        devices = jax.devices()
        if devices and str(devices[0].platform) == "tpu":
            device_kind = str(getattr(devices[0], "device_kind", "")).lower()
            for gen_key in _KNOWN_TPUS:
                if gen_key in device_kind:
                    return _KNOWN_TPUS[gen_key]
    except Exception:
        pass

    raise KeyError(
        "Could not auto-detect TPU generation. Specify explicitly: get_tpu_spec('v4')"
    )


def compute_roofline(
    flops: int,
    bytes_accessed: int,
    spec: TPUSpec,
    measured_latency_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Roofline analysis for a given operation on a TPU.

    Returns arithmetic intensity, ridge point, bottleneck classification,
    and the attainable performance ceiling.  When *measured_latency_ms* is
    provided, also returns achieved throughput and percentage-of-peak metrics.
    """
    if bytes_accessed <= 0:
        return {
            "arithmetic_intensity": float("inf"),
            "ridge_point": 0.0,
            "bottleneck": "compute_bound",
            "attainable_tflops": spec.peak_tflops_bf16,
            "pct_peak_compute": 0.0,
            "pct_peak_bandwidth": 0.0,
        }

    arithmetic_intensity = flops / bytes_accessed  # FLOP/byte
    ridge_point = (spec.peak_tflops_bf16 * 1e12) / (spec.hbm_bandwidth_gb_s * 1e9)

    if arithmetic_intensity < ridge_point:
        bottleneck = "memory_bound"
        attainable_tflops = arithmetic_intensity * spec.hbm_bandwidth_gb_s * 1e9 / 1e12
    else:
        bottleneck = "compute_bound"
        attainable_tflops = spec.peak_tflops_bf16

    result: Dict[str, Any] = {
        "arithmetic_intensity": round(arithmetic_intensity, 4),
        "ridge_point": round(ridge_point, 4),
        "bottleneck": bottleneck,
        "attainable_tflops": round(attainable_tflops, 3),
    }

    if measured_latency_ms is not None and measured_latency_ms > 0:
        measured_tflops = flops / (measured_latency_ms / 1000.0) / 1e12
        measured_bw = bytes_accessed / (measured_latency_ms / 1000.0) / 1e9
        result["measured_tflops"] = round(measured_tflops, 3)
        result["measured_bandwidth_gb_s"] = round(measured_bw, 1)
        result["pct_peak_compute"] = round(
            measured_tflops / spec.peak_tflops_bf16 * 100.0, 2
        )
        result["pct_peak_bandwidth"] = round(
            measured_bw / spec.hbm_bandwidth_gb_s * 100.0, 2
        )

    return result


def estimate_mxu_utilization(
    m: int, n: int, k: int, spec: TPUSpec,
) -> Dict[str, Any]:
    """Estimate MXU utilization for a matmul ``[M,K] x [K,N]``.

    The MXU processes tiles of *mxu_shape* each cycle.  Undersized
    dimensions waste lanes.  Returns utilization fraction, tile counts,
    and a flag indicating GEMV-like (under-saturated) workloads -- the
    exact pattern Henry Ko identified in the NSA single-query bottleneck.
    """
    mxu_m, mxu_n = spec.mxu_shape

    m_util = min(m, mxu_m) / mxu_m
    n_util = min(n, mxu_n) / mxu_n
    utilization = m_util * n_util

    m_tiles = math.ceil(m / mxu_m)
    n_tiles = math.ceil(n / mxu_n)
    k_tiles = math.ceil(k / mxu_m)

    padded_m = m_tiles * mxu_m
    padded_n = n_tiles * mxu_n
    padding_waste = (
        1.0 - (m * n) / (padded_m * padded_n)
        if padded_m * padded_n > 0
        else 0.0
    )

    return {
        "mxu_utilization": round(utilization, 4),
        "m_utilization": round(m_util, 4),
        "n_utilization": round(n_util, 4),
        "m_tiles": m_tiles,
        "n_tiles": n_tiles,
        "k_tiles": k_tiles,
        "padding_waste": round(padding_waste, 4),
        "is_gemv_like": m < mxu_m or n < mxu_n,
    }


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


class RegionTracker:
    """Accumulates wall-clock timing data for annotated trace regions.

    Use with TraceRegion context managers to build up a region-level breakdown
    that feeds into heuristic_diagnosis and the benchmark_summary JSON.
    """

    def __init__(self) -> None:
        self._regions: List[Dict[str, Any]] = []
        self._total_ms: float = 0.0

    def record(self, name: str, elapsed_ms: float, **metadata: Any) -> None:
        entry: Dict[str, Any] = {"name": name, "elapsed_ms": round(elapsed_ms, 3)}
        entry.update(metadata)
        self._regions.append(entry)
        self._total_ms += elapsed_ms

    def region(self, name: str, **metadata: Any) -> "TraceRegion":
        """Convenience: return a TraceRegion bound to this tracker."""
        return TraceRegion(name, tracker=self, **metadata)

    def summarize(self) -> List[Dict[str, Any]]:
        """Return regions sorted by elapsed time desc, each with a pct field."""
        result = []
        for r in self._regions:
            entry = dict(r)
            entry["pct"] = round(r["elapsed_ms"] / self._total_ms * 100.0, 2) if self._total_ms > 0 else 0.0
            result.append(entry)
        return sorted(result, key=lambda x: x["elapsed_ms"], reverse=True)

    def reset(self) -> None:
        self._regions.clear()
        self._total_ms = 0.0


class TraceRegion:
    """Context manager wrapping jax.profiler.TraceAnnotation with wall-clock timing.

    When used with a RegionTracker, the elapsed time and metadata are
    automatically recorded on exit.  Works even when JAX is not installed
    (the annotation is simply skipped).

    Usage::

        tracker = RegionTracker()
        with TraceRegion("attention", tracker=tracker):
            result = attention_fn(q, k, v)
    """

    def __init__(self, name: str, *, tracker: Optional[RegionTracker] = None, **metadata: Any) -> None:
        self.name = name
        self.tracker = tracker
        self.metadata = metadata
        self._annotation: Any = None
        self._start_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "TraceRegion":
        try:
            jax = _import_jax()
            self._annotation = jax.profiler.TraceAnnotation(self.name)
            self._annotation.__enter__()
        except Exception:
            self._annotation = None
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.elapsed_ms = (time.perf_counter_ns() - self._start_ns) / 1e6
        if self._annotation is not None:
            try:
                self._annotation.__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass
        if self.tracker is not None:
            self.tracker.record(self.name, self.elapsed_ms, **self.metadata)
        return False


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
    logdir: Optional[pathlib.Path] = None,
) -> Dict[str, str]:
    target = f"{user}@{host}" if user else host
    xprof_logdir = (logdir or DEFAULT_LOGDIR) / DEFAULT_XPROF_SUBDIR
    return {
        "perfetto_tunnel": f"ssh -L {perfetto_port}:127.0.0.1:{perfetto_port} {target}",
        "xprof_tunnel": f"ssh -L {xprof_port}:127.0.0.1:{xprof_port} {target}",
        "collect_profile": f"python -m jax.collect_profile {profiler_port} 3000 --log_dir={xprof_logdir} --no_perfetto_link",
        "xprof_url": f"http://localhost:{xprof_port}/",
    }


def collect_device_metadata() -> Dict[str, Any]:
    """Collect JAX device and platform information for the profiling report.

    Returns a dict with platform, device kind, count, JAX version, and
    per-device details.  Returns a minimal fallback dict if JAX is
    unavailable or device enumeration fails.
    """
    try:
        jax = _import_jax()
        devices = jax.devices()
        if not devices:
            return {"platform": "cpu", "device_count": 0, "jax_version": str(getattr(jax, "__version__", "unknown"))}
        first = devices[0]
        return {
            "platform": str(first.platform),
            "device_kind": str(getattr(first, "device_kind", "unknown")),
            "device_count": len(devices),
            "devices": [
                {"id": d.id, "platform": str(d.platform), "device_kind": str(getattr(d, "device_kind", "unknown"))}
                for d in devices
            ],
            "jax_version": str(getattr(jax, "__version__", "unknown")),
        }
    except Exception:
        return {"platform": "unavailable", "device_count": 0}


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


def compare_summaries(baseline: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two profiling summaries and return a structured delta report.

    Useful for agent workflows: profile -> optimise -> re-profile -> compare.
    Positive *_delta_pct values mean the metric *increased* (usually bad for
    latency).  The ``verdict`` field gives a one-word agent-friendly label.
    """
    def _delta(key: str) -> Dict[str, Any]:
        b = float(baseline.get(key, 0.0))
        c = float(current.get(key, 0.0))
        abs_delta = round(c - b, 3)
        pct_delta = round((c - b) / b * 100.0, 2) if b != 0 else 0.0
        return {"baseline": round(b, 3), "current": round(c, 3), "delta": abs_delta, "delta_pct": pct_delta}

    warm_d = _delta("warm_latency_ms_p50")
    cold_d = _delta("cold_latency_ms")
    compile_d = _delta("compile_overhead_ms")

    # Verdict: improved / regressed / neutral (5% threshold)
    p50_pct = warm_d["delta_pct"]
    if p50_pct <= -5.0:
        verdict = "improved"
    elif p50_pct >= 5.0:
        verdict = "regressed"
    else:
        verdict = "neutral"

    return {
        "warm_latency_ms_p50": warm_d,
        "cold_latency_ms": cold_d,
        "compile_overhead_ms": compile_d,
        "verdict": verdict,
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
    for _ in range(max(0, measure_iters)):
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

    result: Dict[str, Any] = {
        "runtime": "jax",
        "capture": args.capture,
        "mode": args.mode,
        "input_shape": [int(x.strip()) for x in args.input_shape.split(",") if x.strip()],
        "dtype": args.dtype,
        "logdir": str(args.logdir),
        "annotated_regions": [],
    }

    if args.remote_host:
        result["remote_instructions"] = build_remote_instructions(
            host=args.remote_host,
            user=args.remote_user,
            perfetto_port=args.perfetto_port,
            xprof_port=args.xprof_port,
            profiler_port=args.profiler_port,
            logdir=args.logdir,
        )

    # Collect device metadata (best-effort, non-blocking).
    if not args.dry_run:
        try:
            result["device_metadata"] = collect_device_metadata()
        except Exception:
            pass

    if args.dry_run:
        result.update(summarize_timings(0.0, []))
        result["diagnosis"] = heuristic_diagnosis(result)
        return result

    if args.capture == "server":
        result.update(summarize_timings(0.0, []))
        result["notes"] = [
            f"Start JAX profiler server via jax.profiler.start_server({args.profiler_port}) in the target process.",
            f"Then capture with: python -m jax.collect_profile {args.profiler_port} 3000 --log_dir={xprof_dir} --no_perfetto_link",
        ]
        result["diagnosis"] = heuristic_diagnosis(result)
        return result

    fn = _load_callable(args)
    jax = _import_jax()
    profiler = jax.profiler

    # Determine trace output directory and perfetto link based on --mode.
    # JAX's start_trace always produces XPlane protos (viewable in TensorBoard /
    # XProf); create_perfetto_link additionally uploads to ui.perfetto.dev.
    if args.mode == "perfetto":
        active_trace_dir = trace_dir
        create_perfetto_link = True
    elif args.mode == "xprof":
        active_trace_dir = xprof_dir
        create_perfetto_link = False
    else:  # both
        active_trace_dir = xprof_dir
        create_perfetto_link = True
    # --perfetto-link flag can force link creation in any mode.
    if args.perfetto_link:
        create_perfetto_link = True

    active_trace_dir.mkdir(parents=True, exist_ok=True)
    result["output_dir"] = str(active_trace_dir)

    options = None
    try:
        options = profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.host_tracer_level = 2
        options.device_tracer_level = 1
    except Exception:
        options = None

    profiler.start_trace(str(active_trace_dir), create_perfetto_link=create_perfetto_link, profiler_options=options)
    try:
        timings = _run_timed_callable(lambda: fn(), args.measure_iters, args.warmup_iters)
    finally:
        profiler.stop_trace()
    result.update(timings)

    # Collect trace artifacts for CI/remote consumption (non-blocking).
    artifacts = _collect_trace_artifacts(active_trace_dir)
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
