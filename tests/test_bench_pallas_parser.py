"""Tests for tpu_vm.parse_bench_output() -- whitelist-based bench output parser."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so tpu_vm is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tpu_vm import parse_bench_output

# ---------------------------------------------------------------------------
# Sample outputs (hardcoded from pallas/bench.py format)
# ---------------------------------------------------------------------------

SAMPLE_SUCCESS_OUTPUT = """\
============================================================
AutoKernel JAX/Pallas Benchmark Harness
============================================================

=== HARDWARE INFO ===
platform: tpu
device_kind: TPU v6e
device_count: 8
jax_version: 0.5.1
tpu_generation: 6
tpu_name: v6e
tpu_peak_tflops_bf16: 918.0
tpu_hbm_bandwidth_gb_s: 2400.0
tpu_vmem_mib_per_core: 64

=== CORRECTNESS ===

--- Stage 1: Smoke Test ---
  PASS (max_abs_error=1.234567e-04)

--- Stage 2: Shape Sweep ---
  PASS: small bfloat16 (max_err=1.23e-04)
  PASS: medium bfloat16 (max_err=2.34e-04)
  PASS: large bfloat16 (max_err=3.45e-04)

--- Correctness Summary ---
smoke_test: PASS
shape_sweep: PASS (8 configs, worst_err=3.45e-04 at large/bfloat16)
correctness: PASS

=== PERFORMANCE (large: M=2048, N=2048, K=2048, dtype=bfloat16) ===

  Benchmarking: large ...
    kernel: 123.45 us | ref: 100.00 us | speedup: 0.810x | 1.115 TFLOPS | 12.1% peak

--- Performance Summary (primary: large) ---
latency_us: 123.4
throughput_tflops: 1.115
bandwidth_gb_s: 456.7
pct_peak_compute: 12.1
pct_peak_bandwidth: 19.0
arithmetic_intensity: 1024.00
bottleneck: compute
flops: 137438953472
bytes: 134217728
peak_vram_mb: 128

=== COMPARISON VS JAX REFERENCE ===
ref_latency_us: 100.0
kernel_latency_us: 123.4
speedup_vs_ref: 0.811x
ref_tflops: 1.374
kernel_tflops: 1.115

kernel_type: matmul
total_bench_time_s: 45.2
"""

SAMPLE_FAILURE_OUTPUT = """\
============================================================
AutoKernel JAX/Pallas Benchmark Harness
============================================================

correctness: FAIL
throughput_tflops: 0.000
kernel_type: matmul
"""


class TestParseBenchOutput:
    """Test parse_bench_output() whitelist parser."""

    def test_success_case_extracts_all_whitelisted_keys(self):
        """All 10 whitelisted keys should be present with correct values."""
        result = parse_bench_output(SAMPLE_SUCCESS_OUTPUT)

        assert result["correctness"] == "PASS"
        assert result["throughput_tflops"] == 1.115
        assert result["latency_us"] == 123.4
        assert result["bandwidth_gb_s"] == 456.7
        assert result["pct_peak_compute"] == 12.1
        assert result["pct_peak_bandwidth"] == 19.0
        assert result["bottleneck"] == "compute"
        assert result["speedup_vs_ref"] == 0.811  # trailing 'x' stripped
        assert result["kernel_type"] == "matmul"
        assert result["peak_vram_mb"] == 128.0

    def test_success_case_excludes_hardware_keys(self):
        """Hardware info keys must NOT appear in the result."""
        result = parse_bench_output(SAMPLE_SUCCESS_OUTPUT)

        assert "tpu_peak_tflops_bf16" not in result
        assert "tpu_hbm_bandwidth_gb_s" not in result
        assert "jax_version" not in result
        assert "platform" not in result
        assert "device_kind" not in result
        assert "device_count" not in result
        assert "tpu_generation" not in result
        assert "tpu_name" not in result
        assert "tpu_vmem_mib_per_core" not in result

    def test_failure_case(self):
        """Parser should extract metrics from failure output."""
        result = parse_bench_output(SAMPLE_FAILURE_OUTPUT)

        assert result["correctness"] == "FAIL"
        assert result["throughput_tflops"] == 0.0
        assert result["kernel_type"] == "matmul"

    def test_empty_output_returns_empty_dict(self):
        """Empty string should return empty dict."""
        result = parse_bench_output("")
        assert result == {}

    def test_non_whitelisted_keys_ignored(self):
        """Lines with non-whitelisted keys must be ignored."""
        output = "garbage_key: 999\nfoo_bar: hello\ncorrectness: PASS\n"
        result = parse_bench_output(output)

        assert "garbage_key" not in result
        assert "foo_bar" not in result
        assert result["correctness"] == "PASS"

    def test_last_occurrence_wins(self):
        """When a key appears multiple times, last occurrence wins."""
        output = "correctness: FAIL\ncorrectness: PASS\n"
        result = parse_bench_output(output)
        assert result["correctness"] == "PASS"

        output2 = "correctness: PASS\ncorrectness: FAIL\n"
        result2 = parse_bench_output(output2)
        assert result2["correctness"] == "FAIL"

    def test_speedup_x_suffix_stripped(self):
        """The trailing 'x' in speedup_vs_ref should be stripped."""
        output = "speedup_vs_ref: 2.345x\n"
        result = parse_bench_output(output)
        assert result["speedup_vs_ref"] == 2.345

    def test_type_coercion(self):
        """String keys stay as strings, numeric keys become floats."""
        output = "correctness: PASS\nbottleneck: memory-bound\nkernel_type: softmax\nthroughput_tflops: 3.14\n"
        result = parse_bench_output(output)

        assert isinstance(result["correctness"], str)
        assert isinstance(result["bottleneck"], str)
        assert isinstance(result["kernel_type"], str)
        assert isinstance(result["throughput_tflops"], float)
