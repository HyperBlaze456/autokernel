import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "profile_jax.py"


class ProfileJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("profile_jax", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cls.mod = module

    def test_parse_args_defaults_to_programmatic_both_and_expected_logdir(self):
        args = self.mod.parse_args([
            "--module", "mypkg.model",
            "--function", "run_step",
            "--input-shape", "1,2048",
        ])
        self.assertEqual(args.capture, "programmatic")
        self.assertEqual(args.mode, "both")
        self.assertTrue(str(args.logdir).endswith("workspace/jax_profile"))

    def test_summarize_timings_reports_compile_and_warm_stats(self):
        summary = self.mod.summarize_timings(cold_latency_ms=1200.0, warm_latencies_ms=[100.0, 110.0, 130.0])
        self.assertAlmostEqual(summary["compile_overhead_ms"], 1090.0)
        self.assertAlmostEqual(summary["warm_latency_ms_mean"], 113.33, places=2)
        self.assertAlmostEqual(summary["warm_latency_ms_p50"], 110.0)
        self.assertAlmostEqual(summary["warm_latency_ms_p90"], 126.0)

    def test_summarize_timings_single_warm_iteration(self):
        summary = self.mod.summarize_timings(cold_latency_ms=500.0, warm_latencies_ms=[50.0])
        self.assertAlmostEqual(summary["compile_overhead_ms"], 450.0)
        self.assertEqual(summary["measure_iters"], 1)
        self.assertAlmostEqual(summary["warm_latency_ms_p50"], 50.0)
        self.assertAlmostEqual(summary["warm_latency_ms_p90"], 50.0)

    def test_summarize_timings_empty_warm(self):
        summary = self.mod.summarize_timings(cold_latency_ms=100.0, warm_latencies_ms=[])
        self.assertEqual(summary["measure_iters"], 0)
        self.assertAlmostEqual(summary["warm_latency_ms_mean"], 0.0)
        self.assertAlmostEqual(summary["compile_overhead_ms"], 100.0)

    def test_summarize_timings_cold_less_than_warm(self):
        """compile_overhead is clamped to 0 when cold < warm p50."""
        summary = self.mod.summarize_timings(cold_latency_ms=50.0, warm_latencies_ms=[100.0, 110.0])
        self.assertAlmostEqual(summary["compile_overhead_ms"], 0.0)

    def test_build_remote_instructions_includes_perfetto_and_xprof_tunnels(self):
        instructions = self.mod.build_remote_instructions(host="tpu-vm", user="alice", perfetto_port=9001, xprof_port=8791)
        self.assertIn("ssh -L 9001:127.0.0.1:9001 alice@tpu-vm", instructions["perfetto_tunnel"])
        self.assertIn("ssh -L 8791:127.0.0.1:8791 alice@tpu-vm", instructions["xprof_tunnel"])

    def test_heuristic_diagnosis_prefers_xla_rewrite_when_compile_dominates(self):
        diagnosis = self.mod.heuristic_diagnosis({
            "cold_latency_ms": 1500.0,
            "warm_latency_ms_mean": 120.0,
            "compile_overhead_ms": 1380.0,
            "annotated_regions": [],
        })
        self.assertEqual(diagnosis["primary_bottleneck"], "compile-bound")
        self.assertFalse(diagnosis["pallas_recommended"])
        self.assertIn("XLA/JAX rewrite", diagnosis["next_actions"][0])

    def test_heuristic_diagnosis_flags_gather_heavy_regions_for_rewrite_first(self):
        diagnosis = self.mod.heuristic_diagnosis({
            "cold_latency_ms": 220.0,
            "warm_latency_ms_mean": 180.0,
            "compile_overhead_ms": 40.0,
            "annotated_regions": [
                {"name": "build_kv_slice", "pct": 72.0},
                {"name": "attention_core", "pct": 18.0},
            ],
        })
        self.assertEqual(diagnosis["primary_bottleneck"], "xla-unfriendly-indexing")
        self.assertFalse(diagnosis["pallas_recommended"])
        self.assertIn("rewrite indexing", " ".join(diagnosis["next_actions"]))

    # --- Cold-vs-warm call order tests ---

    def test_run_timed_callable_cold_call_is_first_invocation(self):
        """The very first fn() call must be the cold (compile) measurement,
        not preceded by warmup iterations."""
        call_log = []

        class FakeArray:
            def __init__(self, idx):
                self.idx = idx

            def block_until_ready(self):
                call_log.append(("block", self.idx))

        call_counter = [0]

        def fake_fn():
            call_counter[0] += 1
            idx = call_counter[0]
            call_log.append(("call", idx))
            return FakeArray(idx)

        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = []

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            result = self.mod._run_timed_callable(fake_fn, measure_iters=2, warmup_iters=3)

        # Call #1 must be the cold call (before any warmup)
        self.assertEqual(call_log[0], ("call", 1))
        self.assertEqual(call_log[1], ("block", 1))

        # Then warmup: calls 2, 3, 4
        self.assertEqual(call_log[2], ("call", 2))
        self.assertEqual(call_log[3], ("block", 2))
        self.assertEqual(call_log[4], ("call", 3))
        self.assertEqual(call_log[5], ("block", 3))
        self.assertEqual(call_log[6], ("call", 4))
        self.assertEqual(call_log[7], ("block", 4))

        # Then measure: calls 5, 6
        self.assertEqual(call_log[8], ("call", 5))
        self.assertEqual(call_log[9], ("block", 5))
        self.assertEqual(call_log[10], ("call", 6))
        self.assertEqual(call_log[11], ("block", 6))

        # Total: 1 cold + 3 warmup + 2 measure = 6 calls
        self.assertEqual(call_counter[0], 6)

    def test_run_timed_callable_cold_latency_exceeds_warm(self):
        """With a simulated compile delay on the first call, cold > warm and
        compile_overhead is positive."""
        first_call = [True]

        class FakeArray:
            def block_until_ready(self):
                pass

        def fake_fn():
            if first_call[0]:
                first_call[0] = False
                # Simulate JIT compilation delay (~20ms)
                import time as _t
                _t.sleep(0.02)
            return FakeArray()

        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = []

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            result = self.mod._run_timed_callable(fake_fn, measure_iters=3, warmup_iters=2)

        # Cold latency should be meaningfully larger than warm mean
        self.assertGreater(result["cold_latency_ms"], result["warm_latency_ms_mean"])
        self.assertGreater(result["compile_overhead_ms"], 0.0)
        self.assertGreater(result["compile_fraction"], 0.0)

    def test_run_timed_callable_zero_warmup(self):
        """warmup_iters=0 should still work: 1 cold + N measure calls."""
        call_count = [0]

        def fake_fn():
            call_count[0] += 1
            return None  # no block_until_ready

        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = []

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            result = self.mod._run_timed_callable(fake_fn, measure_iters=5, warmup_iters=0)

        # 1 cold + 0 warmup + 5 measure = 6
        self.assertEqual(call_count[0], 6)
        self.assertEqual(result["measure_iters"], 5)

    # --- Trace artifact collection tests ---

    def test_collect_trace_artifacts_finds_profile_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = pathlib.Path(tmpdir)
            # Create fake trace artifacts
            (trace_dir / "events.xplane.pb").write_bytes(b"fake")
            (trace_dir / "trace.trace.json.gz").write_bytes(b"fake")
            (trace_dir / "run.perfetto-trace").write_bytes(b"fake")
            (trace_dir / "irrelevant.txt").write_text("not a trace")
            (trace_dir / "subdir").mkdir()
            (trace_dir / "subdir" / "nested.xplane.pb").write_bytes(b"fake")

            artifacts = self.mod._collect_trace_artifacts(trace_dir)

        self.assertEqual(len(artifacts), 4)
        names = [pathlib.Path(a).name for a in artifacts]
        self.assertIn("events.xplane.pb", names)
        self.assertIn("trace.trace.json.gz", names)
        self.assertIn("run.perfetto-trace", names)
        self.assertIn("nested.xplane.pb", names)
        # irrelevant.txt should not be included
        self.assertNotIn("irrelevant.txt", names)

    def test_collect_trace_artifacts_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = self.mod._collect_trace_artifacts(pathlib.Path(tmpdir))
        self.assertEqual(artifacts, [])

    def test_collect_trace_artifacts_nonexistent_dir(self):
        artifacts = self.mod._collect_trace_artifacts(pathlib.Path("/nonexistent/dir"))
        self.assertEqual(artifacts, [])


if __name__ == "__main__":
    unittest.main()
