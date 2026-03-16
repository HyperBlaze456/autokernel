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

    # --- RegionTracker tests ---

    def test_region_tracker_record_and_summarize(self):
        tracker = self.mod.RegionTracker()
        tracker.record("attention", 50.0)
        tracker.record("ffn", 30.0)
        tracker.record("layernorm", 20.0)
        summary = tracker.summarize()
        self.assertEqual(len(summary), 3)
        # Sorted by elapsed_ms descending
        self.assertEqual(summary[0]["name"], "attention")
        self.assertEqual(summary[1]["name"], "ffn")
        self.assertEqual(summary[2]["name"], "layernorm")
        # Percentages should sum to 100
        total_pct = sum(r["pct"] for r in summary)
        self.assertAlmostEqual(total_pct, 100.0, places=1)
        self.assertAlmostEqual(summary[0]["pct"], 50.0)

    def test_region_tracker_with_metadata(self):
        tracker = self.mod.RegionTracker()
        tracker.record("matmul", 100.0, device="tpu", flops=1e12)
        summary = tracker.summarize()
        self.assertEqual(summary[0]["device"], "tpu")
        self.assertEqual(summary[0]["flops"], 1e12)

    def test_region_tracker_reset(self):
        tracker = self.mod.RegionTracker()
        tracker.record("op", 10.0)
        self.assertEqual(len(tracker.summarize()), 1)
        tracker.reset()
        self.assertEqual(len(tracker.summarize()), 0)

    def test_region_tracker_empty_summarize(self):
        tracker = self.mod.RegionTracker()
        self.assertEqual(tracker.summarize(), [])

    def test_region_tracker_single_region_is_100_pct(self):
        tracker = self.mod.RegionTracker()
        tracker.record("only_op", 42.0)
        summary = tracker.summarize()
        self.assertAlmostEqual(summary[0]["pct"], 100.0)

    # --- TraceRegion tests ---

    def test_trace_region_records_elapsed_time(self):
        region = self.mod.TraceRegion("test_op")
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            with region:
                import time as _t
                _t.sleep(0.01)  # 10ms
        self.assertGreater(region.elapsed_ms, 5.0)  # at least 5ms

    def test_trace_region_with_tracker(self):
        tracker = self.mod.RegionTracker()
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            with self.mod.TraceRegion("op1", tracker=tracker):
                pass
            with self.mod.TraceRegion("op2", tracker=tracker):
                pass
        summary = tracker.summarize()
        self.assertEqual(len(summary), 2)
        names = {r["name"] for r in summary}
        self.assertEqual(names, {"op1", "op2"})

    def test_trace_region_convenience_via_tracker(self):
        tracker = self.mod.RegionTracker()
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            with tracker.region("conv"):
                pass
        self.assertEqual(len(tracker.summarize()), 1)
        self.assertEqual(tracker.summarize()[0]["name"], "conv")

    def test_trace_region_metadata_passes_through(self):
        tracker = self.mod.RegionTracker()
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            with self.mod.TraceRegion("op", tracker=tracker, layer=3, kind="fwd"):
                pass
        entry = tracker.summarize()[0]
        self.assertEqual(entry["layer"], 3)
        self.assertEqual(entry["kind"], "fwd")

    def test_trace_region_works_without_jax(self):
        """TraceRegion should not raise even if JAX is missing."""
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            with self.mod.TraceRegion("safe_op") as r:
                pass
        self.assertGreaterEqual(r.elapsed_ms, 0.0)

    # --- collect_device_metadata tests ---

    def test_collect_device_metadata_with_devices(self):
        fake_device = mock.MagicMock()
        fake_device.platform = "tpu"
        fake_device.device_kind = "TPU v4"
        fake_device.id = 0

        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = [fake_device]
        fake_jax.__version__ = "0.5.1"

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            meta = self.mod.collect_device_metadata()

        self.assertEqual(meta["platform"], "tpu")
        self.assertEqual(meta["device_kind"], "TPU v4")
        self.assertEqual(meta["device_count"], 1)
        self.assertEqual(meta["jax_version"], "0.5.1")
        self.assertEqual(len(meta["devices"]), 1)

    def test_collect_device_metadata_no_jax(self):
        with mock.patch.object(self.mod, "_import_jax", side_effect=RuntimeError("no jax")):
            meta = self.mod.collect_device_metadata()
        self.assertEqual(meta["platform"], "unavailable")
        self.assertEqual(meta["device_count"], 0)

    def test_collect_device_metadata_empty_devices(self):
        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = []
        fake_jax.__version__ = "0.5.0"

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            meta = self.mod.collect_device_metadata()
        self.assertEqual(meta["platform"], "cpu")
        self.assertEqual(meta["device_count"], 0)

    # --- compare_summaries tests ---

    def test_compare_summaries_improved(self):
        baseline = {"warm_latency_ms_p50": 100.0, "cold_latency_ms": 500.0, "compile_overhead_ms": 400.0}
        current = {"warm_latency_ms_p50": 80.0, "cold_latency_ms": 450.0, "compile_overhead_ms": 370.0}
        result = self.mod.compare_summaries(baseline, current)
        self.assertEqual(result["verdict"], "improved")
        self.assertAlmostEqual(result["warm_latency_ms_p50"]["delta"], -20.0)
        self.assertAlmostEqual(result["warm_latency_ms_p50"]["delta_pct"], -20.0)

    def test_compare_summaries_regressed(self):
        baseline = {"warm_latency_ms_p50": 100.0, "cold_latency_ms": 500.0, "compile_overhead_ms": 400.0}
        current = {"warm_latency_ms_p50": 120.0, "cold_latency_ms": 550.0, "compile_overhead_ms": 430.0}
        result = self.mod.compare_summaries(baseline, current)
        self.assertEqual(result["verdict"], "regressed")
        self.assertAlmostEqual(result["warm_latency_ms_p50"]["delta"], 20.0)

    def test_compare_summaries_neutral(self):
        baseline = {"warm_latency_ms_p50": 100.0, "cold_latency_ms": 500.0, "compile_overhead_ms": 400.0}
        current = {"warm_latency_ms_p50": 102.0, "cold_latency_ms": 505.0, "compile_overhead_ms": 403.0}
        result = self.mod.compare_summaries(baseline, current)
        self.assertEqual(result["verdict"], "neutral")

    def test_compare_summaries_zero_baseline(self):
        baseline = {"warm_latency_ms_p50": 0.0, "cold_latency_ms": 0.0, "compile_overhead_ms": 0.0}
        current = {"warm_latency_ms_p50": 50.0, "cold_latency_ms": 100.0, "compile_overhead_ms": 50.0}
        result = self.mod.compare_summaries(baseline, current)
        self.assertEqual(result["warm_latency_ms_p50"]["delta_pct"], 0.0)  # avoid division by zero

    def test_compare_summaries_missing_keys(self):
        """compare_summaries should handle missing keys gracefully."""
        result = self.mod.compare_summaries({}, {})
        self.assertEqual(result["verdict"], "neutral")
        self.assertAlmostEqual(result["warm_latency_ms_p50"]["baseline"], 0.0)
        self.assertAlmostEqual(result["warm_latency_ms_p50"]["current"], 0.0)

    # --- build_remote_instructions logdir tests ---

    def test_build_remote_instructions_uses_custom_logdir(self):
        """build_remote_instructions should use provided logdir, not DEFAULT_LOGDIR."""
        custom_logdir = pathlib.Path("/custom/profile_output")
        instructions = self.mod.build_remote_instructions(host="tpu-vm", logdir=custom_logdir)
        self.assertIn("/custom/profile_output/xprof", instructions["collect_profile"])
        self.assertNotIn("workspace/jax_profile", instructions["collect_profile"])

    def test_build_remote_instructions_default_logdir_when_none(self):
        """build_remote_instructions should fall back to DEFAULT_LOGDIR when logdir is None."""
        instructions = self.mod.build_remote_instructions(host="tpu-vm")
        self.assertIn("workspace/jax_profile", instructions["collect_profile"])

    # --- run_profile integration tests ---

    def test_run_profile_dry_run(self):
        """Dry run returns metadata without executing the target function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "4,128",
                "--dry-run",
                "--logdir", tmpdir,
            ])
            result = self.mod.run_profile(args)

        self.assertEqual(result["mode"], "both")
        self.assertEqual(result["capture"], "programmatic")
        self.assertEqual(result["input_shape"], [4, 128])
        self.assertIn("diagnosis", result)
        self.assertAlmostEqual(result["cold_latency_ms"], 0.0)
        self.assertEqual(result["measure_iters"], 0)

    def test_run_profile_server_capture(self):
        """Server capture mode returns instructions, not execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "4,128",
                "--capture", "server",
                "--logdir", tmpdir,
            ])
            result = self.mod.run_profile(args)

        self.assertIn("notes", result)
        self.assertTrue(any("collect_profile" in n for n in result["notes"]))
        self.assertIn("diagnosis", result)

    def _make_fake_jax_and_fn(self):
        """Helper: return a mock jax module and a no-op callable."""
        fake_profiler = mock.MagicMock()
        fake_profiler.ProfileOptions.side_effect = AttributeError("not available")
        fake_jax = mock.MagicMock()
        fake_jax.profiler = fake_profiler
        fake_jax.devices.return_value = []

        def fake_fn():
            return None

        return fake_jax, fake_profiler, fake_fn

    def test_run_profile_mode_perfetto_traces_to_trace_dir(self):
        """mode=perfetto should trace to trace_dir with create_perfetto_link=True."""
        fake_jax, fake_profiler, fake_fn = self._make_fake_jax_and_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "2,64",
                "--mode", "perfetto",
                "--logdir", tmpdir,
            ])
            with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax), \
                 mock.patch.object(self.mod, "_load_callable", return_value=fake_fn):
                result = self.mod.run_profile(args)

        call_args = fake_profiler.start_trace.call_args
        trace_path = call_args[0][0]
        self.assertTrue(trace_path.endswith("/trace"))
        self.assertTrue(call_args[1]["create_perfetto_link"])
        self.assertIn("diagnosis", result)

    def test_run_profile_mode_xprof_traces_to_xprof_dir(self):
        """mode=xprof should trace to xprof_dir with create_perfetto_link=False."""
        fake_jax, fake_profiler, fake_fn = self._make_fake_jax_and_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "2,64",
                "--mode", "xprof",
                "--logdir", tmpdir,
            ])
            with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax), \
                 mock.patch.object(self.mod, "_load_callable", return_value=fake_fn):
                result = self.mod.run_profile(args)

        call_args = fake_profiler.start_trace.call_args
        trace_path = call_args[0][0]
        self.assertTrue(trace_path.endswith("/xprof"))
        self.assertFalse(call_args[1]["create_perfetto_link"])

    def test_run_profile_mode_both_traces_to_xprof_dir_with_perfetto(self):
        """mode=both should trace to xprof_dir with create_perfetto_link=True."""
        fake_jax, fake_profiler, fake_fn = self._make_fake_jax_and_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "2,64",
                "--mode", "both",
                "--logdir", tmpdir,
            ])
            with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax), \
                 mock.patch.object(self.mod, "_load_callable", return_value=fake_fn):
                result = self.mod.run_profile(args)

        call_args = fake_profiler.start_trace.call_args
        trace_path = call_args[0][0]
        self.assertTrue(trace_path.endswith("/xprof"))
        self.assertTrue(call_args[1]["create_perfetto_link"])

    def test_run_profile_perfetto_link_overrides_xprof_mode(self):
        """--perfetto-link forces create_perfetto_link=True even in xprof mode."""
        fake_jax, fake_profiler, fake_fn = self._make_fake_jax_and_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "2,64",
                "--mode", "xprof",
                "--perfetto-link",
                "--logdir", tmpdir,
            ])
            with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax), \
                 mock.patch.object(self.mod, "_load_callable", return_value=fake_fn):
                result = self.mod.run_profile(args)

        call_args = fake_profiler.start_trace.call_args
        self.assertTrue(call_args[1]["create_perfetto_link"])

    def test_run_profile_remote_instructions_honor_logdir(self):
        """Remote instructions in run_profile should use args.logdir, not defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "2,64",
                "--dry-run",
                "--logdir", tmpdir,
                "--remote-host", "tpu-vm",
                "--remote-user", "alice",
            ])
            result = self.mod.run_profile(args)

        instructions = result["remote_instructions"]
        self.assertIn(tmpdir, instructions["collect_profile"])
        self.assertNotIn("workspace/jax_profile", instructions["collect_profile"])

    def test_run_profile_programmatic_returns_timings(self):
        """Programmatic capture should return real timing fields."""
        fake_jax, fake_profiler, fake_fn = self._make_fake_jax_and_fn()
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self.mod.parse_args([
                "--module", "fake_module",
                "--function", "fake_fn",
                "--input-shape", "8,256",
                "--warmup-iters", "1",
                "--measure-iters", "2",
                "--logdir", tmpdir,
            ])
            with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax), \
                 mock.patch.object(self.mod, "_load_callable", return_value=fake_fn):
                result = self.mod.run_profile(args)

        self.assertIn("cold_latency_ms", result)
        self.assertIn("warm_latency_ms_mean", result)
        self.assertEqual(result["measure_iters"], 2)
        self.assertIn("diagnosis", result)


if __name__ == "__main__":
    unittest.main()
