import importlib.util
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
# Prefer the new pallas.profiler location; fall back to legacy root shim.
MODULE_PATH = ROOT / "pallas" / "profiler.py"
if not MODULE_PATH.exists():
    MODULE_PATH = ROOT / "profile_jax.py"


class ProfileJaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("pallas.profiler", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
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

    def test_build_profile_cli_args_omits_default_values(self):
        args = self.mod.parse_args([
            "--module", "mypkg.model",
            "--function", "run_step",
            "--input-shape", "1,2048",
        ])
        forwarded = self.mod.build_profile_cli_args(args)
        self.assertEqual(forwarded, ["--module", "mypkg.model", "--function", "run_step", "--input-shape", "1,2048"])

    def test_main_remote_tpu_delegates_to_tpu_vm(self):
        with mock.patch.object(self.mod.tpu_vm, "delegate_profile_jax_from_namespace", return_value=0) as delegate_mock:
            code = self.mod.main([
                "--module", "mypkg.model",
                "--function", "run_step",
                "--input-shape", "1,2048",
                "--remote-tpu",
                "--tpu-project", "demo-project",
            ])
        self.assertEqual(code, 0)
        delegate_mock.assert_called_once()
        delegated_args = delegate_mock.call_args.args[1]
        self.assertEqual(delegated_args, ["--module", "mypkg.model", "--function", "run_step", "--input-shape", "1,2048"])

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

    def test_run_timed_callable_zero_measure_iters(self):
        """measure_iters=0 should not force an extra warm measurement."""
        call_count = [0]

        def fake_fn():
            call_count[0] += 1
            return None

        fake_jax = mock.MagicMock()
        fake_jax.devices.return_value = []

        with mock.patch.object(self.mod, "_import_jax", return_value=fake_jax):
            result = self.mod._run_timed_callable(fake_fn, measure_iters=0, warmup_iters=0)

        self.assertEqual(call_count[0], 1)
        self.assertEqual(result["measure_iters"], 0)
        self.assertAlmostEqual(result["warm_latency_ms_mean"], 0.0)

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


    # --- TPU spec lookup tests ---

    def test_get_tpu_spec_v4(self):
        spec = self.mod.get_tpu_spec("v4")
        self.assertEqual(spec.generation, "v4")
        self.assertAlmostEqual(spec.peak_tflops_bf16, 275.0)
        self.assertAlmostEqual(spec.hbm_bandwidth_gb_s, 1200.0)
        self.assertEqual(spec.mxu_shape, (128, 128))
        self.assertEqual(spec.cores_per_chip, 2)
        self.assertAlmostEqual(spec.vmem_mib_per_core, 32.0)
        self.assertAlmostEqual(spec.smem_kib_per_core, 16.0)

    def test_get_tpu_spec_v6e(self):
        spec = self.mod.get_tpu_spec("v6e")
        self.assertEqual(spec.generation, "v6e")
        self.assertAlmostEqual(spec.peak_tflops_bf16, 918.0)
        self.assertAlmostEqual(spec.hbm_bandwidth_gb_s, 1640.0)
        self.assertEqual(spec.mxu_shape, (256, 256))
        self.assertAlmostEqual(spec.vmem_mib_per_core, 64.0)
        self.assertAlmostEqual(spec.smem_kib_per_core, 64.0)

    def test_get_tpu_spec_v5e(self):
        spec = self.mod.get_tpu_spec("v5e")
        self.assertEqual(spec.generation, "v5e")
        self.assertAlmostEqual(spec.peak_tflops_bf16, 197.0)
        self.assertAlmostEqual(spec.hbm_bandwidth_gb_s, 800.0)
        self.assertEqual(spec.mxu_shape, (128, 128))
        self.assertEqual(spec.cores_per_chip, 1)
        self.assertAlmostEqual(spec.vmem_mib_per_core, 128.0)

    def test_get_tpu_spec_v5p(self):
        spec = self.mod.get_tpu_spec("v5p")
        self.assertEqual(spec.generation, "v5p")
        self.assertAlmostEqual(spec.peak_tflops_bf16, 459.0)
        self.assertAlmostEqual(spec.hbm_bandwidth_gb_s, 2765.0)
        self.assertEqual(spec.mxu_shape, (128, 128))
        self.assertEqual(spec.cores_per_chip, 2)

    def test_get_tpu_spec_v6e_single_core(self):
        """v6e has 1 TensorCore per chip (256x256 MXU)."""
        spec = self.mod.get_tpu_spec("v6e")
        self.assertEqual(spec.cores_per_chip, 1)

    def test_get_tpu_spec_case_insensitive(self):
        spec = self.mod.get_tpu_spec("V4")
        self.assertEqual(spec.generation, "v4")
        spec2 = self.mod.get_tpu_spec("TPU v6e")
        self.assertEqual(spec2.generation, "v6e")

    def test_get_tpu_spec_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.mod.get_tpu_spec("v99")

    # --- Roofline analysis tests ---

    def test_compute_roofline_memory_bound(self):
        """Low arithmetic intensity (softmax-like) → memory bound."""
        spec = self.mod.get_tpu_spec("v4")
        # AI = 5M / 4M = 1.25 FLOP/byte
        # ridge point = 275e12 / 1200e9 ≈ 229.17
        result = self.mod.compute_roofline(
            flops=5_000_000, bytes_accessed=4_000_000, spec=spec,
        )
        self.assertEqual(result["bottleneck"], "memory_bound")
        self.assertAlmostEqual(result["arithmetic_intensity"], 1.25, places=2)
        self.assertGreater(result["ridge_point"], 200.0)

    def test_compute_roofline_compute_bound(self):
        """Large matmul → compute bound."""
        spec = self.mod.get_tpu_spec("v4")
        M, N, K = 4096, 4096, 4096
        flops = 2 * M * N * K
        bytes_accessed = (M * K + K * N + M * N) * 2  # bf16
        result = self.mod.compute_roofline(
            flops=flops, bytes_accessed=bytes_accessed, spec=spec,
        )
        self.assertEqual(result["bottleneck"], "compute_bound")
        self.assertGreater(result["arithmetic_intensity"], result["ridge_point"])

    def test_compute_roofline_attainable_memory_bound(self):
        """Memory-bound attainable TFLOPS = AI * bandwidth."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.compute_roofline(
            flops=1_000_000, bytes_accessed=1_000_000, spec=spec,
        )
        # AI = 1.0, attainable = 1.0 * 1200 GB/s = 1.2 TFLOPS
        self.assertAlmostEqual(result["attainable_tflops"], 1.2, places=1)

    def test_compute_roofline_attainable_compute_bound(self):
        """Compute-bound attainable TFLOPS = peak."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.compute_roofline(
            flops=1_000_000_000, bytes_accessed=1, spec=spec,
        )
        self.assertAlmostEqual(result["attainable_tflops"], 275.0)

    def test_compute_roofline_with_measured_latency(self):
        """Measured latency produces throughput and pct-of-peak fields."""
        spec = self.mod.get_tpu_spec("v4")
        flops = 2 * 4096 * 4096 * 4096  # ~137.4 GFLOP
        bytes_accessed = (4096 * 4096 * 3) * 2
        result = self.mod.compute_roofline(
            flops=flops, bytes_accessed=bytes_accessed, spec=spec,
            measured_latency_ms=1.0,
        )
        self.assertIn("measured_tflops", result)
        self.assertIn("pct_peak_compute", result)
        self.assertIn("measured_bandwidth_gb_s", result)
        self.assertIn("pct_peak_bandwidth", result)
        # 137.4 GFLOP in 1ms = 137.4 TFLOPS
        self.assertGreater(result["measured_tflops"], 100.0)
        self.assertGreater(result["pct_peak_compute"], 0.0)

    def test_compute_roofline_no_measured_fields_without_latency(self):
        """Without measured_latency_ms, measured_* fields should be absent."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.compute_roofline(
            flops=1000, bytes_accessed=1000, spec=spec,
        )
        self.assertNotIn("measured_tflops", result)
        self.assertNotIn("pct_peak_compute", result)

    def test_compute_roofline_zero_bytes(self):
        """Zero bytes → compute bound with inf arithmetic intensity."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.compute_roofline(flops=1000, bytes_accessed=0, spec=spec)
        self.assertEqual(result["bottleneck"], "compute_bound")
        self.assertEqual(result["arithmetic_intensity"], float("inf"))

    def test_compute_roofline_v6e_higher_ridge_point(self):
        """v6e has different ridge point than v4 due to different peak/bw ratio."""
        v4 = self.mod.get_tpu_spec("v4")
        v6e = self.mod.get_tpu_spec("v6e")
        r_v4 = self.mod.compute_roofline(flops=1000, bytes_accessed=100, spec=v4)
        r_v6e = self.mod.compute_roofline(flops=1000, bytes_accessed=100, spec=v6e)
        # v6e: 918/1640 ≈ 560, v4: 275/1200 ≈ 229
        self.assertGreater(r_v6e["ridge_point"], r_v4["ridge_point"])

    def test_compute_roofline_v5p_high_bandwidth(self):
        """v5p's high HBM bandwidth (2765 GB/s) lowers the ridge point."""
        v5p = self.mod.get_tpu_spec("v5p")
        v6e = self.mod.get_tpu_spec("v6e")
        r_v5p = self.mod.compute_roofline(flops=1000, bytes_accessed=100, spec=v5p)
        r_v6e = self.mod.compute_roofline(flops=1000, bytes_accessed=100, spec=v6e)
        # v5p: 459/2765 ≈ 166, v6e: 918/1640 ≈ 560
        # v5p ridge point is lower → more ops become compute-bound on v5p
        self.assertLess(r_v5p["ridge_point"], r_v6e["ridge_point"])

    def test_compute_roofline_same_op_different_bottleneck(self):
        """An op can be memory-bound on v6e but compute-bound on v5p."""
        v5p = self.mod.get_tpu_spec("v5p")
        v6e = self.mod.get_tpu_spec("v6e")
        # AI = 200 FLOP/byte: below v6e ridge (~560) but above v5p ridge (~166)
        r_v5p = self.mod.compute_roofline(flops=200_000, bytes_accessed=1000, spec=v5p)
        r_v6e = self.mod.compute_roofline(flops=200_000, bytes_accessed=1000, spec=v6e)
        self.assertEqual(r_v5p["bottleneck"], "compute_bound")
        self.assertEqual(r_v6e["bottleneck"], "memory_bound")

    # --- MXU utilization tests ---

    def test_estimate_mxu_full_utilization_v4(self):
        """128x128 matmul on v4 (128x128 MXU) → 100% utilization."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.estimate_mxu_utilization(128, 128, 128, spec)
        self.assertAlmostEqual(result["mxu_utilization"], 1.0)
        self.assertFalse(result["is_gemv_like"])
        self.assertAlmostEqual(result["padding_waste"], 0.0)

    def test_estimate_mxu_gemv_like_v4(self):
        """[16,128] x [128,128] on v4 → 12.5% utilization (NSA single-query pattern)."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.estimate_mxu_utilization(16, 128, 128, spec)
        self.assertAlmostEqual(result["mxu_utilization"], 16 / 128)
        self.assertTrue(result["is_gemv_like"])
        self.assertAlmostEqual(result["m_utilization"], 16 / 128)
        self.assertAlmostEqual(result["n_utilization"], 1.0)

    def test_estimate_mxu_oversized_capped(self):
        """Matmul larger than MXU → utilization capped at 100%."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.estimate_mxu_utilization(256, 256, 256, spec)
        self.assertAlmostEqual(result["mxu_utilization"], 1.0)
        self.assertFalse(result["is_gemv_like"])
        self.assertEqual(result["m_tiles"], 2)
        self.assertEqual(result["n_tiles"], 2)

    def test_estimate_mxu_v6e_larger_mxu(self):
        """128x128 matmul on v6e (256x256 MXU) → 25% utilization."""
        spec = self.mod.get_tpu_spec("v6e")
        result = self.mod.estimate_mxu_utilization(128, 128, 128, spec)
        self.assertAlmostEqual(result["mxu_utilization"], 0.25)
        self.assertTrue(result["is_gemv_like"])

    def test_estimate_mxu_v6e_full(self):
        """256x256 matmul on v6e (256x256 MXU) → 100%."""
        spec = self.mod.get_tpu_spec("v6e")
        result = self.mod.estimate_mxu_utilization(256, 256, 256, spec)
        self.assertAlmostEqual(result["mxu_utilization"], 1.0)
        self.assertFalse(result["is_gemv_like"])

    def test_estimate_mxu_padding_waste(self):
        """Non-aligned dimensions → padding waste > 0."""
        spec = self.mod.get_tpu_spec("v4")
        # 100x100 on 128x128 MXU: padded to 128x128, waste = 1 - (10000/16384)
        result = self.mod.estimate_mxu_utilization(100, 100, 128, spec)
        expected_waste = 1.0 - (100 * 100) / (128 * 128)
        self.assertAlmostEqual(result["padding_waste"], round(expected_waste, 4))
        self.assertTrue(result["is_gemv_like"])

    def test_estimate_mxu_tile_counts(self):
        """Verify tile count computation."""
        spec = self.mod.get_tpu_spec("v4")
        result = self.mod.estimate_mxu_utilization(300, 500, 200, spec)
        self.assertEqual(result["m_tiles"], 3)   # ceil(300/128)
        self.assertEqual(result["n_tiles"], 4)   # ceil(500/128)
        self.assertEqual(result["k_tiles"], 2)   # ceil(200/128)


if __name__ == "__main__":
    unittest.main()
