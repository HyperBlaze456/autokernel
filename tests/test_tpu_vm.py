import importlib.util
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tpu_vm.py"


class TPUVMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("tpu_vm", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls.mod = module

    def test_build_ssh_base_uses_iap_and_worker(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="europe-west4-a",
            project="demo-project",
            worker="3",
        )
        cmd = self.mod.build_ssh_base(cfg)
        self.assertEqual(cmd[:6], ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh"])
        self.assertIn("root@v6-test", cmd)
        self.assertIn("demo-project", cmd)
        self.assertIn("--worker=3", cmd)
        self.assertIn("--tunnel-through-iap", cmd)

    def test_build_scp_command_recursive_upload(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
            worker="all",
        )
        cmd = self.mod.build_scp_command(cfg, "/tmp/kernel.py", "/root/autokernel/kernel.py", recursive=True)
        self.assertEqual(cmd[:6], ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "scp"])
        self.assertIn("--recurse", cmd)
        self.assertIn("root@v6-test:/root/autokernel/kernel.py", cmd)
        self.assertIn("--worker=all", cmd)

    def test_build_scp_command_download_direction(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
        )
        cmd = self.mod.build_scp_command(cfg, "/root/autokernel/workspace/result.json", "/tmp/result.json", direction="download")
        self.assertIn("root@v6-test:/root/autokernel/workspace/result.json", cmd)
        self.assertIn("/tmp/result.json", cmd)

    def test_build_repo_command_cd_then_activates_env(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
            remote_root="/root/autokernel",
        )
        command = self.mod.build_repo_command(cfg, "python profile_jax.py --dry-run", python_env=".venv-tpu")
        self.assertEqual(
            command,
            "cd /root/autokernel && . .venv-tpu/bin/activate && python profile_jax.py --dry-run",
        )

    def test_build_qr_create_command_uses_spot_runtime_and_internal_ips(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="europe-west4-a",
            project="demo-project",
            accelerator="v6e-16",
            runtime="v2-alpha-tpuv6e",
            service_account="svc@example.com",
        )
        cmd = self.mod.build_qr_create_command(cfg)
        joined = " ".join(cmd)
        self.assertIn("queued-resources create v6-test-qr", joined)
        self.assertIn("--spot", cmd)
        self.assertIn("--internal-ips", cmd)
        self.assertIn("--runtime-version", cmd)
        self.assertIn("v2-alpha-tpuv6e", cmd)
        self.assertIn("--service-account", cmd)
        self.assertIn("svc@example.com", cmd)

    def test_build_setup_script_targets_autokernel_repo_and_tpu_venv(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="europe-west4-a",
            project="demo-project",
            remote_root="/root/autokernel",
        )
        opts = self.mod.RepoSyncOptions(
            local_root=ROOT,
            repo_url="https://github.com/HyperBlaze456/autokernel.git",
            repo_ref="main",
            python_env=".venv-tpu",
            setup_packages=("pytest>=8", "xprof>=2.19.0"),
        )
        script = self.mod.build_setup_script(opts, cfg)
        self.assertIn("git clone https://github.com/HyperBlaze456/autokernel.git /root/autokernel", script)
        self.assertIn("python3 -m venv .venv-tpu", script)
        self.assertIn("pip install", script)
        self.assertIn("pytest>=8", script)
        self.assertIn("xprof>=2.19.0", script)
        self.assertIn("jax[tpu]>=0.5.0", script)

    def test_create_repo_archive_excludes_workspace_git_and_omc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            (root / "keep.py").write_text("print('ok')\n")
            (root / "workspace").mkdir()
            (root / "workspace" / "artifact.txt").write_text("ignore\n")
            (root / ".git").mkdir()
            (root / ".git" / "config").write_text("ignore\n")
            (root / ".omc").mkdir()
            (root / ".omc" / "state.json").write_text("ignore\n")
            (root / "pkg").mkdir()
            (root / "pkg" / "module.py").write_text("x = 1\n")

            archive = self.mod.create_repo_archive(root)
            self.addCleanup(lambda: shutil.rmtree(archive.parent, ignore_errors=True))

            import tarfile
            with tarfile.open(archive, "r:gz") as tf:
                names = set(tf.getnames())

        self.assertIn("keep.py", names)
        self.assertIn("pkg/module.py", names)
        self.assertNotIn("workspace/artifact.txt", names)
        self.assertNotIn(".git/config", names)
        self.assertNotIn(".omc/state.json", names)

    def test_build_pytest_command_runs_inside_repo_venv(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="europe-west4-a",
            project="demo-project",
            remote_root="/root/autokernel",
        )
        cmd = self.mod.build_pytest_command(cfg, ["tests/test_profile_jax.py", "-q"], python_env=".venv-tpu")
        self.assertIn("cd /root/autokernel", cmd)
        self.assertIn(". .venv-tpu/bin/activate", cmd)
        self.assertIn("python -m pytest tests/test_profile_jax.py -q", cmd)

    def test_action_profile_jax_runs_setup_sync_then_remote_python(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
            remote_root="/root/autokernel",
        )
        args = self.mod.argparse.Namespace(
            profile_args="--module mypkg.model --function run_step --input-shape 1,2048",
            skip_setup=False,
            skip_sync=False,
            dry_run=False,
            repo_url="https://github.com/HyperBlaze456/autokernel.git",
            repo_ref="main",
            local_root=str(ROOT),
            python_env=".venv-tpu",
            package=None,
        )

        with mock.patch.object(self.mod, "action_setup", return_value=0) as setup_mock, \
             mock.patch.object(self.mod, "action_sync_repo", return_value=0) as sync_mock, \
             mock.patch.object(self.mod, "run_command", return_value=0) as run_mock:
            code = self.mod.action_profile_jax(args, cfg)

        self.assertEqual(code, 0)
        setup_mock.assert_called_once()
        sync_mock.assert_called_once()
        run_cmd = run_mock.call_args.args[0]
        self.assertIn("profile_jax.py --module mypkg.model --function run_step --input-shape 1,2048", " ".join(run_cmd))

    def test_action_create_waits_for_active_then_vm_ready(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
        )
        args = self.mod.argparse.Namespace(max_retry=5, retry_interval=1, dry_run=False)
        with mock.patch.object(self.mod, "get_qr_state", return_value="NOT_FOUND"), \
             mock.patch.object(self.mod, "run_command", return_value=0) as run_mock, \
             mock.patch.object(self.mod, "wait_for_active", return_value=0) as active_mock, \
             mock.patch.object(self.mod, "wait_for_vm_ready", return_value=0) as ready_mock:
            code = self.mod.action_create(args, cfg)
        self.assertEqual(code, 0)
        run_mock.assert_called_once()
        active_mock.assert_called_once()
        ready_mock.assert_called_once()

    def test_action_watchdog_syncs_repo_after_setup(self):
        cfg = self.mod.TPUVMConfig(
            name="v6-test",
            zone="us-central2-b",
            project="demo-project",
        )
        args = self.mod.argparse.Namespace(
            max_retry=5,
            retry_interval=1,
            dry_run=False,
            repo_url="https://github.com/HyperBlaze456/autokernel.git",
            repo_ref="main",
            local_root=str(ROOT),
            python_env=".venv-tpu",
            package=None,
            failure_threshold=3,
        )
        with mock.patch.object(self.mod, "action_create", return_value=0) as create_mock, \
             mock.patch.object(self.mod, "action_setup", return_value=0) as setup_mock, \
             mock.patch.object(self.mod, "action_sync_repo", return_value=0) as sync_mock, \
             mock.patch.object(self.mod, "action_watch", return_value=0) as watch_mock:
            code = self.mod.action_watchdog(args, cfg)
        self.assertEqual(code, 0)
        create_mock.assert_called_once()
        setup_mock.assert_called_once()
        sync_mock.assert_called_once()
        watch_mock.assert_called_once()

    def test_require_project_raises_when_missing(self):
        with self.assertRaises(SystemExit):
            self.mod._require_project("")


if __name__ == "__main__":
    unittest.main()
