#!/usr/bin/env python3
"""TPU spot lifecycle + remote test/profile runner for AutoKernel.

This module is the Python adaptation of the user's TPU spot-manager shell scripts,
rewritten for this repository's JAX/TPU workflow rather than the original
Docker/vLLM/tpu-inference setup.

What it handles:
- create / inspect / delete TPU spot queued resources
- open SSH sessions and run arbitrary remote commands
- bootstrap a TPU-friendly Python environment on the TPU VM
- sync the current local repo snapshot to the remote TPU checkout
- run TPU-oriented tests and `profile_jax.py` remotely
- watch / watchdog loops for preemption recovery

Typical flow:
    python tpu_vm.py create --project my-project
    python tpu_vm.py setup --project my-project
    python tpu_vm.py test --project my-project
    python tpu_vm.py profile-jax --project my-project --profile-args "--module mypkg.model --function run_step --input-shape 1,2048"

The module also exposes helper functions so local entrypoints like
`profile_jax.py --remote-tpu ...` can delegate into the same lifecycle.
"""

from __future__ import annotations

import argparse
import io
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REMOTE_ROOT = "/root/autokernel"
DEFAULT_NAME = "v6-hyperblaze"
DEFAULT_ZONE = "europe-west4-a"
DEFAULT_ACCELERATOR = "v6e-16"
DEFAULT_RUNTIME = "v2-alpha-tpuv6e"
DEFAULT_PYTHON_ENV = ".venv-tpu"
DEFAULT_TEST_ARGS = ["tests/test_profile_jax.py", "tests/test_tpu_vm.py", "-q"]
DEFAULT_SETUP_PACKAGES = [
    "pytest>=8.0.0",
    "numpy>=2.2.0",
    "pandas>=2.2.0",
    "matplotlib>=3.10.0",
    "xprof>=2.19.0",
]
JAX_TPU_INSTALL_TARGET = "jax[tpu]>=0.5.0"
JAX_TPU_WHEEL_INDEX = "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
DEFAULT_MAX_RETRY = 50
DEFAULT_RETRY_INTERVAL = 60
DEFAULT_WATCH_FAILURE_THRESHOLD = 5
SYNC_EXCLUDE_NAMES = {
    ".git",
    ".venv",
    ".venv-tpu",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "workspace",
    ".omc",
    ".omx",
    "build",
    "dist",
}
SYNC_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".log"}


@dataclass(frozen=True)
class TPUVMConfig:
    name: str
    zone: str
    project: str
    worker: str = "0"
    remote_root: str = DEFAULT_REMOTE_ROOT
    tunnel_through_iap: bool = True
    accelerator: str = DEFAULT_ACCELERATOR
    runtime: str = DEFAULT_RUNTIME
    service_account: str = ""


@dataclass(frozen=True)
class RepoSyncOptions:
    local_root: Path
    repo_url: str
    repo_ref: str
    python_env: str = DEFAULT_PYTHON_ENV
    setup_packages: tuple[str, ...] = tuple(DEFAULT_SETUP_PACKAGES)


def _project_from_env() -> str:
    return os.environ.get("TPU_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""


def _default_repo_url(local_root: Path | None = None) -> str:
    root = Path(local_root or SCRIPT_DIR)
    try:
        proc = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "https://github.com/HyperBlaze456/autokernel.git"


def _default_repo_ref(local_root: Path | None = None) -> str:
    root = Path(local_root or SCRIPT_DIR)
    for cmd in (["git", "branch", "--show-current"], ["git", "rev-parse", "--short", "HEAD"]):
        try:
            proc = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
            value = proc.stdout.strip()
            if value:
                return value
        except Exception:
            continue
    return "main"


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def _print_banner(message: str) -> None:
    print(f"[tpu_vm] {message}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage an AutoKernel TPU spot VM and run TPU-oriented tests/profile flows remotely."
    )
    parser.add_argument("--name", default=os.environ.get("TPU_NAME", DEFAULT_NAME))
    parser.add_argument("--zone", default=os.environ.get("TPU_ZONE", DEFAULT_ZONE))
    parser.add_argument("--project", default=_project_from_env())
    parser.add_argument("--worker", default=os.environ.get("TPU_WORKER", "0"))
    parser.add_argument("--remote-root", default=os.environ.get("TPU_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--accelerator", default=os.environ.get("TPU_ACCELERATOR", DEFAULT_ACCELERATOR))
    parser.add_argument("--runtime", default=os.environ.get("TPU_RUNTIME", DEFAULT_RUNTIME))
    parser.add_argument("--service-account", default=os.environ.get("TPU_SERVICE_ACCOUNT", ""))
    parser.add_argument("--no-iap", action="store_true", default=False, help="Do not use --tunnel-through-iap.")
    parser.add_argument("--dry-run", action="store_true", default=False)

    sub = parser.add_subparsers(dest="action", required=True)

    create_p = sub.add_parser("create", help="Create or wait for the TPU spot queued resource.")
    create_p.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY)
    create_p.add_argument("--retry-interval", type=int, default=DEFAULT_RETRY_INTERVAL)

    sub.add_parser("status", help="Show queued-resource and TPU VM state.")
    sub.add_parser("delete", help="Delete the TPU VM / queued resource.")
    sub.add_parser("ssh", help="Open an interactive SSH session to the TPU VM.")

    run_p = sub.add_parser("run", help="Run an arbitrary shell command on the TPU VM.")
    run_p.add_argument("--command", required=True, help="Remote shell command to execute.")
    run_p.add_argument("--activate-env", action="store_true", default=False, help="Run inside the repo root with the TPU Python env activated.")
    _add_repo_args(run_p)

    setup_p = sub.add_parser("setup", help="Clone/update the repo and bootstrap the TPU Python environment.")
    _add_repo_args(setup_p)

    sync_p = sub.add_parser("sync", help="Copy a local file or directory to the TPU VM.")
    sync_p.add_argument("--source", required=True, help="Local file or directory to upload.")
    sync_p.add_argument("--dest", required=True, help="Remote destination path.")
    sync_p.add_argument("--recursive", action="store_true", default=False)

    sync_repo_p = sub.add_parser("sync-repo", help="Upload the local repo snapshot to the TPU checkout.")
    _add_repo_args(sync_repo_p)

    pull_p = sub.add_parser("pull", help="Download a remote file or directory from the TPU VM.")
    pull_p.add_argument("--source", required=True, help="Remote path to download.")
    pull_p.add_argument("--dest", required=True, help="Local destination path.")
    pull_p.add_argument("--recursive", action="store_true", default=False)

    test_p = sub.add_parser("test", help="Run TPU-related pytest targets remotely.")
    _add_repo_args(test_p)
    test_p.add_argument("--pytest-args", default=shell_join(DEFAULT_TEST_ARGS), help="Arguments passed to pytest on the TPU VM.")
    test_p.add_argument("--skip-setup", action="store_true", default=False)
    test_p.add_argument("--skip-sync", action="store_true", default=False)

    profile_p = sub.add_parser("profile-jax", help="Run profile_jax.py remotely on the TPU VM.")
    _add_repo_args(profile_p)
    profile_p.add_argument("--profile-args", default="", help="Arguments forwarded to profile_jax.py.")
    profile_p.add_argument("--skip-setup", action="store_true", default=False)
    profile_p.add_argument("--skip-sync", action="store_true", default=False)

    watch_p = sub.add_parser("watch", help="Watch TPU state and recreate on preemption.")
    watch_p.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY)
    watch_p.add_argument("--retry-interval", type=int, default=DEFAULT_RETRY_INTERVAL)
    watch_p.add_argument("--failure-threshold", type=int, default=DEFAULT_WATCH_FAILURE_THRESHOLD)

    watchdog_p = sub.add_parser("watchdog", help="create -> setup -> watch with automatic recovery.")
    watchdog_p.add_argument("--max-retry", type=int, default=DEFAULT_MAX_RETRY)
    watchdog_p.add_argument("--retry-interval", type=int, default=DEFAULT_RETRY_INTERVAL)
    watchdog_p.add_argument("--failure-threshold", type=int, default=DEFAULT_WATCH_FAILURE_THRESHOLD)
    _add_repo_args(watchdog_p)

    bench_p = sub.add_parser("bench-pallas", help="Run pallas/bench.py remotely and return structured results.")
    _add_repo_args(bench_p)
    bench_p.add_argument("--kernel-type", default=None, help="Kernel type to benchmark (forwarded as --kernel to bench.py).")
    bench_p.add_argument("--bench-args", default="", help="Extra arguments forwarded to pallas/bench.py (e.g. '--quick --sizes large').")
    bench_p.add_argument("--skip-setup", action="store_true", default=False)
    bench_p.add_argument("--skip-sync", action="store_true", default=False, help="Skip full repo sync (kernel.py is ALWAYS synced).")

    return parser.parse_args(argv)


def _add_repo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-url", default=os.environ.get("TPU_REPO_URL") or _default_repo_url())
    parser.add_argument("--repo-ref", default=os.environ.get("TPU_REPO_REF") or _default_repo_ref())
    parser.add_argument("--local-root", default=str(SCRIPT_DIR), help="Local repo root to sync from.")
    parser.add_argument("--python-env", default=os.environ.get("TPU_PYTHON_ENV", DEFAULT_PYTHON_ENV))
    parser.add_argument(
        "--package",
        action="append",
        default=None,
        help="Extra package to install into the remote TPU venv. Repeatable.",
    )


def add_remote_tpu_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_sync_mode: str = "repo",
    default_local_root: str | None = None,
) -> None:
    parser.add_argument("--remote-tpu", action="store_true", default=False, help="Run this command on a remote TPU VM.")
    parser.add_argument("--tpu-name", default=os.environ.get("TPU_NAME", DEFAULT_NAME))
    parser.add_argument("--tpu-zone", default=os.environ.get("TPU_ZONE", DEFAULT_ZONE))
    parser.add_argument("--tpu-project", default=_project_from_env())
    parser.add_argument("--tpu-worker", default=os.environ.get("TPU_WORKER", "0"))
    parser.add_argument("--tpu-remote-root", default=os.environ.get("TPU_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--tpu-accelerator", default=os.environ.get("TPU_ACCELERATOR", DEFAULT_ACCELERATOR))
    parser.add_argument("--tpu-runtime", default=os.environ.get("TPU_RUNTIME", DEFAULT_RUNTIME))
    parser.add_argument("--tpu-service-account", default=os.environ.get("TPU_SERVICE_ACCOUNT", ""))
    parser.add_argument("--tpu-no-iap", action="store_true", default=False)
    parser.add_argument("--tpu-dry-run", action="store_true", default=False)
    parser.add_argument("--tpu-sync", choices=["repo", "none"], default=default_sync_mode)
    parser.add_argument("--tpu-skip-setup", action="store_true", default=False)
    parser.add_argument("--tpu-skip-sync", action="store_true", default=False)
    parser.add_argument("--tpu-repo-url", default=os.environ.get("TPU_REPO_URL") or _default_repo_url(Path(default_local_root or SCRIPT_DIR)))
    parser.add_argument("--tpu-repo-ref", default=os.environ.get("TPU_REPO_REF") or _default_repo_ref(Path(default_local_root or SCRIPT_DIR)))
    parser.add_argument("--tpu-local-root", default=str(Path(default_local_root or SCRIPT_DIR).resolve()))
    parser.add_argument("--tpu-python-env", default=os.environ.get("TPU_PYTHON_ENV", DEFAULT_PYTHON_ENV))
    parser.add_argument("--tpu-package", action="append", default=None)


def config_from_namespace(ns: argparse.Namespace, *, prefix: str = "") -> TPUVMConfig:
    return TPUVMConfig(
        name=getattr(ns, f"{prefix}name"),
        zone=getattr(ns, f"{prefix}zone"),
        project=_require_project(getattr(ns, f"{prefix}project")),
        worker=getattr(ns, f"{prefix}worker"),
        remote_root=getattr(ns, f"{prefix}remote_root"),
        tunnel_through_iap=not getattr(ns, f"{prefix}no_iap"),
        accelerator=getattr(ns, f"{prefix}accelerator"),
        runtime=getattr(ns, f"{prefix}runtime"),
        service_account=getattr(ns, f"{prefix}service_account"),
    )


def repo_options_from_namespace(ns: argparse.Namespace, *, prefix: str = "") -> RepoSyncOptions:
    extra_packages = list(DEFAULT_SETUP_PACKAGES)
    supplied = getattr(ns, f"{prefix}package", None) or []
    extra_packages.extend(supplied)
    # stable order with duplicate removal
    deduped: list[str] = []
    for item in extra_packages:
        if item not in deduped:
            deduped.append(item)
    return RepoSyncOptions(
        local_root=Path(getattr(ns, f"{prefix}local_root")).expanduser().resolve(),
        repo_url=getattr(ns, f"{prefix}repo_url"),
        repo_ref=getattr(ns, f"{prefix}repo_ref"),
        python_env=getattr(ns, f"{prefix}python_env"),
        setup_packages=tuple(deduped),
    )


def _require_project(project: str) -> str:
    if project:
        return project
    raise SystemExit("TPU project is required. Pass --project or set TPU_PROJECT / GOOGLE_CLOUD_PROJECT.")


def queued_resource_name(config: TPUVMConfig) -> str:
    return f"{config.name}-qr"


def build_ssh_base(config: TPUVMConfig, *, worker: str | None = None) -> list[str]:
    worker_value = worker or config.worker
    cmd = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        f"root@{config.name}",
        "--zone",
        config.zone,
        "--project",
        config.project,
        f"--worker={worker_value}",
    ]
    if config.tunnel_through_iap:
        cmd.append("--tunnel-through-iap")
    return cmd


def build_status_command(config: TPUVMConfig) -> list[str]:
    return build_tpu_vm_describe_command(config, format_value="value(state)")


def build_tpu_vm_describe_command(config: TPUVMConfig, *, format_value: str = "value(state)") -> list[str]:
    return [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "describe",
        config.name,
        "--zone",
        config.zone,
        "--project",
        config.project,
        f"--format={format_value}",
    ]


def build_tpu_vm_delete_command(config: TPUVMConfig) -> list[str]:
    return [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        config.name,
        "--zone",
        config.zone,
        "--project",
        config.project,
        "--quiet",
    ]


def build_qr_describe_command(config: TPUVMConfig, *, format_value: str = "value(state.state)") -> list[str]:
    return [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "describe",
        queued_resource_name(config),
        "--zone",
        config.zone,
        "--project",
        config.project,
        f"--format={format_value}",
    ]


def build_qr_create_command(config: TPUVMConfig) -> list[str]:
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "create",
        queued_resource_name(config),
        "--node-id",
        config.name,
        "--zone",
        config.zone,
        "--project",
        config.project,
        "--accelerator-type",
        config.accelerator,
        "--runtime-version",
        config.runtime,
        "--spot",
        "--internal-ips",
    ]
    if config.service_account:
        cmd.extend(["--service-account", config.service_account])
    return cmd


def build_qr_delete_command(config: TPUVMConfig) -> list[str]:
    return [
        "gcloud",
        "compute",
        "tpus",
        "queued-resources",
        "delete",
        queued_resource_name(config),
        "--zone",
        config.zone,
        "--project",
        config.project,
        "--quiet",
        "--force",
    ]


def build_scp_command(
    config: TPUVMConfig,
    source: str,
    dest: str,
    *,
    recursive: bool = False,
    direction: str = "upload",
) -> list[str]:
    cmd = ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "scp"]
    if recursive:
        cmd.append("--recurse")
    if direction == "upload":
        src_arg = source
        dst_arg = f"root@{config.name}:{dest}"
    elif direction == "download":
        src_arg = f"root@{config.name}:{source}"
        dst_arg = dest
    else:
        raise ValueError(f"Unknown copy direction: {direction}")
    cmd.extend(
        [
            src_arg,
            dst_arg,
            "--zone",
            config.zone,
            "--project",
            config.project,
            f"--worker={config.worker}",
        ]
    )
    if config.tunnel_through_iap:
        cmd.append("--tunnel-through-iap")
    return cmd


def build_remote_shell_command(config: TPUVMConfig, command: str, *, worker: str | None = None) -> list[str]:
    cmd = build_ssh_base(config, worker=worker)
    cmd.append(f"--command={command}")
    return cmd


def build_repo_command(config: TPUVMConfig, inner: str, *, python_env: str | None = None) -> str:
    if python_env:
        inner = f". {shlex.quote(python_env)}/bin/activate && {inner}"
    return f"cd {shlex.quote(config.remote_root)} && {inner}"


def build_python_script_command(config: TPUVMConfig, script_name: str, argv: Sequence[str], *, python_env: str) -> str:
    base = ["python", script_name, *argv]
    return build_repo_command(config, shell_join(base), python_env=python_env)


def build_pytest_command(config: TPUVMConfig, pytest_args: Sequence[str], *, python_env: str) -> str:
    cmd = ["python", "-m", "pytest", *pytest_args]
    return build_repo_command(config, shell_join(cmd), python_env=python_env)


def build_setup_script(options: RepoSyncOptions, config: TPUVMConfig) -> str:
    remote_root = shlex.quote(config.remote_root)
    repo_url = shlex.quote(options.repo_url)
    repo_ref = shlex.quote(options.repo_ref)
    python_env = shlex.quote(options.python_env)
    package_install = ""
    if options.setup_packages:
        package_install = (
            f"$PYBIN -m pip install {' '.join(shlex.quote(p) for p in options.setup_packages)}\n"
        )
    script = textwrap.dedent(
        f"""
        set -euo pipefail
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y git curl python3 python3-venv
        mkdir -p {shlex.quote(str(Path(config.remote_root).parent))}
        if [ ! -d {remote_root}/.git ]; then
          git clone {repo_url} {remote_root}
        fi
        cd {remote_root}
        git fetch --all --tags
        REPO_REF={repo_ref}
        if git rev-parse --verify "$REPO_REF" >/dev/null 2>&1; then
          git checkout "$REPO_REF"
        elif git rev-parse --verify "origin/$REPO_REF" >/dev/null 2>&1; then
          git checkout -B "$REPO_REF" "origin/$REPO_REF"
        else
          echo "WARN: repo ref '$REPO_REF' not found; continuing with current checkout"
        fi
        python3 -m venv {python_env}
        . {python_env}/bin/activate
        PYBIN="$(pwd)/{options.python_env}/bin/python"
        "$PYBIN" -m pip install --upgrade pip setuptools wheel
        {package_install}$PYBIN -m pip install {shlex.quote(JAX_TPU_INSTALL_TARGET)} -f {shlex.quote(JAX_TPU_WHEEL_INDEX)}
        """
    ).strip()
    return f"bash -lc {shlex.quote(script)}"


def should_sync_path(path: Path) -> bool:
    rel_parts = path.parts
    if any(part in SYNC_EXCLUDE_NAMES for part in rel_parts):
        return False
    if path.name in SYNC_EXCLUDE_NAMES:
        return False
    if path.suffix in SYNC_EXCLUDE_SUFFIXES:
        return False
    return True


def create_repo_archive(local_root: Path) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="autokernel-tpu-sync-"))
    archive_path = tmpdir / "repo-sync.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(local_root.rglob("*")):
            rel = path.relative_to(local_root)
            if not should_sync_path(rel):
                continue
            tar.add(path, arcname=str(rel), recursive=False)
    return archive_path


def run_command(cmd: list[str], *, dry_run: bool = False) -> int:
    printable = shell_join(cmd)
    print(printable)
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def capture_command(cmd: list[str], *, dry_run: bool = False) -> tuple[int, str, str]:
    printable = shell_join(cmd)
    print(printable)
    if dry_run:
        return 0, "", ""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    return proc.returncode, stdout, stderr


def ensure_remote_parent(dest: str, config: TPUVMConfig, *, dry_run: bool = False) -> int:
    parent = str(Path(dest).parent)
    return run_command(build_remote_shell_command(config, f"mkdir -p {shlex.quote(parent)}"), dry_run=dry_run)


def get_qr_state(config: TPUVMConfig, *, dry_run: bool = False) -> str:
    code, stdout, _ = capture_command(build_qr_describe_command(config), dry_run=dry_run)
    if dry_run:
        return "DRY_RUN"
    if code != 0:
        return "NOT_FOUND"
    return stdout or "UNKNOWN"


def get_vm_state(config: TPUVMConfig, *, dry_run: bool = False) -> str:
    code, stdout, _ = capture_command(build_status_command(config), dry_run=dry_run)
    if dry_run:
        return "DRY_RUN"
    if code != 0:
        return "NOT_FOUND"
    return stdout or "UNKNOWN"


def wait_for_active(
    config: TPUVMConfig,
    *,
    max_retry: int = DEFAULT_MAX_RETRY,
    retry_interval: int = DEFAULT_RETRY_INTERVAL,
    dry_run: bool = False,
) -> int:
    _print_banner(f"Waiting for TPU activation (up to {max_retry} retries, every {retry_interval}s)")
    for attempt in range(1, max_retry + 1):
        state = get_qr_state(config, dry_run=dry_run)
        if state == "ACTIVE":
            _print_banner(f"TPU queued resource is ACTIVE ({attempt}/{max_retry})")
            return 0
        if state == "FAILED":
            _print_banner("Queued resource entered FAILED state")
            return 1
        _print_banner(f"Queued resource state: {state} ({attempt}/{max_retry})")
        if not dry_run:
            time.sleep(retry_interval)
    _print_banner("Timed out waiting for TPU activation")
    return 1


def wait_for_vm_ready(
    config: TPUVMConfig,
    *,
    max_retry: int = DEFAULT_MAX_RETRY,
    retry_interval: int = DEFAULT_RETRY_INTERVAL,
    dry_run: bool = False,
) -> int:
    _print_banner(f"Waiting for TPU VM READY (up to {max_retry} retries, every {retry_interval}s)")
    for attempt in range(1, max_retry + 1):
        state = get_vm_state(config, dry_run=dry_run)
        if state == "READY":
            _print_banner(f"TPU VM is READY ({attempt}/{max_retry})")
            return 0
        _print_banner(f"TPU VM state: {state} ({attempt}/{max_retry})")
        if not dry_run:
            time.sleep(retry_interval)
    _print_banner("Timed out waiting for TPU VM readiness")
    return 1


def delete_qr(config: TPUVMConfig, *, dry_run: bool = False) -> int:
    return run_command(build_qr_delete_command(config), dry_run=dry_run)


def action_create(args: argparse.Namespace, config: TPUVMConfig) -> int:
    state = get_qr_state(config, dry_run=args.dry_run)
    if state == "ACTIVE":
        _print_banner("Queued resource already ACTIVE")
        return 0
    if state in {"FAILED", "SUSPENDED"}:
        _print_banner(f"Deleting unhealthy queued resource state={state} before recreate")
        delete_qr(config, dry_run=args.dry_run)
    elif state not in {"NOT_FOUND", "UNKNOWN", "DRY_RUN"}:
        _print_banner(f"Queued resource already exists with state={state}; waiting")
        return wait_for_active(config, max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)

    code = run_command(build_qr_create_command(config), dry_run=args.dry_run)
    if code != 0:
        return code
    code = wait_for_active(config, max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)
    if code != 0:
        return code
    return wait_for_vm_ready(config, max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)


def action_status(args: argparse.Namespace, config: TPUVMConfig) -> int:
    qr_state = get_qr_state(config, dry_run=args.dry_run)
    vm_state = get_vm_state(config, dry_run=args.dry_run)
    print(f"queued_resource_state: {qr_state}")
    print(f"vm_state: {vm_state}")
    return 0


def action_delete(args: argparse.Namespace, config: TPUVMConfig) -> int:
    delete_qr(config, dry_run=args.dry_run)
    return run_command(build_tpu_vm_delete_command(config), dry_run=args.dry_run)


def action_ssh(args: argparse.Namespace, config: TPUVMConfig) -> int:
    return run_command(build_ssh_base(config), dry_run=args.dry_run)


def action_run(args: argparse.Namespace, config: TPUVMConfig) -> int:
    if args.activate_env:
        options = repo_options_from_namespace(args)
        command = build_repo_command(config, args.command, python_env=options.python_env)
    else:
        command = args.command
    return run_command(build_remote_shell_command(config, command), dry_run=args.dry_run)


def action_setup(args: argparse.Namespace, config: TPUVMConfig) -> int:
    options = repo_options_from_namespace(args)
    command = build_setup_script(options, config)
    return run_command(build_remote_shell_command(config, command), dry_run=args.dry_run)


def action_sync(args: argparse.Namespace, config: TPUVMConfig) -> int:
    source = str(Path(args.source).expanduser().resolve())
    ensure = ensure_remote_parent(args.dest, config, dry_run=args.dry_run)
    if ensure != 0:
        return ensure
    return run_command(
        build_scp_command(config, source, args.dest, recursive=args.recursive, direction="upload"),
        dry_run=args.dry_run,
    )


def action_sync_repo(args: argparse.Namespace, config: TPUVMConfig) -> int:
    options = repo_options_from_namespace(args)
    archive_path = create_repo_archive(options.local_root)
    remote_archive = f"/tmp/{config.name}-autokernel-sync.tar.gz"
    try:
        ensure = ensure_remote_parent(remote_archive, config, dry_run=args.dry_run)
        if ensure != 0:
            return ensure
        code = run_command(
            build_scp_command(config, str(archive_path), remote_archive, direction="upload"),
            dry_run=args.dry_run,
        )
        if code != 0:
            return code
        extract_script = textwrap.dedent(
            f"""
            set -euo pipefail
            mkdir -p {shlex.quote(config.remote_root)}
            tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(config.remote_root)}
            rm -f {shlex.quote(remote_archive)}
            """
        ).strip()
        return run_command(
            build_remote_shell_command(config, f"bash -lc {shlex.quote(extract_script)}"),
            dry_run=args.dry_run,
        )
    finally:
        try:
            archive_path.unlink(missing_ok=True)
            archive_path.parent.rmdir()
        except Exception:
            pass


def action_pull(args: argparse.Namespace, config: TPUVMConfig) -> int:
    local_dest = str(Path(args.dest).expanduser().resolve())
    parent = Path(local_dest).parent
    parent.mkdir(parents=True, exist_ok=True)
    return run_command(
        build_scp_command(config, args.source, local_dest, recursive=args.recursive, direction="download"),
        dry_run=args.dry_run,
    )


def _prepare_repo_remote_execution(
    args: argparse.Namespace,
    config: TPUVMConfig,
    *,
    skip_setup: bool,
    skip_sync: bool,
) -> int:
    if not skip_setup:
        code = action_setup(args, config)
        if code != 0:
            return code
    if not skip_sync:
        code = action_sync_repo(args, config)
        if code != 0:
            return code
    return 0


def action_test(args: argparse.Namespace, config: TPUVMConfig) -> int:
    prep = _prepare_repo_remote_execution(args, config, skip_setup=args.skip_setup, skip_sync=args.skip_sync)
    if prep != 0:
        return prep
    options = repo_options_from_namespace(args)
    pytest_args = shlex.split(args.pytest_args)
    command = build_pytest_command(config, pytest_args, python_env=options.python_env)
    return run_command(build_remote_shell_command(config, command), dry_run=args.dry_run)


def action_profile_jax(args: argparse.Namespace, config: TPUVMConfig) -> int:
    prep = _prepare_repo_remote_execution(args, config, skip_setup=args.skip_setup, skip_sync=args.skip_sync)
    if prep != 0:
        return prep
    options = repo_options_from_namespace(args)
    profile_args = shlex.split(args.profile_args)
    command = build_python_script_command(config, "profile_jax.py", profile_args, python_env=options.python_env)
    return run_command(build_remote_shell_command(config, command), dry_run=args.dry_run)


def action_watch(args: argparse.Namespace, config: TPUVMConfig, *, repo_args: argparse.Namespace | None = None) -> int:
    consecutive_failures = 0
    while True:
        state = get_vm_state(config, dry_run=args.dry_run)
        if state == "READY":
            consecutive_failures = 0
            _print_banner("TPU READY")
        elif state in {"PREEMPTED", "NOT_FOUND", "TERMINATED"}:
            _print_banner(f"TPU state={state}; recreating")
            delete_qr(config, dry_run=args.dry_run)
            if not args.dry_run:
                time.sleep(5)
            create_ns = argparse.Namespace(max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)
            code = action_create(create_ns, config)
            if code != 0:
                return code
            if repo_args is not None:
                setup_code = action_setup(repo_args, config)
                if setup_code != 0:
                    return setup_code
                sync_code = action_sync_repo(repo_args, config)
                if sync_code != 0:
                    return sync_code
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            _print_banner(f"TPU abnormal state={state}; consecutive_failures={consecutive_failures}")
            if consecutive_failures >= args.failure_threshold:
                _print_banner("Failure threshold exceeded; recreating TPU")
                delete_qr(config, dry_run=args.dry_run)
                if not args.dry_run:
                    time.sleep(5)
                create_ns = argparse.Namespace(max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)
                code = action_create(create_ns, config)
                if code != 0:
                    return code
                if repo_args is not None:
                    setup_code = action_setup(repo_args, config)
                    if setup_code != 0:
                        return setup_code
                    sync_code = action_sync_repo(repo_args, config)
                    if sync_code != 0:
                        return sync_code
                consecutive_failures = 0
        if not args.dry_run:
            time.sleep(args.retry_interval)


def action_watchdog(args: argparse.Namespace, config: TPUVMConfig) -> int:
    create_ns = argparse.Namespace(max_retry=args.max_retry, retry_interval=args.retry_interval, dry_run=args.dry_run)
    code = action_create(create_ns, config)
    if code != 0:
        return code
    code = action_setup(args, config)
    if code != 0:
        return code
    code = action_sync_repo(args, config)
    if code != 0:
        return code
    return action_watch(args, config, repo_args=args)


def delegate_profile_jax_from_namespace(args: argparse.Namespace, forwarded_args: Sequence[str]) -> int:
    config = config_from_namespace(args, prefix="tpu_")
    options = repo_options_from_namespace(args, prefix="tpu_")
    shim = argparse.Namespace(
        dry_run=args.tpu_dry_run,
        skip_setup=args.tpu_skip_setup,
        skip_sync=args.tpu_skip_sync or args.tpu_sync == "none",
        profile_args=shell_join(forwarded_args),
        repo_url=options.repo_url,
        repo_ref=options.repo_ref,
        local_root=str(options.local_root),
        python_env=options.python_env,
        package=list(options.setup_packages[len(DEFAULT_SETUP_PACKAGES):]) or None,
    )
    return action_profile_jax(shim, config)


# ---------------------------------------------------------------------------
# Pallas bench output parsing
# ---------------------------------------------------------------------------

_BENCH_KEY_WHITELIST = frozenset({
    "correctness",
    "throughput_tflops",
    "latency_us",
    "bandwidth_gb_s",
    "pct_peak_compute",
    "pct_peak_bandwidth",
    "bottleneck",
    "speedup_vs_ref",
    "kernel_type",
    "peak_vram_mb",
})

_BENCH_STR_KEYS = frozenset({"correctness", "bottleneck", "kernel_type"})


def parse_bench_output(stdout: str) -> dict:
    """Parse pallas/bench.py text output using a strict key whitelist.

    Only keys in _BENCH_KEY_WHITELIST are captured. All other lines are ignored.
    Last occurrence wins for duplicate keys.
    """
    result: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if ": " not in line:
            continue
        key, _, value = line.partition(": ")
        key = key.strip()
        value = value.strip()
        if key not in _BENCH_KEY_WHITELIST:
            continue
        if key in _BENCH_STR_KEYS:
            result[key] = value
        elif key == "speedup_vs_ref":
            try:
                result[key] = float(value.rstrip("x"))
            except ValueError:
                result[key] = value
        else:
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def _sync_kernel_file(config: TPUVMConfig, local_root: Path, *, dry_run: bool = False) -> int:
    """Sync only kernel.py to the remote TPU (always, even with --skip-sync)."""
    kernel_path = local_root / "kernel.py"
    if not kernel_path.exists():
        _print_banner("WARNING: kernel.py not found at project root, skipping kernel sync")
        return 0
    remote_dest = f"{config.remote_root}/kernel.py"
    return run_command(
        build_scp_command(config, str(kernel_path), remote_dest, direction="upload"),
        dry_run=dry_run,
    )


def run_bench_pallas(
    args: argparse.Namespace,
    config: TPUVMConfig,
    *,
    bench_args_str: str = "",
    kernel_type: str | None = None,
) -> dict:
    """Run pallas/bench.py remotely and return structured results.

    Always syncs kernel.py to TPU. If --skip-sync is NOT set, also syncs the
    full repo. Never raises; always returns a result dict.

    Returns: {"exit_code": int, "metrics": dict, "stdout": str, "stderr": str}
    """
    options = repo_options_from_namespace(args)

    # Always sync kernel.py, even with --skip-sync
    skip_sync = getattr(args, "skip_sync", False)
    skip_setup = getattr(args, "skip_setup", False)

    if not skip_setup:
        code = action_setup(args, config)
        if code != 0:
            return {"exit_code": code, "metrics": {}, "stdout": "", "stderr": "setup failed"}

    if skip_sync:
        # Only sync kernel.py
        code = _sync_kernel_file(config, options.local_root, dry_run=args.dry_run)
        if code != 0:
            return {"exit_code": code, "metrics": {}, "stdout": "", "stderr": "kernel.py sync failed"}
    else:
        code = action_sync_repo(args, config)
        if code != 0:
            return {"exit_code": code, "metrics": {}, "stdout": "", "stderr": "repo sync failed"}

    # Build the remote bench command
    bench_cmd_parts = ["python", "pallas/bench.py"]
    if kernel_type:
        bench_cmd_parts.extend(["--kernel", kernel_type])
    if bench_args_str:
        bench_cmd_parts.extend(shlex.split(bench_args_str))

    remote_cmd = build_repo_command(config, shell_join(bench_cmd_parts), python_env=options.python_env)
    exit_code, stdout, stderr = capture_command(
        build_remote_shell_command(config, remote_cmd), dry_run=args.dry_run
    )

    metrics = parse_bench_output(stdout)
    return {"exit_code": exit_code, "metrics": metrics, "stdout": stdout, "stderr": stderr}


def action_bench_pallas(args: argparse.Namespace, config: TPUVMConfig) -> int:
    """Run pallas/bench.py remotely and print structured results."""
    kernel_type = getattr(args, "kernel_type", None)
    bench_args_str = getattr(args, "bench_args", "") or ""

    result = run_bench_pallas(args, config, bench_args_str=bench_args_str, kernel_type=kernel_type)

    metrics = result["metrics"]
    if metrics:
        _print_banner("Bench results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        _print_banner("No metrics parsed from bench output")
        if result["stderr"]:
            print(f"  stderr: {result['stderr'][:500]}")

    return result["exit_code"]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TPUVMConfig(
        name=args.name,
        zone=args.zone,
        project=_require_project(args.project),
        worker=args.worker,
        remote_root=args.remote_root,
        tunnel_through_iap=not args.no_iap,
        accelerator=args.accelerator,
        runtime=args.runtime,
        service_account=args.service_account,
    )

    if args.action == "create":
        return action_create(args, config)
    if args.action == "status":
        return action_status(args, config)
    if args.action == "delete":
        return action_delete(args, config)
    if args.action == "ssh":
        return action_ssh(args, config)
    if args.action == "run":
        return action_run(args, config)
    if args.action == "setup":
        return action_setup(args, config)
    if args.action == "sync":
        return action_sync(args, config)
    if args.action == "sync-repo":
        return action_sync_repo(args, config)
    if args.action == "pull":
        return action_pull(args, config)
    if args.action == "test":
        return action_test(args, config)
    if args.action == "profile-jax":
        return action_profile_jax(args, config)
    if args.action == "watch":
        return action_watch(args, config)
    if args.action == "watchdog":
        return action_watchdog(args, config)
    if args.action == "bench-pallas":
        return action_bench_pallas(args, config)
    raise SystemExit(f"Unknown action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
