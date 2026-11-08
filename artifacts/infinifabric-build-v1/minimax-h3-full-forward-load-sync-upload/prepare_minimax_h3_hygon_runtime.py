#!/usr/bin/env python3
"""Build and publish the offline Hygon runtime additions for MiniMax-H3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


PREPARE_VERSION = "1.2-minimax-h3-hygon-runtime-pytest-dependency"
EINOPS_FILENAME = "einops-0.8.2-py3-none-any.whl"
EINOPS_SHA256 = "54058201ac7087911181bfec4af6091bb59380360f069276601256a76af08193"
TRITON_FILENAME = "triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl"
TRITON_SHA256 = "e397b16365da22508353ce44a45f5be82f8ee58e49853a3df06aa8f6f83d5581"
PYTEST_FILENAME = "pytest-9.0.3-py3-none-any.whl"
PYTEST_SHA256 = "2c5efc453d45394fdd706ade797c0a81091eccd1d6e4bccfcd476e2b8e0ab5d9"
INICONFIG_FILENAME = "iniconfig-2.0.0-py3-none-any.whl"
INICONFIG_SHA256 = "b6a85871a79d2e3b22d2d1b94ac2824226a63c6b741c88f7ae975f18b6778374"
PACKAGING_FILENAME = "packaging-24.2-py3-none-any.whl"
PACKAGING_SHA256 = "09abb1bccd265c01f4a3aa3f7a7db064b36514d2cba19a2f694fe6150451a759"
PLUGGY_FILENAME = "pluggy-1.5.0-py3-none-any.whl"
PLUGGY_SHA256 = "44e1ad92c8ca002de6377e165f3e0f1be63266ab4d554740532335b9d75ea669"
PYGMENTS_FILENAME = "pygments-2.21.0-py3-none-any.whl"
PYGMENTS_SHA256 = "2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9"
FLASH_FILENAME = "flash_attn-2.6.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl"
FLASH_SHA256 = "8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1"
VISION_FILENAME = "vision-0.24.0.tar.gz"
VISION_SHA256 = "f799cdd1d67a3edbcdc6af8fb416fe1b019b512fb426c0314302cd81518a0095"
FLASH_PARTS = (
    (f"{FLASH_FILENAME}.part-000", "bfaa757e842676886c87e896e5829302a7ed9410dd3e10de12280614e89ec488"),
    (f"{FLASH_FILENAME}.part-001", "33113a801a3a34511f6db536a848221f68eddcb5a4db30a05d8d122df11970e6"),
    (f"{FLASH_FILENAME}.part-002", "1e287480287f0a187257e7fcfd50572e62dac5a9323cddce06e7cf8d5d0149a2"),
    (f"{FLASH_FILENAME}.part-003", "5c25cd0275d8da39cbc72d7b0dde24a32ea305c51b3073edb7a9599e4c76781c"),
    (f"{FLASH_FILENAME}.part-004", "dcd9aed482d4b9d5e8d911de05c6449a825110f7de3aa49b9f06acd735534a24"),
    (f"{FLASH_FILENAME}.part-005", "98e8e63ea06dfd32424b935a1afc1119711d0f0f11e4c0ab7c8fe9295a3a3c50"),
    (f"{FLASH_FILENAME}.part-006", "7c4e4b3d0eb30d6a90864e780755b5496a2a35ce22ec72d877642f0b020fd3d1"),
)
RUNTIME_WHEELS = (
    (EINOPS_FILENAME, EINOPS_SHA256),
    (TRITON_FILENAME, TRITON_SHA256),
    (PYTEST_FILENAME, PYTEST_SHA256),
    (INICONFIG_FILENAME, INICONFIG_SHA256),
    (PACKAGING_FILENAME, PACKAGING_SHA256),
    (PLUGGY_FILENAME, PLUGGY_SHA256),
    (PYGMENTS_FILENAME, PYGMENTS_SHA256),
)


VALIDATION_CODE = r'''import json
import os
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional
import einops
import triton
import pytest
import flash_attn

assert hasattr(torch.ops.torchvision, "nms"), "torchvision::nms is not registered"
assert functional.__name__ == "torchvision.transforms.functional"
fa_so_path = os.environ.get("FA_SO_PATH", "")
assert fa_so_path and os.path.isfile(fa_so_path), f"invalid FA_SO_PATH: {fa_so_path!r}"
q = torch.randn(2, 8, 128, 64, device="cuda:0", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
out = F.scaled_dot_product_attention(q, k, v)
torch.cuda.synchronize(0)
print("HYGON_RUNTIME_VALIDATION=" + json.dumps({
    "torch": torch.__version__,
    "torch_hip": torch.version.hip,
    "torchvision": torchvision.__version__,
    "einops": einops.__version__,
    "triton": triton.__version__,
    "pytest": pytest.__version__,
    "flash_attn": flash_attn.__version__,
    "torchvision_nms": True,
    "fa_so_path": fa_so_path,
    "sdpa_shape": list(out.shape),
}, sort_keys=True))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-helper", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--ready-dir", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--local-target", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required input is missing: {path}")
    actual = sha256_path(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch: {path.name}; expected={expected}; actual={actual}")
    print(f"input_checksum=OK sha256={actual} file={path.name}", flush=True)
    return path


def reconstruct_flash(input_dir: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / FLASH_FILENAME
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as output:
        for filename, expected in FLASH_PARTS:
            part = verify_file(input_dir / filename, expected)
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=4 * 1024 * 1024)
    actual = sha256_path(temporary)
    if actual != FLASH_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"reconstructed FlashAttention checksum mismatch: {actual}")
    os.replace(temporary, destination)
    print(f"flash_reconstruction=OK sha256={actual} file={destination}", flush=True)
    return destination


def validate_tar_member(root: Path, member: tarfile.TarInfo) -> None:
    root = root.resolve()
    target = (root / member.name).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"unsafe tar member path: {member.name!r}")
    if not (member.issym() or member.islnk()):
        return

    link = Path(member.linkname)
    if link.is_absolute():
        raise RuntimeError(f"unsafe tar link: {member.name!r} -> {member.linkname!r}")
    link_target = ((target.parent if member.issym() else root) / link).resolve()
    if link_target != root and root not in link_target.parents:
        raise RuntimeError(f"unsafe tar link: {member.name!r} -> {member.linkname!r}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as stream:
        members = stream.getmembers()
        for member in members:
            validate_tar_member(root, member)
        stream.extractall(destination, members=members)


def load_runtime_helper(path: Path) -> ModuleType:
    if path.name != "install_infinilm_private_runtime.py":
        raise RuntimeError(f"unexpected runtime helper: {path}")
    spec = importlib.util.spec_from_file_location("infinilm_private_runtime_installer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load runtime helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def wheel_identity(path: Path) -> tuple[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise RuntimeError(f"expected one wheel METADATA file: {path}")
        metadata = archive.read(names[0]).decode("utf-8", errors="replace")
    fields: dict[str, str] = {}
    for line in metadata.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            if key in ("Name", "Version") and key not in fields:
                fields[key] = value
    if set(fields) != {"Name", "Version"}:
        raise RuntimeError(f"wheel identity is incomplete: {path}")
    return fields["Name"], fields["Version"]


def stage_runtime(
    manifest_path: Path,
    installer: ModuleType,
    local_target: Path,
    scratch_dir: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path.resolve())
    if manifest.get("schema_version") != 1 or manifest.get("runtime") != "minimax-h3-hygon":
        raise RuntimeError(f"unsupported Hygon runtime manifest: {manifest_path}")
    input_dir = Path(manifest["input_dir"]).resolve()
    ready_dir = Path(manifest["ready_dir"]).resolve()
    runtime_wheels = [verify_file(input_dir / filename, expected) for filename, expected in RUNTIME_WHEELS]
    flash = reconstruct_flash(input_dir, scratch_dir)
    torchvision_info = manifest.get("torchvision", {})
    torchvision = Path(torchvision_info.get("path", "")).resolve()
    expected_torchvision = torchvision_info.get("sha256")
    if not expected_torchvision or torchvision.parent != ready_dir:
        raise RuntimeError("invalid torchvision entry in Hygon runtime manifest")
    verify_file(torchvision, expected_torchvision)

    wheels = [*runtime_wheels, flash, torchvision]
    local_target.mkdir(parents=True, exist_ok=True)
    installer.install_wheels_direct(wheels, local_target)
    distributions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions(path=[str(local_target)])
    }
    flash_candidates = sorted(local_target.glob("flash_attn_2_cuda*.so"))
    if len(flash_candidates) != 1:
        raise RuntimeError(f"expected one flash_attn_2_cuda shared object, found {len(flash_candidates)}")
    os.environ["FA_SO_PATH"] = str(flash_candidates[0].resolve())
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(local_target.resolve()) + (
        os.pathsep + existing if existing else ""
    )
    if str(local_target.resolve()) not in sys.path:
        sys.path.insert(0, str(local_target.resolve()))
    return {
        "distributions": distributions,
        "fa_so_path": os.environ["FA_SO_PATH"],
        "wheels": [str(path) for path in wheels],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "prepare_version": PREPARE_VERSION,
        "result": "FAILED",
        "last_stage": "initialization",
    }
    try:
        if not 1 <= args.jobs <= 64:
            raise ValueError(f"jobs must be between 1 and 64: {args.jobs}")
        input_dir = args.input_dir.resolve()
        ready_dir = args.ready_dir.resolve()
        work_dir = args.work_dir.resolve()
        local_target = args.local_target.resolve()
        installer = load_runtime_helper(args.runtime_helper.resolve())
        work_dir.mkdir(parents=True, exist_ok=True)
        ready_dir.mkdir(parents=True, exist_ok=True)

        result["last_stage"] = "input-verification"
        for filename, expected in RUNTIME_WHEELS:
            verify_file(input_dir / filename, expected)
        vision = verify_file(input_dir / VISION_FILENAME, VISION_SHA256)
        reconstruct_flash(input_dir, work_dir / "reconstructed")

        result["last_stage"] = "torchvision-source-extraction"
        source_parent = work_dir / "torchvision-source"
        safe_extract_tar(vision, source_parent)
        candidates = [path for path in source_parent.iterdir() if (path / "setup.py").is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"expected one torchvision source directory, found {len(candidates)}")
        source_dir = candidates[0]

        result["last_stage"] = "torchvision-build"
        dist_dir = work_dir / "torchvision-dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "BUILD_VERSION": "0.24.0",
                "MAX_JOBS": str(args.jobs),
                "FORCE_CUDA": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(dist_dir)],
            cwd=source_dir,
            env=environment,
            check=False,
        )
        result["torchvision_build_seconds"] = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(f"torchvision build failed with exit code {completed.returncode}")
        built = list(dist_dir.glob("torchvision-0.24.0-*.whl"))
        if len(built) != 1:
            raise RuntimeError(f"expected one torchvision wheel, found {len(built)}")
        name, version = wheel_identity(built[0])
        if name.lower() != "torchvision" or version != "0.24.0":
            raise RuntimeError(f"unexpected built wheel identity: {name} {version}")

        result["last_stage"] = "ready-wheel-publication"
        published = ready_dir / built[0].name
        temporary = published.with_name(f"{published.name}.tmp.{os.getpid()}")
        shutil.copyfile(built[0], temporary)
        temporary.chmod(0o644)
        os.replace(temporary, published)
        published_hash = sha256_path(published)
        draft_manifest = {
            "schema_version": 1,
            "runtime": "minimax-h3-hygon",
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "ready_dir": str(ready_dir),
            "flash_sha256": FLASH_SHA256,
            "torchvision": {
                "path": str(published),
                "sha256": published_hash,
                "version": version,
            },
        }
        draft_path = run_dir / "minimax-h3-hygon-runtime.draft.json"
        write_json(draft_path, draft_manifest)

        result["last_stage"] = "runtime-validation"
        staged = stage_runtime(draft_path, installer, local_target, work_dir / "validation-reconstructed")
        validation_environment = os.environ.copy()
        validation_environment["FA_SO_PATH"] = staged["fa_so_path"]
        completed = subprocess.run(
            [sys.executable, "-c", VALIDATION_CODE],
            env=validation_environment,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Hygon runtime validation failed with exit code {completed.returncode}")

        result["last_stage"] = "state-publication"
        manifest_path = args.state_manifest.resolve()
        manifest_temp = manifest_path.with_name(f"{manifest_path.name}.tmp.{os.getpid()}")
        write_json(manifest_temp, draft_manifest)
        manifest_temp.chmod(0o644)
        os.replace(manifest_temp, manifest_path)
        result.update(
            {
                "result": "PASSED",
                "last_stage": "complete",
                "state_manifest": str(manifest_path),
                "torchvision_wheel": str(published),
                "torchvision_sha256": published_hash,
                "staged_runtime": staged,
            }
        )
        write_json(run_dir / "prepare-minimax-h3-hygon-runtime.json", result)
        print(f"state_manifest={manifest_path}", flush=True)
        print(f"torchvision_wheel={published}", flush=True)
        print("RESULT: PASSED; exit_code=0; last_stage=complete", flush=True)
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        write_json(run_dir / "prepare-minimax-h3-hygon-runtime.json", result)
        print(f"ERROR: stage {result['last_stage']} failed: {error}", file=sys.stderr, flush=True)
        print(f"RESULT: FAILED; exit_code=1; last_stage={result['last_stage']}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
