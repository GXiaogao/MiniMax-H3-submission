#!/usr/bin/env python3
"""Collect a read-only InfiniLM/MiniMax-H3 runtime and offline-wheel diagnosis."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version


DIAGNOSTIC_VERSION = "1.0-infinilm-full-runtime-read-only"

IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "accelerate": "accelerate",
    "av": "av",
    "cv2": "opencv-python-headless",
    "decord": "decord",
    "diffusers": "diffusers",
    "httpx": "httpx",
    "imageio": "imageio",
    "janus": "janus",
    "mooncake": "mooncake-transfer-engine",
    "msgspec": "msgspec",
    "numpy": "numpy",
    "psutil": "psutil",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "xxhash": "xxhash",
    "yaml": "PyYAML",
    "zmq": "pyzmq",
}

REQUIRED_DISTRIBUTIONS = (
    "accelerate",
    "av",
    "decord",
    "diffusers",
    "httpx",
    "janus",
    "msgspec",
    "opencv-python-headless",
    "scikit-image",
    "scipy",
    "xxhash",
)

CONDITIONAL_DISTRIBUTIONS = ("mooncake-transfer-engine",)

PROTECTED_STACK = (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "numpy",
)

PROBE_SCRIPTS = {
    "infinicore-native": r'''import json
import infinicore
import torch
from infinicore.lib import _infinicore
hygon_count = _infinicore.get_device_count(_infinicore.Device.Type.HYGON)
torch_count = torch.cuda.device_count()
assert hygon_count == 8, hygon_count
assert torch.cuda.is_available()
assert torch_count == 8, torch_count
print(json.dumps({"infinicore": str(infinicore.__file__), "torch": torch.__version__, "hygon_count": hygon_count, "torch_device_count": torch_count}))
''',
    "infinilm-core": r'''import json
import infinilm
from infinilm.lib import _infinilm
print(json.dumps({"infinilm": str(infinilm.__file__), "InferEngine": hasattr(_infinilm, "InferEngine"), "DistConfig": hasattr(_infinilm, "DistConfig")}))
''',
    "minimax-h3-adapter": r'''from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder, MiniMaxH3DiffusersAdapter, MiniMaxH3TransformerRunner
print("MiniMaxH3 adapter API: OK")
''',
    "accelerate-api": r'''from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
print("Accelerate API: OK")
''',
    "diffusers-minimax-h3-api": r'''from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
print("Diffusers MiniMax-H3 API: OK")
''',
    "transformers-qwen3vl-api": r'''from transformers import Qwen3VLForConditionalGeneration
print("Transformers Qwen3-VL API: OK")
''',
    "video-runtime": r'''import json, shutil
import av, decord, imageio
print(json.dumps({"av": av.__version__, "decord": decord.__version__, "imageio": imageio.__version__, "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")}))
''',
    "image-runtime": r'''import json
import cv2, scipy, skimage
from PIL import Image
print(json.dumps({"cv2": cv2.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__, "Pillow": Image.__module__}))
''',
    "janus-runtime": r'''import janus
print("janus: OK")
''',
    "msgspec-runtime": r'''import msgspec, xxhash, httpx
print("msgspec/xxhash/httpx: OK")
''',
    "mooncake-optional": r'''from mooncake.engine import TransferEngine
print("Mooncake transfer engine: OK")
''',
}

REQUIRED_PROBES = (
    "infinicore-native",
    "infinilm-core",
    "minimax-h3-adapter",
    "accelerate-api",
    "diffusers-minimax-h3-api",
    "transformers-qwen3vl-api",
    "video-runtime",
    "image-runtime",
    "janus-runtime",
    "msgspec-runtime",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--infinicore-python", required=True, type=Path)
    parser.add_argument("--infini-root", required=True, type=Path)
    parser.add_argument("--wheel-manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def iter_python_files(project_root: Path) -> list[Path]:
    roots = [project_root / "python" / "infinilm", project_root / "examples"]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(root.rglob("*.py"))
    return sorted(path.resolve() for path in files)


def scan_imports(files: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as error:
            found["__scan_error__"].append(f"{path}: {error}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    found[alias.name.split(".", 1)[0]].append(f"{path}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found[node.module.split(".", 1)[0]].append(f"{path}:{node.lineno}")
    return found


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_requirements(name: str) -> list[str]:
    try:
        return importlib.metadata.requires(name) or []
    except importlib.metadata.PackageNotFoundError:
        return []


def run_probe(name: str, code: str, run_dir: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=120,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        returncode = 124
        stdout = error.stdout or ""
        stderr = (error.stderr or "") + "\nERROR: probe timed out after 120 seconds\n"
    (run_dir / f"probe-{name}.stdout.log").write_text(stdout, encoding="utf-8")
    (run_dir / f"probe-{name}.stderr.log").write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "required": name in REQUIRED_PROBES,
        "returncode": returncode,
        "ok": returncode == 0,
        "stdout": stdout.rstrip(),
        "stderr_tail": "\n".join(stderr.rstrip().splitlines()[-12:]),
    }


def inspect_native_files(infinicore_python: Path, infini_root: Path) -> dict[str, Any]:
    expected = [
        infini_root / "lib" / "libinfinirt.so",
        infini_root / "lib" / "libinfiniop.so",
        infini_root / "lib" / "libinfiniccl.so",
        infini_root / "lib" / "libinfinicore_cpp_api.so",
    ]
    expected.extend(sorted(infinicore_python.glob("infinicore/lib/_infinicore*.so")))
    records = []
    for path in expected:
        try:
            info = path.stat()
            records.append({
                "path": str(path),
                "exists": True,
                "readable": os.access(path, os.R_OK),
                "mode": stat.filemode(info.st_mode),
                "size": info.st_size,
            })
        except OSError as error:
            records.append({"path": str(path), "exists": False, "readable": False, "error": str(error)})
    return {"files": records, "all_readable": bool(records) and all(item["readable"] for item in records)}


def load_wheel_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("wheels"), list):
        raise ValueError("wheel manifest has no wheels list")
    return data


def inspect_wheels(manifest: dict[str, Any], installed: dict[str, str | None]) -> dict[str, Any]:
    supported = set(sys_tags())
    candidates: list[dict[str, Any]] = []
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in manifest["wheels"]:
        item = dict(raw)
        try:
            parsed_name, parsed_version, _build, tags = parse_wheel_filename(item["filename"])
            item["parsed_name"] = canonicalize_name(parsed_name)
            item["parsed_version"] = str(parsed_version)
            item["compatible_with_current_python"] = bool(tags & supported)
            item["parse_error"] = None
        except (InvalidVersion, ValueError) as error:
            item["compatible_with_current_python"] = False
            item["parse_error"] = str(error)
        normalized = canonicalize_name(item.get("normalized_name") or item.get("name") or "")
        item["normalized_name"] = normalized
        candidates.append(item)
        by_name[normalized].append(item)

    issues: list[dict[str, Any]] = []
    protected_stack_conflicts: list[dict[str, Any]] = []
    protected = {canonicalize_name(name) for name in PROTECTED_STACK}
    candidate_versions = {
        name: {item.get("version") or item.get("parsed_version") for item in items if item.get("compatible_with_current_python")}
        for name, items in by_name.items()
    }

    for item in candidates:
        if not item.get("compatible_with_current_python"):
            continue
        wheel_name = item["normalized_name"]
        wheel_version = item.get("version") or item.get("parsed_version")
        if wheel_name in protected and installed.get(wheel_name) not in (None, wheel_version):
            protected_stack_conflicts.append({
                "kind": "direct-replacement",
                "wheel": item["filename"],
                "distribution": wheel_name,
                "installed": installed.get(wheel_name),
                "candidate": wheel_version,
            })
        for requirement_text in item.get("requires_dist", []):
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement:
                issues.append({"wheel": item["filename"], "requirement": requirement_text, "status": "invalid-metadata"})
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            dependency = canonicalize_name(requirement.name)
            current = installed.get(dependency)
            satisfied = False
            if current:
                try:
                    satisfied = not requirement.specifier or Version(current) in requirement.specifier
                except InvalidVersion:
                    satisfied = False
            available = sorted(version for version in candidate_versions.get(dependency, set()) if version)
            if not satisfied:
                issue = {
                    "wheel": item["filename"],
                    "requirement": requirement_text,
                    "distribution": dependency,
                    "installed": current,
                    "compatible_local_candidates": available,
                }
                issues.append(issue)
                if dependency in protected:
                    protected_stack_conflicts.append({"kind": "dependency-requirement", **issue})

    for owner in PROTECTED_STACK:
        normalized_owner = canonicalize_name(owner)
        if not installed.get(normalized_owner):
            continue
        for requirement_text in installed_requirements(owner):
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement:
                continue
            if requirement.marker and not requirement.marker.evaluate():
                continue
            dependency = canonicalize_name(requirement.name)
            available = sorted(version for version in candidate_versions.get(dependency, set()) if version)
            incompatible = []
            for version in available:
                try:
                    if requirement.specifier and Version(version) not in requirement.specifier:
                        incompatible.append(version)
                except InvalidVersion:
                    incompatible.append(version)
            if incompatible:
                protected_stack_conflicts.append({
                    "kind": "installed-protected-requirement",
                    "installed_distribution": normalized_owner,
                    "installed_version": installed.get(normalized_owner),
                    "requirement": requirement_text,
                    "dependency": dependency,
                    "incompatible_local_candidates": incompatible,
                })

    availability = {}
    for name in (*REQUIRED_DISTRIBUTIONS, *CONDITIONAL_DISTRIBUTIONS):
        normalized = canonicalize_name(name)
        compatible = [item["filename"] for item in by_name.get(normalized, []) if item.get("compatible_with_current_python")]
        availability[normalized] = {
            "installed_version": installed.get(normalized),
            "compatible_local_wheels": compatible,
            "covered": bool(installed.get(normalized) or compatible),
        }
    return {
        "wheel_count": len(candidates),
        "candidates": candidates,
        "required_and_conditional_availability": availability,
        "candidate_dependency_issues": issues,
        "protected_stack_conflicts": protected_stack_conflicts,
    }


def render_missing_text(result: dict[str, Any]) -> str:
    lines = [
        f"diagnostic_version={DIAGNOSTIC_VERSION}",
        f"runtime_ready={str(result['runtime_ready']).lower()}",
        "",
        "[required missing distributions]",
    ]
    lines.extend(result["required_missing"] or ["NONE"])
    lines.extend(["", "[conditional missing distributions]"])
    lines.extend(result["conditional_missing"] or ["NONE"])
    lines.extend(["", "[failed required probes]"])
    lines.extend(result["failed_required_probes"] or ["NONE"])
    lines.extend(["", "[local wheel gaps]"])
    gaps = [name for name, item in result["wheel_analysis"]["required_and_conditional_availability"].items() if not item["covered"]]
    lines.extend(gaps or ["NONE"])
    lines.extend(["", "[protected stack conflicts]"])
    conflicts = result["wheel_analysis"]["protected_stack_conflicts"]
    lines.extend(json.dumps(item, sort_keys=True) for item in conflicts)
    if not conflicts:
        lines.append("NONE")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    files = iter_python_files(project_root)
    imports = scan_imports(files)
    local_modules = {"infinicore", "infinilm", "_infinicore", "_infinilm"}
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    source_missing_imports = sorted(
        name for name in imports
        if name != "__scan_error__" and name not in local_modules and name not in stdlib and not module_available(name)
    )

    distribution_names = {
        canonicalize_name(name) for name in (
            *REQUIRED_DISTRIBUTIONS,
            *CONDITIONAL_DISTRIBUTIONS,
            *PROTECTED_STACK,
            *IMPORT_TO_DISTRIBUTION.values(),
            "Pillow", "imageio", "packaging", "psutil", "pyyaml", "requests", "networkx", "tifffile", "lazy-loader",
        )
    }
    installed = {name: installed_version(name) for name in sorted(distribution_names)}
    required_missing = sorted(name for name in REQUIRED_DISTRIBUTIONS if not installed.get(canonicalize_name(name)))
    conditional_missing = sorted(name for name in CONDITIONAL_DISTRIBUTIONS if not installed.get(canonicalize_name(name)))

    probes = {name: run_probe(name, code, run_dir) for name, code in PROBE_SCRIPTS.items()}
    failed_required_probes = sorted(name for name in REQUIRED_PROBES if not probes[name]["ok"])
    manifest = load_wheel_manifest(args.wheel_manifest.resolve())
    wheel_analysis = inspect_wheels(manifest, installed)
    native_files = inspect_native_files(args.infinicore_python.resolve(), args.infini_root.resolve())
    runtime_ready = not required_missing and not failed_required_probes and native_files["all_readable"]

    result = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "diagnostic_complete": True,
        "runtime_ready": runtime_ready,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "project_root": str(project_root),
        "files_scanned": len(files),
        "source_imports": sorted(imports),
        "source_scan_errors": imports.get("__scan_error__", []),
        "source_missing_imports": source_missing_imports,
        "source_missing_distributions": {
            name: IMPORT_TO_DISTRIBUTION.get(name, "UNMAPPED") for name in source_missing_imports
        },
        "installed_distributions": installed,
        "required_missing": required_missing,
        "conditional_missing": conditional_missing,
        "native_files": native_files,
        "commands": {"ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe")},
        "probes": probes,
        "failed_required_probes": failed_required_probes,
        "wheel_manifest": str(args.wheel_manifest.resolve()),
        "wheel_analysis": wheel_analysis,
    }
    json_path = run_dir / "full-runtime-diagnostic.json"
    missing_path = run_dir / "missing-runtime.txt"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing_path.write_text(render_missing_text(result), encoding="utf-8")

    print(f"diagnostic_version={DIAGNOSTIC_VERSION}")
    print(f"python={sys.executable}")
    print(f"files_scanned={len(files)}")
    print(f"source_missing_imports={','.join(source_missing_imports) or 'NONE'}")
    print(f"required_missing={','.join(required_missing) or 'NONE'}")
    print(f"conditional_missing={','.join(conditional_missing) or 'NONE'}")
    for name, probe in probes.items():
        print(f"probe={name} result={'OK' if probe['ok'] else 'FAILED'} returncode={probe['returncode']}")
        if not probe["ok"] and probe["stderr_tail"]:
            print(f"probe_error={name}: {probe['stderr_tail'].splitlines()[-1]}")
    print(f"failed_required_probes={','.join(failed_required_probes) or 'NONE'}")
    print(f"protected_stack_conflicts={len(wheel_analysis['protected_stack_conflicts'])}")
    print(f"runtime_ready={str(runtime_ready).lower()}")
    print("diagnostic_complete=true")
    print(f"diagnostic_json={json_path}")
    print(f"missing_runtime={missing_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
