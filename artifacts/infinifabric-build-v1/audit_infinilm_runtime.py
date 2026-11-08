#!/usr/bin/env python3
"""Collect InfiniLM Python runtime evidence without installing packages."""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "accelerate": "accelerate",
    "av": "av",
    "cv2": "opencv-python-headless",
    "decord": "decord",
    "diffusers": "diffusers",
    "einops": "einops",
    "httpx": "httpx",
    "imageio": "imageio",
    "janus": "janus",
    "msgspec": "msgspec",
    "numpy": "numpy",
    "psutil": "psutil",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
    "xxhash": "xxhash",
    "zmq": "pyzmq",
}

PROBE_DISTRIBUTIONS = (
    "accelerate",
    "av",
    "decord",
    "diffusers",
    "huggingface-hub",
    "msgspec",
    "numpy",
    "opencv-python-headless",
    "safetensors",
    "scikit-image",
    "scipy",
    "tokenizers",
    "torch",
    "transformers",
    "xxhash",
)

RUNTIME_PROBE = r'''import json
result = {}
import torch
result["torch"] = torch.__version__
result["torch_device_count"] = torch.cuda.device_count()
import infinicore
result["infinicore"] = str(infinicore.__file__)
import infinilm
result["infinilm"] = str(infinilm.__file__)
from infinilm.multimodal.minimax_h3 import (
    MiniMaxH3ConditionEncoder,
    MiniMaxH3DiffusersAdapter,
    MiniMaxH3TransformerRunner,
)
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration
result["minimax_h3_import_surface"] = "OK"
print("RUNTIME_PROBE_JSON=" + json.dumps(result, sort_keys=True))
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    return parser.parse_args()


def iter_python_files(project_root: Path) -> list[Path]:
    package_root = project_root / "python" / "infinilm"
    files = list(package_root.rglob("*.py"))
    example = project_root / "examples" / "minimax_h3.py"
    if example.is_file():
        files.append(example)
    return sorted(path.resolve() for path in files)


def scan_imports(files: list[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = defaultdict(list)
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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


def distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PROBE_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    files = iter_python_files(project_root)
    if not files:
        print(f"ERROR: no InfiniLM Python sources found below {project_root}", file=sys.stderr)
        return 2

    imports = scan_imports(files)
    local_names = {"infinicore", "infinilm", "_infinicore", "_infinilm"}
    stdlib_names = set(getattr(sys, "stdlib_module_names", ()))
    missing = sorted(
        name
        for name in imports
        if name not in local_names
        and name not in stdlib_names
        and not module_available(name)
    )
    missing_lines = [
        f"{name}\t{IMPORT_TO_DISTRIBUTION.get(name, 'UNMAPPED')}\t{imports[name][0]}"
        for name in missing
    ]
    (run_dir / "missing-imports.txt").write_text(
        "\n".join(missing_lines) + ("\n" if missing_lines else ""), encoding="utf-8"
    )

    try:
        probe = subprocess.run(
            [sys.executable, "-c", RUNTIME_PROBE],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = (error.stderr or "") + "\nERROR: canonical import timed out after 180 seconds\n"
        probe = subprocess.CompletedProcess(error.cmd, 124, stdout, stderr)
    (run_dir / "canonical-import.stdout.log").write_text(probe.stdout, encoding="utf-8")
    (run_dir / "canonical-import.stderr.log").write_text(probe.stderr, encoding="utf-8")

    runtime_ready = probe.returncode == 0
    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "files_scanned": len(files),
        "top_level_imports": sorted(imports),
        "missing_imports": missing,
        "distribution_versions": distribution_versions(),
        "canonical_import_returncode": probe.returncode,
        "runtime_ready": runtime_ready,
    }
    result_path = run_dir / "runtime-audit.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"python={sys.executable}")
    print(f"files_scanned={len(files)}")
    print(f"missing_import_count={len(missing)}")
    for line in missing_lines:
        print(f"missing_import={line}")
    for name, version in result["distribution_versions"].items():
        print(f"distribution={name} version={version or 'NOT_INSTALLED'}")
    print(f"canonical_import={'PASSED' if runtime_ready else 'FAILED'}")
    print(f"canonical_import_returncode={probe.returncode}")
    if probe.stdout:
        print("--- canonical import stdout ---")
        print(probe.stdout.rstrip())
    if probe.stderr:
        print("--- canonical import traceback ---")
        print(probe.stderr.rstrip())
    print(f"runtime_ready={str(runtime_ready).lower()}")
    print(f"audit_json={result_path}")
    print(f"missing_imports_file={run_dir / 'missing-imports.txt'}")
    return 0 if runtime_ready else 3


if __name__ == "__main__":
    raise SystemExit(main())
