#!/usr/bin/env python3
"""Stage the validated private runtime and inspect MiniMax-H3 model assets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


PREFLIGHT_VERSION = "1.2-minimax-h3-assets-permission-safe-discovery"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-helper", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--local-target", required=True, type=Path)
    parser.add_argument("--work-home", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


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


def safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def normalize_model_root(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.name == "diffusers":
        return candidate.parent
    if candidate.name in ("FL2VA", "Ref2VA"):
        return candidate.parent
    return candidate


def inspect_weight_component(component: Path, label: str) -> tuple[list[str], int, int]:
    errors: list[str] = []
    if not component.is_dir():
        return [f"missing {label} component directory: {component}"], 0, 0

    files = [path for path in component.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    index_files = [path for path in files if path.name.endswith(".index.json")]
    referenced_weights: set[Path] = set()
    for index_path in index_files:
        try:
            payload = read_json(index_path)
            weight_map = payload.get("weight_map", {})
            if not isinstance(weight_map, dict) or not weight_map:
                errors.append(f"empty weight_map: {index_path}")
                continue
            for filename in set(weight_map.values()):
                if not isinstance(filename, str):
                    errors.append(f"invalid weight filename in {index_path}: {filename!r}")
                    continue
                referenced = index_path.parent / filename
                referenced_weights.add(referenced)
                if not referenced.is_file() or referenced.stat().st_size == 0:
                    errors.append(f"missing or empty indexed weight: {referenced}")
        except Exception as error:
            errors.append(f"invalid weight index {index_path}: {error}")

    direct_weights = [
        path
        for path in files
        if path.name.endswith(WEIGHT_SUFFIXES) and not path.name.endswith(".index.json")
    ]
    weights = set(direct_weights) | referenced_weights
    if not weights:
        errors.append(f"no model weight files found under {component}")
    for weight in weights:
        if weight.is_file() and weight.stat().st_size == 0:
            errors.append(f"empty model weight: {weight}")
    return errors, len(files), total_bytes


def inspect_partition(partition: Path, label: str) -> dict[str, Any]:
    errors: list[str] = []
    file_count = 0
    total_bytes = 0
    for component_name in ("text_encoder", "transformer"):
        component_errors, component_files, component_bytes = inspect_weight_component(
            partition / component_name,
            f"{label}/{component_name}",
        )
        errors.extend(component_errors)
        file_count += component_files
        total_bytes += component_bytes
    return {
        "path": str(partition),
        "ready": not errors,
        "errors": errors,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def validate_model_layout(path: Path) -> dict[str, Any]:
    root = normalize_model_root(path)
    modular_dir = root / "diffusers" if (root / "diffusers").is_dir() else root
    errors: list[str] = []
    warnings: list[str] = []
    index_path = modular_dir / "model_index.json"
    model_index: dict[str, Any] = {}
    if not index_path.is_file():
        errors.append(f"missing model_index.json: {index_path}")
    else:
        try:
            model_index = read_json(index_path)
            if model_index.get("_class_name") != "MiniMaxH3ModularPipeline":
                errors.append(
                    f"unexpected _class_name in {index_path}: {model_index.get('_class_name')!r}"
                )
        except Exception as error:
            errors.append(f"invalid model index {index_path}: {error}")

    fl2va = inspect_partition(modular_dir, "FL2VA")
    errors.extend(fl2va["errors"])
    for name, descriptor in model_index.items():
        if name.startswith("_") or name in ("text_encoder", "transformer", "transformer_ref"):
            continue
        if isinstance(descriptor, (list, tuple)) and descriptor and descriptor[0] is not None:
            if not (modular_dir / name).exists():
                errors.append(f"missing modular pipeline component: {modular_dir / name}")

    ref2va_dir = root / "Ref2VA"
    ref2va: dict[str, Any]
    if ref2va_dir.is_dir():
        ref2va = inspect_partition(ref2va_dir, "Ref2VA")
        ref_index_path = ref2va_dir / "model_index.json"
        if not ref_index_path.is_file():
            ref2va["errors"].append(f"missing Ref2VA model_index.json: {ref_index_path}")
        else:
            try:
                ref_index = read_json(ref_index_path)
                if ref_index.get("_minimax_h3", {}).get("partition") != "ref2va":
                    ref2va["errors"].append(
                        f"invalid Ref2VA partition marker: {ref_index_path}"
                    )
            except Exception as error:
                ref2va["errors"].append(f"invalid Ref2VA model index {ref_index_path}: {error}")
        ref2va["ready"] = not ref2va["errors"]
    else:
        ref2va = {
            "path": str(ref2va_dir),
            "ready": False,
            "errors": [],
            "file_count": 0,
            "total_bytes": 0,
        }
        warnings.append(f"optional Ref2VA partition not present: {ref2va_dir}")

    return {
        "model_root": str(root),
        "modular_dir": str(modular_dir),
        "fl2va_ready": not errors,
        "ref2va_ready": bool(ref2va["ready"]),
        "fl2va": fl2va,
        "ref2va": ref2va,
        "errors": errors,
        "warnings": warnings,
    }


def safe_validate_model_layout(path: Path) -> dict[str, Any]:
    try:
        return validate_model_layout(path)
    except Exception as error:
        return {
            "model_root": str(path),
            "modular_dir": None,
            "fl2va_ready": False,
            "ref2va_ready": False,
            "fl2va": None,
            "ref2va": None,
            "errors": [f"cannot inspect candidate {path}: {type(error).__name__}: {error}"],
            "warnings": [],
        }


def bounded_discovery(roots: list[Path], max_depth: int = 4, max_directories: int = 5000) -> list[Path]:
    found: list[Path] = []
    visited = 0
    queue: deque[tuple[Path, int]] = deque()
    for root in roots:
        if safe_is_dir(root):
            queue.append((safe_resolve(root), 0))
    while queue and visited < max_directories:
        directory, depth = queue.popleft()
        visited += 1
        index_path = directory / "model_index.json"
        if safe_is_file(index_path):
            try:
                if read_json(index_path).get("_class_name") == "MiniMaxH3ModularPipeline":
                    found.append(normalize_model_root(directory))
            except Exception:
                pass
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (entry for entry in os.scandir(directory) if entry.is_dir(follow_symlinks=False)),
                key=lambda entry: entry.name,
            )
        except OSError:
            continue
        for child in children:
            queue.append((Path(child.path), depth + 1))
    return list(dict.fromkeys(found))


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    if args.model_path is not None:
        return [normalize_model_root(args.model_path)]
    work_home = args.work_home.resolve()
    shared_root = args.shared_root.resolve()
    candidates = [
        work_home / "models" / "MiniMax-H3",
        work_home / "MiniMax-H3",
        shared_root / "MiniMax-H3",
        shared_root / "models" / "MiniMax-H3",
        shared_root / "models" / "minimax-h3",
    ]
    candidates.extend(
        bounded_discovery([work_home / "models", shared_root], max_depth=4)
    )
    return list(dict.fromkeys(safe_resolve(path) for path in candidates if safe_exists(path)))


def write_result(run_dir: Path, result: dict[str, Any]) -> Path:
    result_path = run_dir / "minimax-h3-assets-preflight.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "preflight_version": PREFLIGHT_VERSION,
        "result": "FAILED",
        "runtime_ready": False,
        "model_ready": False,
        "last_stage": "initialization",
    }
    try:
        print(f"preflight_version={PREFLIGHT_VERSION}", flush=True)
        result["last_stage"] = "state-validation"
        manifest = read_json(args.state_manifest.resolve())
        if manifest.get("runtime_mode") != "job-local-wheel-staging":
            raise RuntimeError(f"unsupported runtime state: {manifest.get('runtime_mode')!r}")

        installer = load_runtime_helper(args.runtime_helper.resolve())
        if manifest.get("expected_wheels") != installer.EXPECTED_WHEELS:
            raise RuntimeError("runtime manifest wheel set does not match the uploaded staging helper")
        wheel_dir = Path(manifest["wheel_dir"]).resolve()
        print(f"stage=state-validation status=DONE wheel_dir={wheel_dir}", flush=True)

        result["last_stage"] = "job-local-runtime-staging"
        print("stage=job-local-runtime-staging status=START", flush=True)
        wheels = installer.verify_wheels(wheel_dir)
        local_target = args.local_target.resolve()
        installer.install_wheels_direct(wheels, local_target)
        result["private_distributions"] = installer.check_target_contents(local_target)
        print("stage=job-local-runtime-staging status=DONE", flush=True)

        result["last_stage"] = "job-local-runtime-validation"
        print("stage=job-local-runtime-validation status=START", flush=True)
        result["runtime_validation"] = installer.validate_runtime(
            local_target, run_dir, "assets-preflight"
        )
        result["runtime_ready"] = True
        print("stage=job-local-runtime-validation status=DONE", flush=True)

        result["last_stage"] = "model-assets-discovery"
        print("stage=model-assets-discovery status=START", flush=True)
        candidates = candidate_paths(args)
        reports = [safe_validate_model_layout(path) for path in candidates]
        result["candidates"] = reports
        ready = [report for report in reports if report["fl2va_ready"]]
        result["model_ready"] = bool(ready)
        if ready:
            result["selected_model"] = ready[0]["model_root"]
            result["ref2va_ready"] = ready[0]["ref2va_ready"]
            print(f"selected_model={ready[0]['model_root']}", flush=True)
            print(f"fl2va_ready=true ref2va_ready={str(ready[0]['ref2va_ready']).lower()}", flush=True)
        else:
            print("model_assets_missing=true", flush=True)
            print(
                f"expected_personal_model_path={args.work_home.resolve() / 'models' / 'MiniMax-H3'}",
                flush=True,
            )
            if not candidates:
                print("model_candidates=NONE", flush=True)
            for report in reports:
                print(f"candidate={report['model_root']}", flush=True)
                for error in report["errors"]:
                    print(f"candidate_error={error}", flush=True)
        print("stage=model-assets-discovery status=DONE", flush=True)

        result["result"] = "PASSED" if result["model_ready"] else "MODEL_ASSETS_MISSING"
        result["last_stage"] = "complete"
        result_path = write_result(run_dir, result)
        print(f"RESULT_JSON: {result_path}", flush=True)
        print(
            "PREFLIGHT COMPLETE; "
            f"runtime_ready={str(result['runtime_ready']).lower()}; "
            f"model_ready={str(result['model_ready']).lower()}",
            flush=True,
        )
        return 0
    except Exception as error:
        result["error"] = str(error)
        result_path = write_result(run_dir, result)
        print(f"ERROR: stage {result['last_stage']} failed: {error}", file=sys.stderr, flush=True)
        print(f"RESULT_JSON: {result_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
