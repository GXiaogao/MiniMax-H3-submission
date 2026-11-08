#!/usr/bin/env python3
"""Audit Hygon runtime prerequisites before loading formal MiniMax-H3 weights."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


AUDIT_VERSION = "1.1-minimax-h3-hygon-runtime-overlay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-helper", required=True, type=Path)
    parser.add_argument("--hygon-runtime-helper", required=True, type=Path)
    parser.add_argument("--hygon-runtime-manifest", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--local-target", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/public/home/aclsylqidf/shared/MiniMax-H3"),
    )
    parser.add_argument("--device", type=int, default=0)
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


def load_hygon_runtime_helper(path: Path) -> ModuleType:
    if path.name != "prepare_minimax_h3_hygon_runtime.py":
        raise RuntimeError(f"unexpected Hygon runtime helper: {path}")
    spec = importlib.util.spec_from_file_location("minimax_h3_hygon_runtime_preparer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Hygon runtime helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def package_probe(distribution: str, module_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "distribution": distribution,
        "module": module_name,
        "version": None,
        "import_ok": False,
        "module_file": None,
        "error": None,
    }
    errors: list[str] = []
    try:
        result["version"] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f"distribution not found: {distribution}")
    try:
        module = importlib.import_module(module_name)
        result["import_ok"] = True
        result["module_file"] = str(getattr(module, "__file__", None))
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["error"] = " | ".join(errors) or None
    return result


def discover_flash_attention_shared_objects() -> dict[str, Any]:
    candidates: set[Path] = set()
    configured = os.environ.get("FA_SO_PATH", "").strip()
    if configured:
        candidates.add(Path(configured).expanduser())

    try:
        spec = importlib.util.find_spec("flash_attn")
    except Exception:
        spec = None
    if spec and spec.origin:
        package_dir = Path(spec.origin).resolve().parent
        for directory in (package_dir, package_dir.parent):
            candidates.update(directory.glob("flash_attn_2_cuda*.so"))

    for entry in sys.path:
        if not entry:
            continue
        directory = Path(entry).expanduser()
        try:
            if directory.is_dir():
                candidates.update(directory.glob("flash_attn_2_cuda*.so"))
        except OSError:
            continue

    normalized = sorted({path.resolve() for path in candidates if path.is_file()})
    if not configured and normalized:
        os.environ["FA_SO_PATH"] = str(normalized[0])
    effective = os.environ.get("FA_SO_PATH", "").strip()
    return {
        "configured_value": configured or None,
        "effective_value": effective or None,
        "candidates": [str(path) for path in normalized],
        "shared_object_ready": bool(effective and Path(effective).is_file()),
    }


def probe_torchvision(torch: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "transforms_ok": False,
        "nms_registered": False,
        "error": None,
    }
    try:
        import torchvision
        from torchvision.transforms import functional

        result["version"] = torchvision.__version__
        result["module_file"] = str(torchvision.__file__)
        result["transforms_module"] = functional.__name__
        result["transforms_ok"] = True
        result["nms_registered"] = hasattr(torch.ops.torchvision, "nms")
        if not result["nms_registered"]:
            result["error"] = "torchvision::nms is not registered"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def resolve_diffusers_root(model_path: Path) -> Path:
    root = model_path.expanduser().resolve()
    return root if root.name == "diffusers" else root / "diffusers"


def probe_processor(model_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"ready": False, "path": None, "error": None}
    diffusers_root = resolve_diffusers_root(model_path)
    candidates = [diffusers_root / "processor", diffusers_root]
    errors: list[str] = []
    try:
        from transformers import AutoProcessor
    except Exception as error:
        result["error"] = f"cannot import AutoProcessor: {type(error).__name__}: {error}"
        return result

    for candidate in candidates:
        if not candidate.is_dir():
            errors.append(f"missing directory: {candidate}")
            continue
        try:
            processor = AutoProcessor.from_pretrained(
                candidate,
                local_files_only=True,
                trust_remote_code=False,
            )
            result.update(
                {
                    "ready": True,
                    "path": str(candidate),
                    "class": type(processor).__name__,
                    "image_processor_class": (
                        type(processor.image_processor).__name__
                        if getattr(processor, "image_processor", None) is not None
                        else None
                    ),
                }
            )
            return result
        except Exception as error:
            errors.append(f"{candidate}: {type(error).__name__}: {error}")
    result["error"] = " | ".join(errors)
    return result


def probe_sdpa(torch: Any, device_index: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "device": device_index,
        "flash_forward_ok": False,
        "allclose": False,
        "max_abs_error": None,
        "mean_abs_error": None,
        "error": None,
    }
    try:
        import torch.nn.functional as functional

        backend = torch.backends.cuda
        required = (
            "enable_math_sdp",
            "enable_mem_efficient_sdp",
            "enable_flash_sdp",
            "math_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "flash_sdp_enabled",
        )
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise RuntimeError(f"missing PyTorch SDPA backend controls: {missing}")

        original = {
            "math": backend.math_sdp_enabled(),
            "mem_efficient": backend.mem_efficient_sdp_enabled(),
            "flash": backend.flash_sdp_enabled(),
        }
        torch.manual_seed(42)
        device = torch.device(f"cuda:{device_index}")
        query = torch.randn(2, 8, 128, 64, device=device, dtype=torch.bfloat16)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        try:
            backend.enable_flash_sdp(False)
            backend.enable_mem_efficient_sdp(False)
            backend.enable_math_sdp(True)
            reference = functional.scaled_dot_product_attention(query, key, value)
            torch.cuda.synchronize(device_index)

            backend.enable_math_sdp(False)
            backend.enable_mem_efficient_sdp(False)
            backend.enable_flash_sdp(True)
            actual = functional.scaled_dot_product_attention(query, key, value)
            torch.cuda.synchronize(device_index)
            result["flash_forward_ok"] = True
            difference = (actual.float() - reference.float()).abs()
            result["max_abs_error"] = float(difference.max().item())
            result["mean_abs_error"] = float(difference.mean().item())
            result["allclose"] = bool(
                torch.allclose(actual.float(), reference.float(), rtol=0.02, atol=0.02)
            )
            result["shape"] = list(actual.shape)
            result["dtype"] = str(actual.dtype)
        finally:
            backend.enable_math_sdp(original["math"])
            backend.enable_mem_efficient_sdp(original["mem_efficient"])
            backend.enable_flash_sdp(original["flash"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def evaluate_readiness(report: dict[str, Any]) -> dict[str, Any]:
    missing: list[str] = []
    hardware = report.get("hardware", {})
    if not hardware.get("torch_cuda_available"):
        missing.append("torch.cuda")
    if int(hardware.get("torch_device_count", 0)) < 8:
        missing.append("8 Hygon devices")
    for name in ("torchvision", "einops", "triton", "flash-attn"):
        if not report.get("packages", {}).get(name, {}).get("import_ok"):
            missing.append(name)
    torchvision = report.get("torchvision", {})
    if not torchvision.get("transforms_ok"):
        missing.append("torchvision.transforms")
    if not torchvision.get("nms_registered"):
        missing.append("torchvision::nms")
    if not report.get("processor", {}).get("ready"):
        missing.append("Qwen3-VL processor")
    if not report.get("flash_attention", {}).get("shared_object_ready"):
        missing.append("FA_SO_PATH")
    sdpa = report.get("sdpa", {})
    if not sdpa.get("flash_forward_ok"):
        missing.append("FlashAttention SDPA forward")
    if not sdpa.get("allclose"):
        missing.append("FlashAttention SDPA numerical check")
    return {"ready": not missing, "missing": missing}


def write_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "minimax-h3-hygon-runtime-audit.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "completed": False,
        "last_stage": "initialization",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        print(f"audit_version={AUDIT_VERSION}", flush=True)
        report["last_stage"] = "runtime-state-validation"
        manifest = read_json(args.state_manifest.resolve())
        if manifest.get("runtime_mode") != "job-local-wheel-staging":
            raise RuntimeError(f"unsupported runtime state: {manifest.get('runtime_mode')!r}")
        installer = load_runtime_helper(args.runtime_helper.resolve())
        if manifest.get("expected_wheels") != installer.EXPECTED_WHEELS:
            raise RuntimeError("runtime manifest and staging helper wheel sets differ")
        wheel_dir = Path(manifest["wheel_dir"]).resolve()
        print(f"stage=runtime-state-validation status=DONE wheel_dir={wheel_dir}", flush=True)

        report["last_stage"] = "job-local-runtime-staging"
        print("stage=job-local-runtime-staging status=START", flush=True)
        wheels = installer.verify_wheels(wheel_dir)
        local_target = args.local_target.resolve()
        hygon_target = local_target.parent / "hygon-site-packages"
        installer.install_wheels_direct(wheels, local_target)
        report["private_distributions"] = installer.check_target_contents(local_target)
        report["private_runtime_target"] = str(local_target)
        report["hygon_runtime_target"] = str(hygon_target)
        sys.path.insert(0, str(local_target))
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(local_target) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        hygon_runtime = load_hygon_runtime_helper(args.hygon_runtime_helper.resolve())
        report["hygon_runtime"] = hygon_runtime.stage_runtime(
            args.hygon_runtime_manifest.resolve(),
            installer,
            hygon_target,
            local_target.parent / "hygon-reconstructed",
        )
        print("stage=job-local-runtime-staging status=DONE", flush=True)

        report["last_stage"] = "package-and-device-audit"
        import torch

        report["flash_attention"] = discover_flash_attention_shared_objects()
        packages = {
            "torchvision": package_probe("torchvision", "torchvision"),
            "einops": package_probe("einops", "einops"),
            "triton": package_probe("triton", "triton"),
            "flash-attn": package_probe("flash-attn", "flash_attn"),
        }
        report["packages"] = packages
        report["hardware"] = {
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
        report["torchvision"] = probe_torchvision(torch)
        report["processor"] = probe_processor(args.model_path)
        if torch.cuda.is_available() and 0 <= args.device < torch.cuda.device_count():
            report["sdpa"] = probe_sdpa(torch, args.device)
        else:
            report["sdpa"] = {
                "device": args.device,
                "flash_forward_ok": False,
                "allclose": False,
                "error": "requested audit device is unavailable",
            }

        report["readiness"] = evaluate_readiness(report)
        report["completed"] = True
        report["last_stage"] = "complete"
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        report_path = write_report(run_dir, report)
        for name, value in packages.items():
            print(
                f"package={name} version={value['version'] or 'NOT_INSTALLED'} "
                f"import_ok={str(value['import_ok']).lower()}",
                flush=True,
            )
        print(
            f"torchvision_transforms={str(report['torchvision']['transforms_ok']).lower()} "
            f"torchvision_nms={str(report['torchvision']['nms_registered']).lower()}",
            flush=True,
        )
        print(
            f"fa_so_path={report['flash_attention']['effective_value'] or 'NOT_FOUND'}",
            flush=True,
        )
        print(
            f"sdpa_flash_forward={str(report['sdpa']['flash_forward_ok']).lower()} "
            f"sdpa_allclose={str(report['sdpa']['allclose']).lower()} "
            f"max_abs_error={report['sdpa'].get('max_abs_error')}",
            flush=True,
        )
        for item in report["readiness"]["missing"]:
            print(f"missing_prerequisite={item}", flush=True)
        print(f"AUDIT_JSON: {report_path}", flush=True)
        print(
            "HYGON_RUNTIME_AUDIT COMPLETE; "
            f"native_load_prerequisites_ready={str(report['readiness']['ready']).lower()}",
            flush=True,
        )
        return 0
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        report_path = write_report(run_dir, report)
        print(
            f"ERROR: stage {report['last_stage']} failed: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        print(f"AUDIT_JSON: {report_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
