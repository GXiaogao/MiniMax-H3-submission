#!/usr/bin/env python3
"""Load one formal MiniMax-H3 native component on the selected DCU devices."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


PROBE_VERSION = "1.2-minimax-h3-native-load-hygon-weight-staging"


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
    parser.add_argument("--task", choices=("fl2va", "ref2va"), default="fl2va")
    parser.add_argument("--component", choices=("encoder", "transformer"), default="encoder")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("at least one GPU is required")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate GPU ids are not allowed: {devices}")
    if min(devices) < 0:
        raise ValueError(f"GPU ids must be nonnegative: {devices}")
    return devices


def configure_weight_staging(
    infinicore_module: Any,
    devices: list[int],
    native_cpu_count: int,
) -> dict[str, Any]:
    """Route CPU checkpoint tensors through a selected DCU if CPU is not built."""
    if native_cpu_count > 0:
        return {
            "mode": "native-cpu",
            "native_cpu_count": native_cpu_count,
            "target_device": None,
        }
    if not devices:
        raise ValueError("at least one device is required for Hygon weight staging")

    target_device = f"cuda:{devices[0]}"
    native_from_torch = infinicore_module.from_torch

    def from_torch_with_hygon_weight_staging(torch_tensor: Any) -> Any:
        if getattr(torch_tensor.device, "type", None) == "cpu":
            torch_tensor = torch_tensor.to(device=target_device)
        return native_from_torch(torch_tensor)

    infinicore_module.from_torch = from_torch_with_hygon_weight_staging
    return {
        "mode": "hygon-gpu-overlay",
        "native_cpu_count": native_cpu_count,
        "target_device": target_device,
    }


def resolve_partition(model_root: Path, task: str) -> Path:
    root = model_root.expanduser()
    if root.name == "diffusers":
        root = root.parent
    if task == "fl2va":
        return root / "diffusers"
    return root / "Ref2VA"


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


def memory_snapshot(torch: Any, devices: list[int]) -> list[dict[str, Any]]:
    snapshot = []
    for device in devices:
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        snapshot.append(
            {
                "device": device,
                "name": torch.cuda.get_device_name(device),
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
                "used_bytes": total_bytes - free_bytes,
                "torch_allocated_bytes": torch.cuda.memory_allocated(device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(device),
            }
        )
    return snapshot


def compact_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "_class_name",
        "architectures",
        "model_type",
        "torch_dtype",
        "dtype",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "attention_head_dim",
    )
    return {key: config[key] for key in keys if key in config}


def write_result(run_dir: Path, result: dict[str, Any]) -> Path:
    path = run_dir / "minimax-h3-native-load-probe.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "result": "FAILED",
        "task": args.task,
        "component": args.component,
        "last_stage": "initialization",
    }
    loaded_component: Any = None
    try:
        print(f"probe_version={PROBE_VERSION}", flush=True)
        devices = parse_devices(args.gpus)
        result["devices"] = devices

        result["last_stage"] = "runtime-state-validation"
        manifest = read_json(args.state_manifest.resolve())
        if manifest.get("runtime_mode") != "job-local-wheel-staging":
            raise RuntimeError(f"unsupported runtime state: {manifest.get('runtime_mode')!r}")
        installer = load_runtime_helper(args.runtime_helper.resolve())
        if manifest.get("expected_wheels") != installer.EXPECTED_WHEELS:
            raise RuntimeError("runtime manifest and staging helper wheel sets differ")
        wheel_dir = Path(manifest["wheel_dir"]).resolve()
        print(f"stage=runtime-state-validation status=DONE wheel_dir={wheel_dir}", flush=True)

        result["last_stage"] = "job-local-runtime-staging"
        print("stage=job-local-runtime-staging status=START", flush=True)
        wheels = installer.verify_wheels(wheel_dir)
        local_target = args.local_target.resolve()
        hygon_target = local_target.parent / "hygon-site-packages"
        installer.install_wheels_direct(wheels, local_target)
        result["private_distributions"] = installer.check_target_contents(local_target)
        result["private_runtime_target"] = str(local_target)
        result["hygon_runtime_target"] = str(hygon_target)
        sys.path.insert(0, str(local_target))
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = str(local_target) + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        hygon_runtime = load_hygon_runtime_helper(args.hygon_runtime_helper.resolve())
        result["hygon_runtime"] = hygon_runtime.stage_runtime(
            args.hygon_runtime_manifest.resolve(),
            installer,
            hygon_target,
            local_target.parent / "hygon-reconstructed",
        )
        print("stage=job-local-runtime-staging status=DONE", flush=True)

        result["last_stage"] = "device-validation"
        import infinicore
        import torch
        from infinicore.lib import _infinicore

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is unavailable")
        torch_count = torch.cuda.device_count()
        native_count = _infinicore.get_device_count(_infinicore.Device.Type.HYGON)
        native_cpu_count = _infinicore.get_device_count(_infinicore.Device.Type.CPU)
        if max(devices) >= torch_count:
            raise RuntimeError(f"requested devices {devices}, but torch reports {torch_count}")
        if native_count < len(devices):
            raise RuntimeError(f"requested {len(devices)} devices, but InfiniCore reports {native_count}")
        result["torch_device_count"] = torch_count
        result["infinicore_device_count"] = native_count
        result["weight_staging"] = configure_weight_staging(
            infinicore, devices, native_cpu_count
        )
        result["memory_before"] = memory_snapshot(torch, devices)
        print(
            f"stage=device-validation status=DONE torch_devices={torch_count} "
            f"infinicore_hygon_devices={native_count} "
            f"weight_staging_mode={result['weight_staging']['mode']} "
            f"weight_staging_target={result['weight_staging']['target_device']}",
            flush=True,
        )

        result["last_stage"] = "formal-native-weight-load"
        model_root = args.model_path.expanduser().resolve()
        partition = resolve_partition(model_root, args.task)
        component_path = partition / ("text_encoder" if args.component == "encoder" else "transformer")
        if not (component_path / "config.json").is_file():
            raise FileNotFoundError(f"component config is missing: {component_path / 'config.json'}")
        result["model_root"] = str(model_root)
        result["component_path"] = str(component_path)
        print(
            f"stage=formal-native-weight-load status=START task={args.task} "
            f"component={args.component} path={component_path}",
            flush=True,
        )
        started_at = time.monotonic()
        if args.component == "encoder":
            from infinilm.multimodal.minimax_h3 import MiniMaxH3ConditionEncoder

            loaded_component = MiniMaxH3ConditionEncoder(
                component_path,
                dtype=torch.bfloat16,
                tp_device_ids=devices,
            )
            engine = loaded_component.engine
        else:
            from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner

            loaded_component = MiniMaxH3TransformerRunner(
                component_path,
                device="nvidia",
                tp_device_ids=devices,
            )
            engine = loaded_component.engine
        for device in devices:
            torch.cuda.synchronize(device)
        elapsed_seconds = time.monotonic() - started_at
        result["load_elapsed_seconds"] = elapsed_seconds
        result["engine_config"] = compact_config(engine.hf_config)
        result["native_parameter_key_count"] = len(engine.state_dict_keyname())
        result["memory_after_load"] = memory_snapshot(torch, devices)
        print(
            f"stage=formal-native-weight-load status=DONE elapsed_seconds={elapsed_seconds:.3f} "
            f"parameter_keys={result['native_parameter_key_count']}",
            flush=True,
        )

        result["result"] = "PASSED"
        result["last_stage"] = "complete"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result_path = write_result(run_dir, result)
        print(f"RESULT_JSON: {result_path}", flush=True)
        print(
            f"NATIVE_LOAD_PROBE PASSED; task={args.task}; component={args.component}; "
            f"devices={','.join(str(item) for item in devices)}",
            flush=True,
        )
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result_path = write_result(run_dir, result)
        print(f"ERROR: stage {result['last_stage']} failed: {result['error']}", file=sys.stderr, flush=True)
        traceback.print_exc()
        print(f"RESULT_JSON: {result_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
