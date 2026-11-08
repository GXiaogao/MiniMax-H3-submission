#!/usr/bin/env python3
"""Run the official two-step MiniMax-H3 native FL2VA smoke workload."""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


SMOKE_VERSION = "1.3-minimax-h3-media-structure-scope"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-helper", required=True, type=Path)
    parser.add_argument("--hygon-runtime-helper", required=True, type=Path)
    parser.add_argument("--hygon-runtime-manifest", required=True, type=Path)
    parser.add_argument("--state-manifest", required=True, type=Path)
    parser.add_argument("--local-target", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--example", required=True, type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/public/home/aclsylqidf/shared/MiniMax-H3"),
    )
    parser.add_argument("--first-frame", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text", default="A dog falls asleep")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument(
        "--run-profile",
        choices=("smoke", "formal"),
        default="smoke",
    )
    return parser.parse_args()


def parse_devices(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(devices) != 8:
        raise ValueError(f"exactly eight GPU ids are required: {devices}")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate GPU ids are not allowed: {devices}")
    if min(devices) < 0:
        raise ValueError(f"GPU ids must be nonnegative: {devices}")
    return devices


def validate_device_counts(
    devices: list[int],
    *,
    torch_count: int,
    native_count: int,
) -> None:
    if max(devices) >= torch_count:
        raise RuntimeError(f"requested devices {devices}, but torch reports {torch_count}")
    if max(devices) >= native_count:
        raise RuntimeError(
            f"requested devices {devices}, but InfiniCore reports {native_count}"
        )


def configure_weight_staging(
    infinicore_module: Any,
    devices: list[int],
    native_cpu_count: int,
) -> dict[str, Any]:
    """Route CPU checkpoint tensors through a DCU when native CPU is disabled."""
    if native_cpu_count > 0:
        return {
            "mode": "native-cpu",
            "native_cpu_count": native_cpu_count,
            "target_device": None,
        }
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


def build_official_argv(args: argparse.Namespace) -> list[str]:
    return [
        "fl2va",
        f"--model-path={args.model_path.as_posix()}",
        f"--text={args.text}",
        f"--first-frame={args.first_frame.as_posix()}",
        f"--gpus={args.gpus}",
        f"--num-frames={args.num_frames}",
        f"--steps={args.steps}",
        f"--seed={args.seed}",
        f"--height={args.height}",
        f"--width={args.width}",
        f"--output={args.output.as_posix()}",
    ]


class StageHeartbeat:
    """Print a quiet heartbeat while a long stage is making no log output."""

    def __init__(self, stage: str, interval_seconds: float = 30.0) -> None:
        self.stage = stage
        self.interval_seconds = interval_seconds
        self.started_at = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            elapsed = time.monotonic() - self.started_at
            print(
                f"heartbeat stage={self.stage} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    def __enter__(self) -> "StageHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def load_helper(path: Path, expected_name: str, module_name: str) -> ModuleType:
    resolved = path.resolve()
    if resolved.name != expected_name:
        raise RuntimeError(f"unexpected helper: {resolved}")
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def profile_result_filename(run_profile: str) -> str:
    if run_profile == "smoke":
        return "minimax-h3-native-fl2va-smoke.json"
    if run_profile == "formal":
        return "minimax-h3-native-fl2va-formal.json"
    raise ValueError(f"unsupported run profile: {run_profile}")


def profile_success_label(run_profile: str) -> str:
    if run_profile == "smoke":
        return "NATIVE_FL2VA_SMOKE PASSED"
    if run_profile == "formal":
        return "NATIVE_FL2VA_FORMAL STRUCTURAL_PASSED"
    raise ValueError(f"unsupported run profile: {run_profile}")


def write_result(
    run_dir: Path,
    result: dict[str, Any],
    run_profile: str,
) -> Path:
    path = run_dir / profile_result_filename(run_profile)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
            }
        )
    return snapshot


def load_official_example(path: Path) -> ModuleType:
    resolved = path.resolve()
    module_name = "infinifabric_official_minimax_h3_example"
    spec = importlib.util.spec_from_file_location(
        module_name, resolved
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official example: {resolved}")
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    if not callable(getattr(module, "main", None)):
        raise RuntimeError(f"official example has no callable main(): {resolved}")
    return module


def install_official_stage_tracing(module: ModuleType) -> None:
    """Emit durable markers around every official timing region.

    Native SIGSEGV failures bypass Python exception handling.  The last START
    marker, together with faulthandler's Python stack, identifies the component
    that was active when the process terminated.
    """
    timing_report = getattr(module, "TimingReport", None)
    if timing_report is None or not callable(getattr(timing_report, "measure", None)):
        raise RuntimeError("official example has no TimingReport.measure()")
    if getattr(timing_report, "_infinifabric_native_trace_installed", False):
        return
    original_measure = timing_report.measure

    @contextmanager
    def traced_measure(
        self: Any,
        name: str,
        *,
        synchronize: bool = False,
    ) -> Any:
        print(
            f"native_trace stage={name} status=START "
            f"synchronize={str(synchronize).lower()}",
            flush=True,
        )
        started_at = time.monotonic()
        try:
            with original_measure(self, name, synchronize=synchronize):
                yield
        except BaseException as error:
            print(
                f"native_trace stage={name} status=FAILED "
                f"exception={type(error).__name__} "
                f"elapsed_seconds={time.monotonic() - started_at:.3f}",
                flush=True,
            )
            raise
        else:
            print(
                f"native_trace stage={name} status=DONE "
                f"elapsed_seconds={time.monotonic() - started_at:.3f}",
                flush=True,
            )

    timing_report.measure = traced_measure
    timing_report._infinifabric_native_trace_installed = True


def invoke_official_example(example: Path, official_argv: list[str]) -> None:
    module = load_official_example(example)
    install_official_stage_tracing(module)
    previous_argv = sys.argv[:]
    try:
        sys.argv = [str(example), *official_argv]
        module.main()
    finally:
        sys.argv = previous_argv


def inspect_media(
    output_path: Path,
    *,
    expected_width: int,
    expected_height: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    import av

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"output video is missing or empty: {output_path}")
    with av.open(str(output_path)) as container:
        video_streams = list(container.streams.video)
        audio_streams = list(container.streams.audio)
        if len(video_streams) != 1:
            raise RuntimeError(f"expected one video stream, found {len(video_streams)}")
        if not audio_streams:
            raise RuntimeError("expected at least one audio stream")
        video = video_streams[0]
        width = video.codec_context.width
        height = video.codec_context.height
        decoded_frame_count = sum(1 for _ in container.decode(video=0))
        duration_seconds = (
            float(container.duration / av.time_base)
            if container.duration is not None
            else None
        )
        average_rate = float(video.average_rate) if video.average_rate is not None else None
    if width != expected_width or height != expected_height:
        raise RuntimeError(
            f"unexpected output resolution: {width}x{height}; "
            f"expected {expected_width}x{expected_height}"
        )
    if decoded_frame_count != expected_frame_count:
        raise RuntimeError(
            f"unexpected decoded frame count: {decoded_frame_count}; "
            f"expected {expected_frame_count}"
        )
    return {
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "width": width,
        "height": height,
        "decoded_frame_count": decoded_frame_count,
        "audio_stream_count": len(audio_streams),
        "duration_seconds": duration_seconds,
        "average_frame_rate": average_rate,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "expected_frame_count": expected_frame_count,
    }


def main() -> int:
    faulthandler.enable(all_threads=True)
    print("fatal_signal_trace=enabled", flush=True)
    args = parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "smoke_version": SMOKE_VERSION,
        "run_profile": args.run_profile,
        "result": "FAILED",
        "validation_scope": "media-structure-only",
        "quality_status": "UNASSESSED",
        "last_stage": "initialization",
        "workload": {
            "text": args.text,
            "gpus": args.gpus,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
        },
    }
    try:
        print(f"smoke_version={SMOKE_VERSION}", flush=True)
        devices = parse_devices(args.gpus)
        result["devices"] = devices

        result["last_stage"] = "input-validation"
        model_path = args.model_path.expanduser().resolve()
        first_frame = args.first_frame.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        example = args.example.expanduser().resolve()
        for required in (
            first_frame,
            example,
            model_path / "diffusers" / "model_index.json",
            model_path / "diffusers" / "text_encoder" / "config.json",
            model_path / "diffusers" / "transformer" / "config.json",
        ):
            if not required.is_file():
                raise FileNotFoundError(f"required input is missing: {required}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args.model_path = model_path
        args.first_frame = first_frame
        args.output = output_path
        print("stage=input-validation status=DONE", flush=True)

        result["last_stage"] = "runtime-state-validation"
        manifest = read_json(args.state_manifest.resolve())
        installer = load_helper(
            args.runtime_helper,
            "install_infinilm_private_runtime.py",
            "infinilm_private_runtime_installer",
        )
        if manifest.get("runtime_mode") != "job-local-wheel-staging":
            raise RuntimeError(f"unsupported runtime state: {manifest.get('runtime_mode')!r}")
        if manifest.get("expected_wheels") != installer.EXPECTED_WHEELS:
            raise RuntimeError("runtime manifest and staging helper wheel sets differ")
        wheel_dir = Path(manifest["wheel_dir"]).resolve()
        hygon_runtime = load_helper(
            args.hygon_runtime_helper,
            "prepare_minimax_h3_hygon_runtime.py",
            "minimax_h3_hygon_runtime_preparer",
        )
        print(
            f"stage=runtime-state-validation status=DONE wheel_dir={wheel_dir}",
            flush=True,
        )

        result["last_stage"] = "job-local-runtime-staging"
        print("stage=job-local-runtime-staging status=START", flush=True)
        with StageHeartbeat("job-local-runtime-staging"):
            wheels = installer.verify_wheels(wheel_dir)
            local_target = args.local_target.resolve()
            hygon_target = local_target.parent / "hygon-site-packages"
            installer.install_wheels_direct(wheels, local_target)
            result["private_distributions"] = installer.check_target_contents(local_target)
            if str(local_target) not in sys.path:
                sys.path.insert(0, str(local_target))
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = str(local_target) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
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
        validate_device_counts(
            devices,
            torch_count=torch_count,
            native_count=native_count,
        )
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

        result["last_stage"] = "official-native-fl2va"
        official_argv = build_official_argv(args)
        result["official_argv"] = official_argv
        print("stage=official-native-fl2va status=START", flush=True)
        started_at = time.monotonic()
        with StageHeartbeat("official-native-fl2va"):
            invoke_official_example(example, official_argv)
        for device in devices:
            torch.cuda.synchronize(device)
        result["generation_elapsed_seconds"] = time.monotonic() - started_at
        result["memory_after_generation"] = memory_snapshot(torch, devices)
        print(
            "stage=official-native-fl2va status=DONE "
            f"elapsed_seconds={result['generation_elapsed_seconds']:.3f}",
            flush=True,
        )

        result["last_stage"] = "media-validation"
        print("stage=media-validation status=START", flush=True)
        result["media"] = inspect_media(
            output_path,
            expected_width=args.width,
            expected_height=args.height,
            expected_frame_count=args.num_frames,
        )
        print(
            "stage=media-validation status=DONE "
            f"resolution={result['media']['width']}x{result['media']['height']} "
            f"frames={result['media']['decoded_frame_count']} "
            f"audio_streams={result['media']['audio_stream_count']}",
            flush=True,
        )

        result["result"] = "STRUCTURAL_PASSED" if args.run_profile == "formal" else "PASSED"
        result["last_stage"] = "complete"
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        result_path = write_result(run_dir, result, args.run_profile)
        print(f"RESULT_JSON: {result_path}", flush=True)
        print(
            f"{profile_success_label(args.run_profile)}; "
            "task=fl2va; backend=infinilm; quality_status=UNASSESSED; "
            f"steps={args.steps}; frames={args.num_frames}; output={output_path}",
            flush=True,
        )
        return 0
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        result_path = write_result(run_dir, result, args.run_profile)
        print(
            f"ERROR: stage {result['last_stage']} failed: {result['error']}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        print(f"RESULT_JSON: {result_path}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
