#!/usr/bin/env python3
"""Inspect and validate the Hygon architecture visible to PyTorch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path


SUPPORTED_ARCHITECTURES = frozenset({"gfx906", "gfx936"})


def normalize_arch(raw: str) -> str:
    value = raw.strip()
    match = re.fullmatch(r"(gfx[0-9]+)(?::.*)?", value)
    if match is None:
        raise ValueError(f"Invalid Hygon architecture: {raw!r}")
    return match.group(1)


def validate_arches(raw_arches: Sequence[str], expected_count: int = 8) -> str:
    if len(raw_arches) != expected_count:
        raise ValueError(f"Expected {expected_count} devices, found {len(raw_arches)}")

    normalized = [normalize_arch(raw) for raw in raw_arches]
    observed = sorted(set(normalized))
    if len(observed) != 1:
        raise ValueError(f"Mixed Hygon architectures: {', '.join(observed)}")

    architecture = observed[0]
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported Hygon architecture: {architecture}")
    return architecture


def collect_torch_devices() -> dict[str, object]:
    import torch

    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    if not available:
        raise RuntimeError("DCU is not available through torch.cuda")

    devices: list[dict[str, object]] = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "gcnArchName": getattr(properties, "gcnArchName", ""),
                "total_memory": getattr(properties, "total_memory", None),
                "multi_processor_count": getattr(
                    properties, "multi_processor_count", None
                ),
            }
        )

    architecture = validate_arches(
        [str(device["gcnArchName"]) for device in devices]
    )
    return {
        "python": sys.executable,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "available": available,
        "device_count": count,
        "architecture": architecture,
        "devices": devices,
    }


def write_json_atomic(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--print-arch", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_torch_devices()
    write_json_atomic(args.json, report)
    if args.print_arch:
        print(report["architecture"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
