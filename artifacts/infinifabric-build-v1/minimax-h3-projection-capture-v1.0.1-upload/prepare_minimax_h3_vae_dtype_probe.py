#!/usr/bin/env python3
"""Prepare a diagnostic entry copy without overriding the VAE loader's dtypes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path


PATCH_PROFILE = "preserve-vae-fp32-modules-v1"
REPLACEMENTS = (
    ("pipe.vae.to(device=execution_device, dtype=DTYPE)",
     "pipe.vae.to(device=execution_device)"),
    ("pipe.audio_vae.to(device=execution_device, dtype=DTYPE)",
     "pipe.audio_vae.to(device=execution_device)"),
)


def patch_example(source: Path, destination: Path) -> dict[str, str]:
    source = source.expanduser().resolve(strict=True)
    destination = destination.expanduser().absolute()
    if source == destination.resolve():
        raise ValueError("source and destination must be different files")
    original = source.read_bytes()
    text = original.decode("utf-8")
    tree = ast.parse(text, filename=str(source))
    calls = [ast.get_source_segment(text, node) for node in ast.walk(tree)
             if isinstance(node, ast.Call)]
    updated = original
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError(f"expected target exactly once: {old}")
        if calls.count(old) != 1:
            raise ValueError(f"target must be an executable call: {old}")
        updated = updated.replace(old.encode("utf-8"), new.encode("utf-8"), 1)
    # Parse without importing the example or loading a single model component.
    ast.parse(updated.decode("utf-8"), filename=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also rejects symlinks/hardlinks and previous evidence.
    with destination.open("xb") as stream:
        stream.write(updated)
    return {
        "source_path": str(source),
        "destination_path": str(destination.resolve()),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "destination_sha256": hashlib.sha256(updated).hexdigest(),
        "patch_profile": PATCH_PROFILE,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = patch_example(args.source, args.destination)
    except (OSError, ValueError, SyntaxError) as error:
        print(f"ERROR: VAE precision probe preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
