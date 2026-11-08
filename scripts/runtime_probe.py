#!/usr/bin/env python3
"""Read-only runtime probes. Exit 10 means a specific component needs restoring."""
import importlib
import importlib.metadata as metadata
from importlib.machinery import EXTENSION_SUFFIXES
import importlib.util
import json
from pathlib import Path
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


NEEDS_RESTORE = 10
IMPORTS = (
    "accelerate", "av", "cv2", "diffusers", "h11", "httpcore", "httpx",
    "huggingface_hub", "imageio", "janus", "lazy_loader", "msgspec",
    "safetensors", "scipy", "skimage", "tifffile", "tokenizers",
    "transformers", "typer", "xxhash",
)


def dependency_problems(requirements, installed=None):
    """Check pinned roots and their active transitive requirements, not unrelated apps."""
    if installed is None:
        installed = {}
        for dist in metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                installed.setdefault(canonicalize_name(name), dist)
    pending = [(Requirement(line), "runtime requirements") for line in requirements
               if line.strip() and not line.lstrip().startswith("#")]
    visited = set()
    problems = set()
    while pending:
        req, parent = pending.pop()
        name = canonicalize_name(req.name)
        dist = installed.get(name)
        if dist is None:
            problems.add(f"{parent}: missing {req}")
            continue
        if not req.specifier.contains(dist.version, prereleases=True):
            problems.add(f"{parent}: requires {req}; installed {dist.version}")
        key = (name, frozenset(req.extras))
        if key in visited:
            continue
        visited.add(key)
        for text in dist.requires or ():
            child = Requirement(text)
            if child.marker is None or any(
                child.marker.evaluate({"extra": extra}) for extra in ("", *req.extras)
            ):
                pending.append((child, f"{req.name} {dist.version}"))
    return sorted(problems)


def check_dependencies(path):
    problems = dependency_problems(Path(path).read_text(encoding="utf-8-sig").splitlines())
    for problem in problems:
        print(problem)
    return NEEDS_RESTORE if problems else 0


def check_imports():
    for name in IMPORTS:
        importlib.import_module(name)
    from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline  # noqa: F401
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    return 0


def check_registration(name, project):
    expected = Path(project).resolve()
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        print(f"{name}: editable registration is missing")
        return NEEDS_RESTORE
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(expected / "python"):
        print(f"{name}: Python resolves a different or missing source package")
        return NEEDS_RESTORE
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if not direct.get("dir_info", {}).get("editable") or direct.get("url", "").rstrip("/") != expected.as_uri():
        print(f"{name}: editable installation points to a different source directory")
        return NEEDS_RESTORE
    return 0


def check_extension(name, project):
    package = Path(project) / "python" / name
    for suffix in EXTENSION_SUFFIXES:
        if any(package.rglob(f"_{name}{suffix}")):
            return 0
    print(f"{name}: no native extension for this Python ABI under {package}")
    return NEEDS_RESTORE


def check_project_import(name, project):
    # Public imports choose the canonical pybind module name. Importing
    # infinilm.lib._infinilm separately can register CacheConfig twice.
    importlib.import_module("infinicore")
    module = importlib.import_module(name)
    expected = Path(project).resolve() / "python"
    if not Path(module.__file__).resolve().is_relative_to(expected):
        raise RuntimeError(f"{name} loaded from unexpected path: {module.__file__}")
    for loaded in tuple(sys.modules.values()):
        filename = getattr(loaded, "__file__", None)
        if filename and any(Path(filename).name == f"_{name}{suffix}" for suffix in EXTENSION_SUFFIXES):
            if not Path(filename).resolve().is_relative_to(expected):
                raise RuntimeError(f"native extension loaded from unexpected path: {filename}")
            print(f"{name}: {module.__file__}; extension: {filename}")
            return 0
    raise RuntimeError(f"import {name} did not load its native extension")


def write_constraints(path):
    # Preserve the vendor's DCU build and current NumPy ABI when pip resolves
    # missing transitive dependencies. Incompatibility must fail, not replace them.
    lines = []
    for name in ("torch", "numpy", "torchvision", "torchaudio"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name in ("torch", "numpy"):
                raise RuntimeError(f"the base image must provide {name}") from None
            continue
        lines.append(f"{name}=={version}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main():
    command, *args = sys.argv[1:]
    actions = {
        "dependencies": check_dependencies,
        "dependency-imports": check_imports,
        "registration": check_registration,
        "extension": check_extension,
        "project-import": check_project_import,
        "constraints": write_constraints,
    }
    return actions[command](*args)


if __name__ == "__main__":
    sys.exit(main())
