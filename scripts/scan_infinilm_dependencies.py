#!/usr/bin/env python3
"""Find Python imports used by InfiniLM that are unavailable in this environment."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path


# Import names and distribution names sometimes differ. Unknown names are left
# as comments in the generated requirements file for manual review.
IMPORT_TO_DISTRIBUTION = {
    "PIL": "Pillow",
    "OpenSSL": "pyOpenSSL",
    "accelerate": "accelerate",
    "av": "av",
    "cv2": "opencv-python-headless",
    "dateutil": "python-dateutil",
    "decord": "decord",
    "diffusers": "diffusers",
    "dotenv": "python-dotenv",
    "einops": "einops",
    "httpx": "httpx",
    "imageio": "imageio",
    "imageio_ffmpeg": "imageio-ffmpeg",
    "janus": "janus",
    "msgspec": "msgspec",
    "numpy": "numpy",
    "psutil": "psutil",
    "requests": "requests",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "sentencepiece": "sentencepiece",
    "skimage": "scikit-image",
    "soundfile": "soundfile",
    "tiktoken": "tiktoken",
    "tokenizers": "tokenizers",
    "tqdm": "tqdm",
    "transformers": "transformers",
    "xxhash": "xxhash",
    "yaml": "PyYAML",
}


def parse_args() -> argparse.Namespace:
    work_home = Path(os.environ.get("WORK_HOME", "/root/private_data"))
    log_dir = Path(os.environ.get("LOG_DIR", work_home / "logs"))
    parser = argparse.ArgumentParser(
        description=(
            "Statically scan InfiniLM Python sources and write a requirements "
            "file for imports that are not installed."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=work_home / "Projects" / "InfiniLM",
        help="InfiniLM repository root",
    )
    parser.add_argument(
        "--requirements-out",
        type=Path,
        default=log_dir / "infinilm_missing_requirements.txt",
        help="generated requirements file",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help=(
            "files or directories to scan; defaults to python/infinilm and "
            "examples/minimax_h3*.py"
        ),
    )
    return parser.parse_args()


def default_paths(project_root: Path) -> list[Path]:
    paths = [project_root / "python" / "infinilm"]
    for name in ("minimax_h3.py", "minimax_h3_compare.py"):
        candidate = project_root / "examples" / name
        if candidate.exists():
            paths.append(candidate)
    return paths


def python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.py"))
        else:
            raise FileNotFoundError(f"scan path does not exist: {path}")
    return sorted(files)


def imported_names(path: Path) -> list[tuple[str, int]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".", 1)[0], node.lineno))
        elif isinstance(node, ast.Call):
            is_import_module = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            )
            is_dunder_import = isinstance(node.func, ast.Name) and node.func.id == "__import__"
            if (is_import_module or is_dunder_import) and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    name = value.value.lstrip(".").split(".", 1)[0]
                    if name:
                        found.append((name, node.lineno))
    return found


def local_top_level_names(project_root: Path) -> set[str]:
    # _infinilm is the extension built from this repository. It may be imported
    # without a package prefix even though the installed binary lives below
    # python/infinilm, so it must never be treated as a PyPI requirement.
    names: set[str] = {"_infinilm"}
    for base in (project_root, project_root / "python"):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                names.add(child.name)
            elif child.is_file() and child.suffix == ".py":
                names.add(child.stem)
    return names


def module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError):
        return False


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    if not project_root.is_dir():
        print(f"ERROR: project root does not exist: {project_root}", file=sys.stderr)
        return 2

    scan_paths = [path.expanduser() for path in args.paths] or default_paths(project_root)
    files = python_files(scan_paths)
    if not files:
        print("ERROR: no Python files found", file=sys.stderr)
        return 2

    occurrences: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    parse_errors: list[tuple[Path, str]] = []
    for path in files:
        try:
            for name, line in imported_names(path):
                occurrences[name].append((path, line))
        except (OSError, SyntaxError, UnicodeError) as error:
            parse_errors.append((path, str(error)))

    local_names = local_top_level_names(project_root)
    stdlib_names = set(getattr(sys, "stdlib_module_names", ()))
    missing = sorted(
        name
        for name in occurrences
        if name not in stdlib_names
        and name not in local_names
        and not module_is_available(name)
    )

    requirements_out = args.requirements_out.expanduser().resolve()
    requirements_out.parent.mkdir(parents=True, exist_ok=True)
    mapped = sorted({IMPORT_TO_DISTRIBUTION[name] for name in missing if name in IMPORT_TO_DISTRIBUTION})
    unmapped = sorted(name for name in missing if name not in IMPORT_TO_DISTRIBUTION)

    lines = [
        "# Generated by scan_infinilm_dependencies.py",
        "# Review this file before downloading or installing packages.",
        *mapped,
    ]
    if unmapped:
        lines.extend(["", "# Unmapped import names; determine their distribution names:"])
        lines.extend(f"# {name}" for name in unmapped)
    requirements_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"project root: {project_root}")
    print(f"Python files scanned: {len(files)}")
    print(f"top-level imports found: {len(occurrences)}")
    print(f"missing imports: {len(missing)}")
    for name in missing:
        distribution = IMPORT_TO_DISTRIBUTION.get(name, "UNMAPPED")
        first_path, first_line = occurrences[name][0]
        try:
            shown_path = first_path.relative_to(project_root)
        except ValueError:
            shown_path = first_path
        print(f"  {name} -> {distribution} ({shown_path}:{first_line})")

    if parse_errors:
        print(f"WARNING: files with parse errors: {len(parse_errors)}")
        for path, error in parse_errors:
            print(f"  {path}: {error}")
    if unmapped:
        print("WARNING: unmapped imports require manual package-name review")
    print(f"requirements file: {requirements_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
