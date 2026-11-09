#!/usr/bin/env python3
"""Validate notebook JSON and Python syntax without executing the notebooks."""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
PYTHON_DIRS = [ROOT / "scripts", ROOT / "tools", ROOT / "tools" / "templates", ROOT / "vendor"]

MAGIC_RE = re.compile(r"^\s*[%!?]")
GET_IPYTHON_RE = re.compile(r"get_ipython\(\)")


def strip_ipython(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if MAGIC_RE.match(line):
            continue
        if GET_IPYTHON_RE.search(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def check_notebook(path: Path, strict_placeholders: bool) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path.name}: invalid JSON: {exc}"]

    if data.get("nbformat") != 4:
        errors.append(f"{path.name}: expected nbformat 4")
    if not isinstance(data.get("metadata", {}).get("kernelspec"), dict):
        errors.append(f"{path.name}: missing kernelspec metadata")

    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        return [f"{path.name}: no cells"]

    code_count = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"{path.name}: cell {index} is not an object")
            continue
        if cell.get("cell_type") not in ("markdown", "code", "raw"):
            errors.append(f"{path.name}: cell {index} has unknown cell_type")
        if not isinstance(cell.get("source"), list):
            errors.append(f"{path.name}: cell {index} source is not a list")
        if not isinstance(cell.get("metadata"), dict):
            errors.append(f"{path.name}: cell {index} metadata is not an object")
        if cell.get("cell_type") != "code":
            continue
        if "outputs" not in cell or "execution_count" not in cell:
            errors.append(f"{path.name}: code cell {index} is missing outputs/execution_count")
        code_count += 1
        source = "".join(cell.get("source", []))
        cleaned = strip_ipython(source)
        if not cleaned.strip():
            continue
        try:
            compile(cleaned, f"{path.name}:cell{index}", "exec")
        except SyntaxError as exc:
            errors.append(f"{path.name}: cell {index}: {exc}")
        if strict_placeholders and ("__OWNER__" in source or "__REPO__" in source):
            errors.append(f"{path.name}: cell {index}: personalization placeholder left in code")

    if code_count == 0:
        errors.append(f"{path.name}: no code cells")

    text = path.read_text(encoding="utf-8")
    if strict_placeholders and ("__OWNER__" in text or "__REPO__" in text):
        errors.append(f"{path.name}: personalization placeholder still present")

    return errors


def check_python(path: Path) -> list[str]:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return [f"{path.relative_to(ROOT)}: {exc.msg}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if __OWNER__/__REPO__ placeholders remain",
    )
    args = parser.parse_args()

    errors: list[str] = []
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not notebooks:
        errors.append("no notebooks found; run tools/build_notebooks.py")

    for path in notebooks:
        errors.extend(check_notebook(path, args.strict))

    for directory in PYTHON_DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.py")):
            errors.extend(check_python(path))

    if errors:
        print("Validation failed:\n")
        for error in errors:
            print(" -", error)
        return 1

    print(f"OK: {len(notebooks)} notebook(s) and all Python files are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
