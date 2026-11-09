#!/usr/bin/env python3
"""Replace repository placeholders after you fork or rename the repo.

Usage:
    python tools/personalize.py --owner octocat --repo unlimited-ocr-everywhere
    python tools/personalize.py --owner octocat --repo unlimited-ocr-everywhere --dry-run

It replaces __OWNER__ / __REPO__ in README.md, notebooks, docs and
CITATION.cff, and YOUR_GITHUB_USERNAME in the naming kit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", ".venv", ".venv-uocr", "__pycache__", "output", "input", "wheel"}
# Files that must keep their placeholders so the tooling and instructions keep
# working after personalization.
SKIP_FILES = {
    ROOT / "tools" / "personalize.py",
    ROOT / "tools" / "validate_notebooks.py",
    ROOT / ".github" / "workflows" / "validate-notebooks.yml",
    ROOT / "REPO_NAME_AND_DESCRIPTION.txt",
}
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".ipynb",
    ".py",
    ".sh",
    ".yml",
    ".yaml",
    ".cff",
    ".toml",
    ".json",
    ".cfg",
    ".ini",
}
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate(value: str, label: str) -> str:
    value = value.strip()
    if not value or not NAME_RE.match(value):
        raise SystemExit(
            f"Invalid {label}: {value!r}. Use letters, numbers, dot, dash or underscore."
        )
    return value


def iter_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path in SKIP_FILES:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", required=True, help="GitHub user or organization")
    parser.add_argument("--repo", required=True, help="repository name")
    parser.add_argument("--dry-run", action="store_true", help="show changes only")
    args = parser.parse_args()

    owner = validate(args.owner, "owner")
    repo = validate(args.repo, "repo")

    replacements = {
        "__OWNER__": owner,
        "__REPO__": repo,
        "YOUR_GITHUB_USERNAME": owner,
    }

    changed = []
    for path in iter_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)
        if new_text == text:
            continue
        changed.append(path)
        if args.dry_run:
            print("would update", path.relative_to(ROOT))
        else:
            path.write_text(new_text, encoding="utf-8")
            print("updated", path.relative_to(ROOT))

    if not changed:
        print("No placeholders found. The repo is already personalized.")
        return 0

    print(f"\n{len(changed)} file(s) {'would be ' if args.dry_run else ''}updated.")
    if not args.dry_run:
        print(
            "Next: review the diff, commit, and create the GitHub repo. "
            "See REPO_NAME_AND_DESCRIPTION.txt for the listing copy."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
