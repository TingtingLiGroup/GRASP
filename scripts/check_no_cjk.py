#!/usr/bin/env python3
"""Fail if CJK characters appear in release-scoped files.

This is a release gate to ensure the published artifacts (sdist + wheel) are
English-only.

Default scope:
  - README.md
  - grasp_tool/**
  - docs/**
  - scripts/**
  - pyproject.toml

It can also scan built artifacts under dist/ (wheel + sdist).
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _iter_text_files(paths: Sequence[Path]) -> Iterator[Path]:
    include_exts = {".py", ".md", ".toml", ".sh", ".txt"}
    for p in paths:
        if p.is_dir():
            for f in p.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() in include_exts:
                    yield f
        elif p.is_file():
            if p.suffix.lower() in include_exts:
                yield p


def _scan_text(content: str) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    for i, line in enumerate(content.splitlines(), 1):
        if _CJK_RE.search(line):
            hits.append((i, line.rstrip("\n")[:240]))
    return hits


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _scan_paths(root: Path, targets: Sequence[str]) -> List[str]:
    msgs: List[str] = []
    files = list(_iter_text_files([root / t for t in targets]))
    for f in files:
        try:
            text = _read_text(f)
        except Exception as e:
            msgs.append(f"ERROR: failed to read {f}: {e}")
            continue
        for line_no, snippet in _scan_text(text):
            rel = str(f.relative_to(root))
            msgs.append(f"{rel}:{line_no}: {snippet}")
    return msgs


def _scan_wheel(path: Path) -> List[str]:
    msgs: List[str] = []
    include_exts = (".py", ".md", ".toml", ".txt")
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(include_exts):
                continue
            try:
                data = zf.read(name)
            except Exception:
                continue
            text = data.decode("utf-8", errors="ignore")
            for line_no, snippet in _scan_text(text):
                msgs.append(f"{path.name}:{name}:{line_no}: {snippet}")
    return msgs


def _scan_sdist(path: Path) -> List[str]:
    msgs: List[str] = []
    include_exts = (".py", ".md", ".toml", ".txt")
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            if not member.name.endswith(include_exts):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                data = f.read()
            except Exception:
                continue
            text = data.decode("utf-8", errors="ignore")
            for line_no, snippet in _scan_text(text):
                msgs.append(f"{path.name}:{member.name}:{line_no}: {snippet}")
    return msgs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail if CJK characters are present")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--scan-dist",
        action="store_true",
        help="Also scan dist/*.whl and dist/*.tar.gz",
    )
    args = parser.parse_args()

    root: Path = args.root
    targets = ["README.md", "pyproject.toml", "grasp_tool", "docs", "scripts"]

    msgs = _scan_paths(root, targets)
    if args.scan_dist:
        dist = root / "dist"
        if dist.exists():
            for whl in sorted(dist.glob("*.whl")):
                msgs.extend(_scan_wheel(whl))
            for tgz in sorted(dist.glob("*.tar.gz")):
                msgs.extend(_scan_sdist(tgz))

    if msgs:
        print("CJK characters detected:")
        for m in msgs:
            print(m)
        return 1

    print("OK: no CJK characters detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
