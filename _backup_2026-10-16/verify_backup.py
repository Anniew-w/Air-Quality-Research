#!/usr/bin/env python3
"""Verify the research snapshot using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

METADATA_DIR = "_backup_2026-10-16"


def inventory(root):
    import os
    files, directories = [], []
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(n for n in names if n != ".git" and
                          not (Path(directory) == root and n == METADATA_DIR))
        for name in names:
            child = Path(directory) / name
            if child.is_symlink():
                raise ValueError("Unexpected directory symlink: " + str(child))
            directories.append(child.relative_to(root).as_posix())
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.name == ".git":
                continue
            if path.is_symlink():
                raise ValueError("Unexpected symlink: " + str(path))
            files.append(path.relative_to(root).as_posix())
    return sorted(files), sorted(directories)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).resolve().with_name("MODIFIED_FILE.json"))
    parser.add_argument("--restore-empty-dirs", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest["files"]
    actual, _ = inventory(root)
    absent = sorted(set(expected) - set(actual))
    uncovered = sorted(set(actual) - set(expected))
    mismatched = []
    for name in sorted(set(expected) & set(actual)):
        entry = expected[name]
        path = root / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            mismatched.append(name)
    if absent or uncovered or mismatched:
        print("FAIL files=%d missing=%d unbacked=%d mismatched=%d" %
              (len(expected), len(absent), len(uncovered), len(mismatched)))
        if not args.summary_only:
            for category, paths in (("missing", absent), ("unbacked", uncovered),
                                    ("mismatched", mismatched)):
                for path in paths[:5]:
                    print("  %s: %s" % (category, path))
        return 1
    if args.restore_empty_dirs:
        for name in manifest.get("directories", []):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid directory entry")
            destination = (root / relative).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                raise ValueError("Directory entry escapes backup root")
            destination.mkdir(parents=True, exist_ok=True)
    print("PASS files=%d bytes=%d sha256=all_match" %
          (len(expected), sum(item["bytes"] for item in expected.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
