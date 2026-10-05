"""Validate hashes recorded in the release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffusion_lm_nano.artifacts import sha256


def validate_manifest(
    manifest_path: Path,
    *,
    require_all: bool = False,
) -> tuple[int, int]:
    root = manifest_path.parent.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[object] = []
    for section in ("artifacts", "publication_configs", "external_inputs"):
        entries = manifest.get(section)
        if not isinstance(entries, list):
            raise ValueError(f"manifest {section} must be a list")
        records.extend(entries)
    checked = 0
    skipped = 0
    seen: set[str] = set()
    for entry in records:
        if not isinstance(entry, dict):
            raise ValueError("manifest records must be objects")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("manifest records require path and sha256")
        if relative in seen:
            raise ValueError(f"duplicate manifest path {relative}")
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            if require_all:
                raise FileNotFoundError(path)
            skipped += 1
            continue
        if sha256(path) != expected:
            raise ValueError(f"hash mismatch for {relative}")
        checked += 1
    return checked, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.json"),
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail when any recorded file is missing",
    )
    args = parser.parse_args()
    checked, skipped = validate_manifest(
        args.manifest,
        require_all=args.require_all,
    )
    print(f"validated {checked} hashes; skipped {skipped} missing files")


if __name__ == "__main__":
    main()
