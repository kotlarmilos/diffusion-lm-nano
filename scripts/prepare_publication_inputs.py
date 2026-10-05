from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

from diffusion_lm_nano.artifacts import sha256


def _load_records(manifest_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    external = manifest.get("external_inputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(external, list) or not isinstance(artifacts, list):
        raise ValueError("manifest must contain external_inputs and artifacts lists")
    checkpoints = [
        entry
        for entry in artifacts
        if isinstance(entry, dict)
        and entry.get("path") == "artifacts/publication-checkpoint.pt"
    ]
    if len(checkpoints) != 1:
        raise ValueError("manifest must contain one publication checkpoint")
    return external, checkpoints[0]


def _verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def _prepare_external(
    root: Path,
    record: dict[str, object],
    *,
    verify_only: bool,
) -> None:
    relative = record.get("path")
    expected = record.get("sha256")
    source_url = record.get("source_url")
    if not all(isinstance(value, str) for value in (relative, expected, source_url)):
        raise ValueError("external input records require path, sha256, and source_url")
    path = root / relative
    if path.is_file() and sha256(path) == expected:
        print(f"verified {path}")
        return
    if verify_only:
        _verify(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(source_url, timeout=60) as response:
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
    _verify(path, expected)
    print(f"downloaded and verified {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/manifest.json"),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify local bytes without downloading missing external inputs",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="prepare base inputs before the publication checkpoint is available",
    )
    args = parser.parse_args()
    manifest_path = args.root / args.manifest
    external, checkpoint = _load_records(manifest_path)
    for record in external:
        if not isinstance(record, dict):
            raise ValueError("external input records must be objects")
        _prepare_external(args.root, record, verify_only=args.verify_only)
    if args.skip_checkpoint:
        return
    checkpoint_path = args.root / str(checkpoint["path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"place publication-checkpoint.pt at {checkpoint_path} before verification"
        )
    _verify(checkpoint_path, str(checkpoint["sha256"]))
    print(f"verified {checkpoint_path}")


if __name__ == "__main__":
    main()
