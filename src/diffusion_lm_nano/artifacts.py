"""Artifact hashing and manifest updates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not target.is_dir():
        raise FileNotFoundError(target)
    for entry in sorted(
        target.rglob("*"),
        key=lambda item: item.relative_to(target).as_posix(),
    ):
        if entry.is_symlink():
            raise ValueError(f"cannot hash symlink in directory: {entry}")
        if not entry.is_file():
            continue
        relative = entry.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(entry.stat().st_size.to_bytes(8, "big"))
        with entry.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def record_artifact(
    manifest_path: str | Path,
    artifact_path: str | Path,
    *,
    kind: str,
    status: str,
    provenance: dict[str, Any],
) -> None:
    if status not in {"measured", "simulated", "pending"}:
        raise ValueError("artifact status must be measured, simulated, or pending")
    manifest_file = Path(manifest_path)
    root = manifest_file.parent.parent
    artifact_file = Path(artifact_path)
    relative = artifact_file.resolve().relative_to(root.resolve()).as_posix()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    entry = {
        "path": relative,
        "kind": kind,
        "status": status,
        "sha256": sha256(artifact_file),
        "bytes": artifact_file.stat().st_size,
        "provenance": provenance,
    }
    entries = [
        existing
        for existing in manifest.get("artifacts", [])
        if existing.get("path") != relative
    ]
    entries.append(entry)
    manifest["artifacts"] = sorted(entries, key=lambda item: item["path"])
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
