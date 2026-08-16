"""Artifact hashing and manifest updates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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

