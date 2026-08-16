from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import urllib.request


FILES = {
    "checkpoints/gpt2-nano.pt": (
        "https://huggingface.co/kotlarmilos/gpt2-nano/resolve/main/"
        "checkpoints/final.pt",
        "9a293ff319e7e11e7c5c553a5e4957a996493311d04b097933c1e2fcb49bf86e",
    ),
    "data/train.bin": (
        "https://huggingface.co/kotlarmilos/gpt2-nano/resolve/main/"
        "bpe-shards/train_000000.bin",
        "f08410ea036cc56b78ffc3f6edbf94f5a1d469aeb7c708bb56e4ce544623a69e",
    ),
    "data/bpe-tokenizer/vocab.json": (
        "https://huggingface.co/kotlarmilos/gpt2-nano/resolve/main/"
        "bpe-tokenizer/vocab.json",
        "459dba6d2c6ae095718de3db92119bf5cefe19eb9dc0a88781687a5f74f7a4a3",
    ),
    "data/bpe-tokenizer/merges.json": (
        "https://huggingface.co/kotlarmilos/gpt2-nano/resolve/main/"
        "bpe-tokenizer/merges.json",
        "fcdc81f4b7edc355b2b14132a2a5b8f18912d0839b5eef8786ba5b1884d6814e",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    for relative, (url, expected) in FILES.items():
        path = args.root / relative
        if path.is_file() and sha256(path) == expected:
            print(f"verified {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        if sha256(path) != expected:
            path.unlink(missing_ok=True)
            raise ValueError(f"downloaded file failed SHA-256 verification: {path}")
        print(f"downloaded and verified {path}")


if __name__ == "__main__":
    main()
