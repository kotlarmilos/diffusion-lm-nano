"""Public adapters for byte data and the common gpt2-nano token-id format."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
import json
from pathlib import Path
import struct
import sys
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import Dataset


class Tokenizer(Protocol):
    vocab_size: int
    mask_token_id: int
    pad_token_id: int

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class ByteTokenizer:
    vocab_size = 258
    pad_token_id = 256
    mask_token_id = 257

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        values = bytes(token for token in token_ids if 0 <= token < 256)
        return values.decode("utf-8", errors="replace")


class GPT2NanoTokenizerAdapter:
    """Load the custom gpt2-nano BPE vocabulary and merge table."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        path = Path(tokenizer_path)
        with (path / "vocab.json").open(encoding="utf-8") as handle:
            vocab = json.load(handle)
        with (path / "merges.json").open(encoding="utf-8") as handle:
            merges = json.load(handle)
        if not isinstance(vocab, dict) or not all(
            isinstance(token, str) and isinstance(token_id, int)
            for token, token_id in vocab.items()
        ):
            raise ValueError("vocab.json must map token strings to integer IDs")
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("vocab.json IDs must be contiguous from zero")
        if not isinstance(merges, list) or not all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(piece, str) for piece in pair)
            for pair in merges
        ):
            raise ValueError("merges.json must contain string pairs")
        self._vocab: dict[str, int] = vocab
        self._inverse = {token_id: token for token, token_id in vocab.items()}
        self._merges: list[list[str]] = merges
        base_size = len(vocab)
        self.pad_token_id = base_size
        self.mask_token_id = base_size + 1
        self.vocab_size = base_size + 2

    def encode(self, text: str) -> list[int]:
        words = [list(word) + ["<|EOW|>"] for word in text.split(" ")]
        for first, second in self._merges:
            merged = first + second
            for word in words:
                index = 0
                while index < len(word) - 1:
                    if word[index] == first and word[index + 1] == second:
                        word[index : index + 2] = [merged]
                    else:
                        index += 1
        tokens = []
        for word in words:
            for token in word:
                try:
                    tokens.append(self._vocab[token])
                except KeyError as error:
                    raise ValueError(
                        f"token {token!r} is not in the gpt2-nano vocabulary"
                    ) from error
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        pieces = []
        for value in token_ids:
            if value in {self.pad_token_id, self.mask_token_id}:
                continue
            try:
                piece = self._inverse[value]
            except KeyError as error:
                raise ValueError(f"token id {value} is outside the vocabulary") from error
            pieces.append(" " if piece == "<|EOW|>" else piece)
        return "".join(pieces)


class BinaryTokenDataset(Dataset[Tensor]):
    """Read headered little-endian uint16 token blocks used by gpt2-nano."""

    def __init__(
        self,
        path: str | Path,
        sequence_length: int,
        token_limit: int | None = None,
    ) -> None:
        self.path = Path(path)
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if token_limit is not None and token_limit < 1:
            raise ValueError("token_limit must be positive")
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        header_bytes = 256 * 4
        with self.path.open("rb") as handle:
            header = handle.read(header_bytes)
        if len(header) != header_bytes:
            raise ValueError("token shard has a truncated header")
        metadata = struct.unpack("<256i", header)
        if metadata[0] != 20250320 or metadata[1] != 1:
            raise ValueError("token shard has an unsupported header")
        token_count = metadata[2]
        expected_bytes = header_bytes + token_count * 2
        if token_count < 1 or self.path.stat().st_size != expected_bytes:
            raise ValueError("token shard size does not match its token count")
        values = array("H")
        with self.path.open("rb") as handle:
            handle.seek(header_bytes)
            values.fromfile(handle, token_count)
        if sys.byteorder != "little":
            values.byteswap()
        if token_limit is not None and max(values) >= token_limit:
            raise ValueError("token shard contains an ID outside the base vocabulary")
        self._values = values
        self.sequence_length = sequence_length
        if len(values) < sequence_length:
            raise ValueError("token file is shorter than one sequence")

    def __len__(self) -> int:
        return len(self._values) // self.sequence_length

    def __getitem__(self, index: int) -> Tensor:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        return torch.tensor(
            self._values[start : start + self.sequence_length], dtype=torch.long
        )


class JsonlTokenDataset(Dataset[Tensor]):
    """Read JSONL records containing an ``input_ids`` integer list."""

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        records: list[Tensor] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                ids = record.get("input_ids")
                if not isinstance(ids, list) or not all(
                    isinstance(value, int) for value in ids
                ):
                    raise ValueError(f"invalid input_ids on line {line_number}")
                for start in range(0, len(ids) - sequence_length + 1, sequence_length):
                    records.append(
                        torch.tensor(ids[start : start + sequence_length])
                    )
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Tensor:
        return self._records[index]
