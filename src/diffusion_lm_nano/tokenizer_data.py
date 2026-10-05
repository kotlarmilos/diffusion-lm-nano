"""Reader for the headered gpt2-nano token shard."""

from __future__ import annotations

from array import array
from pathlib import Path
import struct
import sys

import torch
from torch import Tensor
from torch.utils.data import Dataset


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
