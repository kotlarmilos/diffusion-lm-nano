import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from diffusion_lm_nano.config import (
    DiffusionConfig,
    ModelConfig,
    ProjectConfig,
    TrainConfig,
)
from diffusion_lm_nano.artifacts import sha256
from diffusion_lm_nano.tokenizer_data import (
    BinaryTokenDataset,
    ByteTokenizer,
    GPT2NanoTokenizerAdapter,
)
from diffusion_lm_nano.train import _load_corpus, train


class GPT2NanoTokenizerTests(unittest.TestCase):
    def test_binary_dataset_reads_headered_gpt2_nano_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            header = [20250320, 1, 6] + [0] * 253
            path.write_bytes(
                struct.pack("<256i", *header)
                + struct.pack("<6H", 1, 2, 3, 4, 5, 6)
            )
            dataset = BinaryTokenDataset(path, 3, token_limit=7)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0].tolist(), [1, 2, 3])
            self.assertEqual(dataset[1].tolist(), [4, 5, 6])

    def test_binary_dataset_rejects_invalid_header_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            for magic, version in ((0, 1), (20250320, 2)):
                with self.subTest(magic=magic, version=version):
                    invalid_header = [magic, version, 2] + [0] * 253
                    path.write_bytes(
                        struct.pack("<256i", *invalid_header)
                        + struct.pack("<2H", 1, 2)
                    )
                    with self.assertRaisesRegex(ValueError, "unsupported header"):
                        BinaryTokenDataset(path, 2)

            invalid_count = [20250320, 1, 3] + [0] * 253
            path.write_bytes(
                struct.pack("<256i", *invalid_count) + struct.pack("<2H", 1, 2)
            )
            with self.assertRaisesRegex(ValueError, "token count"):
                BinaryTokenDataset(path, 2)

    def test_binary_dataset_rejects_token_outside_base_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            valid_header = [20250320, 1, 2] + [0] * 253
            path.write_bytes(
                struct.pack("<256i", *valid_header) + struct.pack("<2H", 1, 7)
            )
            with self.assertRaisesRegex(ValueError, "base vocabulary"):
                BinaryTokenDataset(path, 2, token_limit=7)

    def test_training_loader_applies_model_token_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.bin"
            header = [20250320, 1, 2] + [0] * 253
            path.write_bytes(
                struct.pack("<256i", *header) + struct.pack("<2H", 1, 256)
            )
            config = ProjectConfig(
                model=ModelConfig(),
                diffusion=DiffusionConfig(),
                train=TrainConfig(corpus_path=str(path)),
            )

            with self.assertRaisesRegex(ValueError, "base vocabulary"):
                _load_corpus(config, ByteTokenizer())

    def test_directory_hash_binds_names_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "vocab.json").write_text('{"x": 0}', encoding="utf-8")
            (path / "merges.json").write_text("[]", encoding="utf-8")
            first = sha256(path)
            self.assertEqual(sha256(path), first)
            (path / "merges.json").write_text('[["x", "x"]]', encoding="utf-8")
            self.assertNotEqual(sha256(path), first)

    def test_custom_bpe_adapter_round_trips_and_reserves_special_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "vocab.json").write_text(
                json.dumps({"h": 0, "i": 1, "<|EOW|>": 2, "hi": 3})
            )
            (path / "merges.json").write_text(json.dumps([["h", "i"]]))
            tokenizer = GPT2NanoTokenizerAdapter(path)

        self.assertEqual(tokenizer.encode("hi"), [3, 2])
        self.assertEqual(tokenizer.decode([3, 2]), "hi ")
        self.assertEqual(tokenizer.pad_token_id, 4)
        self.assertEqual(tokenizer.mask_token_id, 5)
        self.assertEqual(tokenizer.vocab_size, 6)

    def test_noncontiguous_vocabulary_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "vocab.json").write_text(json.dumps({"x": 1}))
            (path / "merges.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "contiguous"):
                GPT2NanoTokenizerAdapter(path)

    def test_jsonl_token_dataset_runs_through_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "tokens.jsonl"
            corpus.write_text(
                json.dumps({"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}) + "\n",
                encoding="utf-8",
            )
            manifest = root / "artifacts" / "manifest.json"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps({"schema_version": 1, "artifacts": []}),
                encoding="utf-8",
            )
            config = ProjectConfig(
                model=ModelConfig(
                    vocab_size=258,
                    max_length=4,
                    d_model=8,
                    n_heads=2,
                    n_layers=1,
                    d_ff=16,
                    mask_token_id=257,
                    pad_token_id=256,
                ),
                diffusion=DiffusionConfig(
                    schedule="linear",
                    sampling_steps=2,
                    temperature=0.0,
                ),
                train=TrainConfig(
                    steps=1,
                    batch_size=1,
                    log_every=1,
                    device="cpu",
                    corpus_path=str(corpus),
                ),
            )
            config_path = root / "config.toml"
            config_path.write_text("# test configuration\n", encoding="utf-8")

            checkpoint_path = root / "artifacts" / "checkpoint.pt"
            result = train(
                config,
                config_path,
                checkpoint_path,
                root / "artifacts" / "sample.json",
                manifest,
            )

            self.assertGreater(result["final_train_loss"], 0)
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            self.assertIsInstance(checkpoint["source"]["git_revision"], str)
            self.assertEqual(
                checkpoint["source"]["git_state_captured"],
                "training_start",
            )


if __name__ == "__main__":
    unittest.main()
