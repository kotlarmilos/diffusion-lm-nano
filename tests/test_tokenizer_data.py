import json
from pathlib import Path
import tempfile
import unittest

from diffusion_lm_nano.config import (
    DiffusionConfig,
    ModelConfig,
    ProjectConfig,
    TrainConfig,
)
from diffusion_lm_nano.tokenizer_data import GPT2NanoTokenizerAdapter
from diffusion_lm_nano.train import train


class GPT2NanoTokenizerTests(unittest.TestCase):
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

            result = train(
                config,
                root / "config.toml",
                root / "artifacts" / "checkpoint.pt",
                root / "artifacts" / "sample.json",
                manifest,
            )

            self.assertGreater(result["final_train_loss"], 0)


if __name__ == "__main__":
    unittest.main()
