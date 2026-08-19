import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from scripts.analyze_publication_baseline import analyze


class PublicationBaselineTests(unittest.TestCase):
    def test_analyze_compares_evaluation_with_holdout_majority_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.toml"
            config.write_text(
                """
[model]
vocab_size = 5
max_length = 4
mask_token_id = 4
pad_token_id = 3

[train]
validation_fraction = 0.25
""".strip()
                + "\n",
                encoding="utf-8",
            )
            tokens = [0] * 12 + [2, 2, 2, 1]
            header = [20250320, 1, len(tokens)] + [0] * 253
            corpus = root / "tokens.bin"
            corpus.write_bytes(
                struct.pack("<256i", *header)
                + struct.pack(f"<{len(tokens)}H", *tokens)
            )
            vocab = root / "vocab.json"
            vocab.write_text(
                json.dumps({"a": 0, "b": 1, "<|EOW|>": 2}),
                encoding="utf-8",
            )
            evaluation = root / "evaluation.json"
            evaluation.write_text(
                json.dumps({"masked_accuracy": 0.5}),
                encoding="utf-8",
            )
            sample = root / "sample.json"
            sample.write_text(
                json.dumps(
                    {
                        "token_ids": [2, 2, 2, 2],
                        "stopping_reason": "all_revealed",
                        "forward_passes": 4,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint = root / "checkpoint.pt"
            torch.save(
                {
                    "model_config": {"max_length": 4, "pad_token_id": 3},
                    "train_config": {"validation_fraction": 0.25},
                },
                checkpoint,
            )

            result = analyze(
                config,
                checkpoint,
                corpus,
                vocab,
                evaluation,
                sample,
            )

        holdout = result["holdout"]
        self.assertEqual(holdout["sequences"], 1)
        self.assertEqual(holdout["majority_token_id"], 2)
        self.assertEqual(holdout["majority_token_piece"], "<|EOW|>")
        self.assertEqual(holdout["majority_token_fraction"], 0.75)
        self.assertEqual(
            result["evaluation"]["accuracy_minus_holdout_majority_fraction"],
            -0.25,
        )
        self.assertTrue(result["unconditional_sample"]["all_majority_token"])
        self.assertEqual(result["unconditional_sample"]["unique_token_ids"], 1)


if __name__ == "__main__":
    unittest.main()
