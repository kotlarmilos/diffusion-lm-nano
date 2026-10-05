from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import shutil
import struct
import unittest

import torch
import torch.nn.functional as F

from diffusion_lm_nano.adaptive_study import (
    AdaptiveStudyConfig,
    _verify_manifest_expectations,
    select_adaptive_candidate,
    split_holdout_windows,
)
from diffusion_lm_nano.artifacts import sha256
from diffusion_lm_nano.config import (
    AdaptiveDecodeConfig,
    DiffusionConfig,
    ModelConfig,
)
from diffusion_lm_nano.corruption import IGNORE_INDEX, corrupt_tokens
from diffusion_lm_nano.evaluate import main as evaluate_main
from diffusion_lm_nano.model import MaskedDiffusionTransformer
from diffusion_lm_nano.objective import masked_cross_entropy
from diffusion_lm_nano.sampler import sample
from diffusion_lm_nano.schedules import LinearSchedule
from diffusion_lm_nano.tokenizer_data import BinaryTokenDataset
from scripts.analyze_publication_baseline import analyze_majority_control


ROOT = Path(__file__).parents[1]


class PositionModel(torch.nn.Module):
    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, length = tokens.shape
        logits = torch.full((batch, length, 8), -20.0)
        positions = torch.arange(length)
        logits[:, positions, positions.remainder(6)] = 20.0
        return logits


class ReleaseTests(unittest.TestCase):
    def workspace(self, name: str) -> Path:
        path = ROOT / "tests" / ".work" / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, path, True)
        return path

    def test_corruption_and_loss_only_use_selected_tokens(self) -> None:
        clean = torch.tensor([[1, 2, 0]])
        attention = torch.tensor([[True, True, False]])
        corrupted, targets, selected = corrupt_tokens(
            clean,
            torch.ones(1),
            LinearSchedule(),
            mask_token_id=4,
            attention_mask=attention,
        )
        self.assertEqual(corrupted.tolist(), [[4, 4, 0]])
        self.assertEqual(targets.tolist(), [[1, 2, IGNORE_INDEX]])
        self.assertEqual(selected.tolist(), [[True, True, False]])

        logits = torch.zeros((1, 3, 5), requires_grad=True)
        expected = F.cross_entropy(logits[0, :2], clean[0, :2])
        loss = masked_cross_entropy(logits, targets)
        self.assertEqual(float(loss.detach()), float(expected.detach()))
        loss.backward()
        self.assertTrue(bool(logits.grad[0, 2].eq(0).all()))

    def test_stopping_rules_preserve_completion_and_selection_order(self) -> None:
        disabled = sample(
            PositionModel(),
            batch_size=1,
            length=6,
            mask_token_id=7,
            pad_token_id=6,
            vocab_size=8,
            max_passes=4,
            strategy="adaptive",
            temperature=0.0,
            adaptive_config=AdaptiveDecodeConfig(min_passes=2),
        )
        self.assertEqual(disabled.forward_passes, 4)
        self.assertEqual(disabled.stopping_reason, "maximum_passes")
        self.assertFalse(bool(disabled.tokens.eq(7).any()))

        enabled = sample(
            PositionModel(),
            batch_size=1,
            length=6,
            mask_token_id=7,
            pad_token_id=6,
            vocab_size=8,
            max_passes=4,
            strategy="adaptive",
            temperature=0.0,
            adaptive_config=AdaptiveDecodeConfig(
                min_passes=2,
                entropy_threshold=0.05,
            ),
        )
        self.assertEqual(enabled.forward_passes, 2)
        self.assertEqual(enabled.stopping_reason, "entropy_threshold")
        self.assertFalse(bool(enabled.tokens.eq(7).any()))
        self.assertTrue(enabled.trajectory[-1].adaptive_finalization)

        selected = select_adaptive_candidate([
            {
                "initially_masked_token_accuracy": 0.8,
                "exact_sequence_recovery_rate": 1.0,
                "mean_realized_forward_passes": 2.0,
                "adaptive_config": {
                    "min_passes": 2,
                    "entropy_threshold": 0.1,
                    "confidence_delta": 0.0,
                    "confidence_patience": 0,
                },
            },
            {
                "initially_masked_token_accuracy": 0.9,
                "exact_sequence_recovery_rate": 0.0,
                "mean_realized_forward_passes": 4.0,
                "adaptive_config": {
                    "min_passes": 4,
                    "entropy_threshold": 0.0,
                    "confidence_delta": 0.0,
                    "confidence_patience": 0,
                },
            },
        ])
        self.assertEqual(selected["initially_masked_token_accuracy"], 0.9)

    def test_holdout_windows_are_disjoint_and_bounded(self) -> None:
        calibration, heldout = split_holdout_windows(20, 0.25, 2, 3)
        self.assertEqual(calibration, (15, 16))
        self.assertEqual(heldout, (17, 18, 19))
        self.assertFalse(set(calibration) & set(heldout))
        with self.assertRaisesRegex(ValueError, "too small"):
            split_holdout_windows(20, 0.25, 3, 3)

    def test_binary_dataset_checks_header_size_and_vocabulary(self) -> None:
        root = self.workspace("dataset")
        path = root / "tokens.bin"
        header = [20250320, 1, 6] + [0] * 253
        path.write_bytes(
            struct.pack("<256i", *header)
            + struct.pack("<6H", 1, 2, 3, 4, 5, 6)
        )
        dataset = BinaryTokenDataset(path, 3, token_limit=7)
        self.assertEqual(dataset[1].tolist(), [4, 5, 6])

        path.write_bytes(path.read_bytes() + b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "size"):
            BinaryTokenDataset(path, 3)

        bad = root / "bad.bin"
        bad.write_bytes(
            struct.pack("<256i", *header)
            + struct.pack("<6H", 1, 2, 3, 4, 5, 7)
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            BinaryTokenDataset(bad, 3, token_limit=7)

    def test_release_hashes_and_measured_claims(self) -> None:
        expected_hashes = {
            "artifacts/adaptive-decoding-publication.json":
                "a0ba4b1013b8706b4afab7e7264f79d405ef624b814c2650138bdd249e021227",
            "artifacts/publication-benchmark.json":
                "54b029c612ef991641426a2b2e6fbbce9c27d577b4bb91084397de2536386e40",
            "artifacts/publication-evaluation.json":
                "a640af60fef7b71246110fd38dbccaa5b452391ae1ab19dc29f7b988225a6031",
            "artifacts/publication-majority-control.json":
                "d2c47dd2a50298d04555f870c9edd31f71af37f791a637eb009487c0b34fea5b",
            "artifacts/publication-sample.json":
                "0626020efbd69504a045a1e38fe75ef7f3c6966608312215f1a0e5a80f42efe4",
            "configs/adaptive_study_publication.toml":
                "352ba4a151ea77764f14ad8062b66776ea844738e4d37a7d0ec730bcdcf52028",
            "configs/publication.toml":
                "9cd8db87701526814ec650500e1e9d929172488bd5ed5d6c7067eb68a64f20ce",
        }
        for relative, expected in expected_hashes.items():
            self.assertEqual(sha256(ROOT / relative), expected)

        evaluation = json.loads(
            (ROOT / "artifacts/publication-evaluation.json").read_text()
        )
        self.assertAlmostEqual(evaluation["masked_accuracy"], 0.36029, places=5)
        self.assertAlmostEqual(
            evaluation["masked_cross_entropy"],
            5.14059,
            places=5,
        )
        control = json.loads(
            (ROOT / "artifacts/publication-majority-control.json").read_text()
        )
        self.assertAlmostEqual(
            control["holdout"]["majority_token_fraction"],
            0.36205,
            places=5,
        )
        self.assertTrue(control["unconditional_sample"]["all_majority_token"])

        adaptive = json.loads(
            (ROOT / "artifacts/adaptive-decoding-publication.json").read_text()
        )
        self.assertEqual(adaptive["calibration"]["selected_config"], {
            "confidence_delta": 0.0,
            "confidence_patience": 0,
            "entropy_threshold": 0.0,
            "min_passes": 4,
        })
        for row in adaptive["heldout"]["policy_results"]:
            self.assertEqual(row["mean_realized_forward_passes"], 32.0)
            self.assertEqual(row["stopping_reasons"], {"maximum_passes": 128})

    def test_manifest_expectations_reject_changed_inputs(self) -> None:
        root = self.workspace("manifest")
        artifacts = root / "artifacts"
        configs = root / "configs"
        data = root / "data"
        artifacts.mkdir()
        configs.mkdir()
        data.mkdir()
        checkpoint = artifacts / "publication-checkpoint.pt"
        corpus = data / "train.bin"
        training_config = configs / "publication.toml"
        study_config = configs / "adaptive.toml"
        output = artifacts / "adaptive.json"
        for path, content in (
            (checkpoint, b"checkpoint"),
            (corpus, b"corpus"),
            (training_config, b"training"),
            (study_config, b"study"),
        ):
            path.write_bytes(content)
        manifest = artifacts / "manifest.json"
        manifest.write_text(
            json.dumps({
                "artifacts": [{
                    "path": "artifacts/adaptive.json",
                    "provenance": {
                        "immutable_input_expectations": {
                            name: {
                                "path": path.relative_to(root).as_posix(),
                                "sha256": sha256(path),
                            }
                            for name, path in {
                                "checkpoint": checkpoint,
                                "corpus": corpus,
                                "training_config": training_config,
                                "study_config": study_config,
                            }.items()
                        }
                    },
                }]
            }),
            encoding="utf-8",
        )
        config = AdaptiveStudyConfig(
            checkpoint_path=checkpoint,
            training_config_path=training_config,
            corpus_path=corpus,
            output_path=output,
            manifest_path=manifest,
            device="cpu",
            precision="float32",
            seed=1,
            initial_timestep=0.75,
            max_passes=4,
            validation_fraction=0.1,
            calibration_sequences=1,
            heldout_sequences=1,
            warmup_runs=0,
            policies=("fixed_order", "fixed_fraction", "confidence", "adaptive"),
            adaptive_min_passes=(2,),
            adaptive_entropy_thresholds=(0.0,),
            adaptive_confidence_deltas=(0.0,),
            adaptive_confidence_patiences=(0,),
        )
        _verify_manifest_expectations(config, study_config)
        corpus.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "corpus hash"):
            _verify_manifest_expectations(config, study_config)

    def test_majority_control_is_bound_to_checkpoint_inputs(self) -> None:
        root = self.workspace("control")
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
            json.dumps({"evaluated_sequences": 1, "masked_accuracy": 0.5}),
            encoding="utf-8",
        )
        sample_path = root / "sample.json"
        sample_path.write_text(
            json.dumps({
                "token_ids": [2, 2, 2, 2],
                "stopping_reason": "all_revealed",
                "forward_passes": 4,
            }),
            encoding="utf-8",
        )
        checkpoint = root / "checkpoint.pt"
        torch.save(
            {
                "model_config": {"max_length": 4, "pad_token_id": 3},
                "train_config": {"validation_fraction": 0.25},
                "source": {
                    "git_dirty": False,
                    "git_state_captured": "training_start",
                },
                "training_inputs": {
                    "training_config_sha256": sha256(config),
                    "corpus_sha256": sha256(corpus),
                },
            },
            checkpoint,
        )

        result = analyze_majority_control(
            config,
            checkpoint,
            corpus,
            vocab,
            evaluation,
            sample_path,
        )
        self.assertEqual(result["holdout"]["majority_token_fraction"], 0.75)
        self.assertTrue(result["unconditional_sample"]["all_majority_token"])
        corpus.write_bytes(corpus.read_bytes() + b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "corpus hash"):
            analyze_majority_control(
                config,
                checkpoint,
                corpus,
                vocab,
                evaluation,
                sample_path,
            )

    def test_cpu_evaluation_smoke_path(self) -> None:
        root = self.workspace("evaluation")
        model_config = ModelConfig(
            vocab_size=8,
            max_length=4,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=8,
            dropout=0.0,
            mask_token_id=7,
            pad_token_id=6,
        )
        diffusion_config = DiffusionConfig(
            schedule="linear",
            sampling_steps=2,
            temperature=0.0,
        )
        model = MaskedDiffusionTransformer(model_config)
        checkpoint = root / "checkpoint.pt"
        torch.save(
            {
                "model_config": asdict(model_config),
                "diffusion_config": asdict(diffusion_config),
                "train_config": {"validation_fraction": 0.5},
                "model_state": model.state_dict(),
            },
            checkpoint,
        )
        tokens = [0, 1, 2, 3, 1, 2, 3, 4]
        header = [20250320, 1, len(tokens)] + [0] * 253
        corpus = root / "tokens.bin"
        corpus.write_bytes(
            struct.pack("<256i", *header)
            + struct.pack(f"<{len(tokens)}H", *tokens)
        )
        output = root / "evaluation.json"
        with redirect_stdout(io.StringIO()):
            evaluate_main([
                "--checkpoint",
                str(checkpoint),
                "--corpus-path",
                str(corpus),
                "--timestep",
                "1",
                "--device",
                "cpu",
                "--output",
                str(output),
            ])
        result = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["evaluated_sequences"], 1)
        self.assertEqual(result["masked_tokens"], 4)
        self.assertIsNotNone(result["masked_cross_entropy"])


if __name__ == "__main__":
    unittest.main()
