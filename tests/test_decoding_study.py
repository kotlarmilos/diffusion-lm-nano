import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


if torch is not None:
    class RightContextModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size = 8
            self.register_buffer("target", torch.tensor([1, 2, 3, 4]))
            self.register_buffer("wrong", torch.tensor([2, 3, 4, 5]))

        def forward(self, tokens, t):
            batch, length = tokens.shape
            logits = torch.full(
                (batch, length, self.vocab_size),
                -5.0,
                device=tokens.device,
            )
            batch_index = torch.arange(batch, device=tokens.device)
            for position in range(length):
                if position == length - 1:
                    can_predict = torch.ones(batch, dtype=torch.bool, device=tokens.device)
                else:
                    can_predict = tokens[:, position + 1].eq(self.target[position + 1])
                preferred = torch.where(
                    can_predict,
                    self.target[position].expand(batch),
                    self.wrong[position].expand(batch),
                )
                score = torch.where(
                    can_predict,
                    torch.full((batch,), 6.0, device=tokens.device),
                    torch.full((batch,), 2.0, device=tokens.device),
                )
                logits[batch_index, position, preferred] = score
            return logits

    class ConfidentReconstructionModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vocab_size = 8

        def forward(self, tokens, t):
            batch, length = tokens.shape
            logits = torch.full(
                (batch, length, self.vocab_size),
                -20.0,
                device=tokens.device,
            )
            positions = torch.arange(length, device=tokens.device)
            logits[:, positions, positions.remainder(self.vocab_size - 2)] = 20.0
            return logits


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DecodingStudyTests(unittest.TestCase):
    def test_parse_step_budgets_sorts_and_validates(self) -> None:
        from diffusion_lm_nano.decoding_study import parse_step_budgets

        self.assertEqual(parse_step_budgets("4,1 2"), (1, 2, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_step_budgets("0,2")

    def test_metric_counts_ignore_padding(self) -> None:
        from diffusion_lm_nano.decoding_study import accumulate_token_metrics

        predictions = torch.tensor([[1, 9, 3]])
        references = torch.tensor([[1, 0, 6]])
        counts = accumulate_token_metrics(predictions, references, pad_token_id=6)

        self.assertEqual(counts.correct, 1)
        self.assertEqual(counts.total, 2)
        self.assertEqual(counts.accuracy, 0.5)

    def test_confidence_ordering_can_beat_fixed_order_on_context_toy(self) -> None:
        from diffusion_lm_nano.config import DiffusionConfig, ModelConfig
        from diffusion_lm_nano.decoding_study import run_budget_sweep

        model_config = ModelConfig(
            vocab_size=8,
            max_length=4,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=4,
            pad_token_id=6,
            mask_token_id=7,
        )
        diffusion_config = DiffusionConfig(
            schedule="linear",
            sampling_steps=4,
            temperature=0.0,
        )
        rows = run_budget_sweep(
            RightContextModel(),
            [torch.tensor([[1, 2, 3, 4]])],
            model_config=model_config,
            diffusion_config=diffusion_config,
            step_budgets=(4,),
            ordering_strategies=("confidence", "fixed_order"),
            seed=3,
            device=torch.device("cpu"),
            initial_timestep=1.0,
        )
        by_strategy = {row["ordering_strategy"]: row for row in rows}

        self.assertEqual(by_strategy["confidence"]["correct_tokens"], 4)
        self.assertLess(
            by_strategy["fixed_order"]["token_accuracy"],
            by_strategy["confidence"]["token_accuracy"],
        )

    def test_matched_adaptive_study_reuses_examples_and_reports_realized_passes(
        self,
    ) -> None:
        from diffusion_lm_nano.adaptive_study import (
            PreparedExample,
            run_matched_evaluation,
        )
        from diffusion_lm_nano.config import (
            AdaptiveDecodeConfig,
            DiffusionConfig,
            ModelConfig,
        )
        from diffusion_lm_nano.schedules import LinearSchedule

        model_config = ModelConfig(
            vocab_size=8,
            max_length=4,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=4,
            pad_token_id=6,
            mask_token_id=7,
        )
        example = PreparedExample(
            dataset_index=91,
            reference=torch.tensor([[0, 1, 2, 3]]),
            corrupted=torch.full((1, 4), 7),
            selected=torch.ones((1, 4), dtype=torch.bool),
            corruption_seed=101,
            decoding_seed=202,
        )
        result = run_matched_evaluation(
            ConfidentReconstructionModel(),
            [example],
            policies=("fixed_order", "fixed_fraction", "confidence", "adaptive"),
            adaptive_config=AdaptiveDecodeConfig(
                min_passes=2,
                entropy_threshold=0.05,
            ),
            model_config=model_config,
            diffusion_config=DiffusionConfig(
                schedule="linear",
                sampling_steps=4,
                temperature=0.0,
            ),
            schedule=LinearSchedule(),
            max_passes=4,
            initial_timestep=1.0,
            device=torch.device("cpu"),
            precision="float32",
            warmup_runs=0,
        )
        paired = result["paired_examples"][0]
        self.assertEqual(paired["dataset_index"], 91)
        self.assertEqual(paired["initially_masked_tokens"], 4)
        self.assertEqual(set(paired["policies"]), {
            "fixed_order",
            "fixed_fraction",
            "confidence",
            "adaptive",
        })
        for policy in ("fixed_order", "fixed_fraction", "confidence"):
            self.assertEqual(
                paired["policies"][policy]["realized_forward_passes"],
                4,
            )
        self.assertEqual(
            paired["policies"]["adaptive"]["realized_forward_passes"],
            2,
        )
        self.assertEqual(
            paired["policies"]["adaptive"]["stopping_reason"],
            "entropy_threshold",
        )
        self.assertTrue(
            all(
                policy_result["complete_output"]
                for policy_result in paired["policies"].values()
            )
        )

    def test_fixed_policy_trajectory_skips_adaptive_diagnostics(self) -> None:
        from diffusion_lm_nano.adaptive_study import (
            PreparedExample,
            _evaluate_example,
        )
        from diffusion_lm_nano.config import DiffusionConfig, ModelConfig
        from diffusion_lm_nano.schedules import LinearSchedule

        model_config = ModelConfig(
            vocab_size=8,
            max_length=4,
            d_model=4,
            n_heads=1,
            n_layers=1,
            d_ff=4,
            pad_token_id=6,
            mask_token_id=7,
        )
        example = PreparedExample(
            dataset_index=1,
            reference=torch.tensor([[0, 1, 2, 3]]),
            corrupted=torch.full((1, 4), 7),
            selected=torch.ones((1, 4), dtype=torch.bool),
            corruption_seed=2,
            decoding_seed=3,
        )
        with patch(
            "diffusion_lm_nano.sampler.normalized_entropy",
            side_effect=AssertionError("fixed policy computed adaptive diagnostics"),
        ):
            record = _evaluate_example(
                ConfidentReconstructionModel(),
                example,
                policy="fixed_order",
                adaptive_config=None,
                model_config=model_config,
                diffusion_config=DiffusionConfig(
                    schedule="linear",
                    sampling_steps=4,
                    temperature=0.0,
                ),
                schedule=LinearSchedule(),
                max_passes=4,
                initial_timestep=1.0,
                device=torch.device("cpu"),
                precision="float32",
                include_trajectory=True,
            )

        self.assertEqual(len(record["trajectory"]), 4)
        self.assertTrue(
            all(step["normalized_entropy"] is None for step in record["trajectory"])
        )

    def test_calibration_selection_prefers_accuracy_before_pass_count(self) -> None:
        from diffusion_lm_nano.adaptive_study import select_adaptive_candidate

        faster = {
            "initially_masked_token_accuracy": 0.8,
            "exact_sequence_recovery_rate": 0.4,
            "mean_realized_forward_passes": 2.0,
            "adaptive_config": {
                "min_passes": 2,
                "entropy_threshold": 0.3,
                "confidence_delta": 0.0,
                "confidence_patience": 0,
            },
        }
        more_accurate = {
            "initially_masked_token_accuracy": 0.9,
            "exact_sequence_recovery_rate": 0.5,
            "mean_realized_forward_passes": 4.0,
            "adaptive_config": {
                "min_passes": 4,
                "entropy_threshold": 0.1,
                "confidence_delta": 0.0,
                "confidence_patience": 0,
            },
        }
        self.assertIs(
            select_adaptive_candidate([faster, more_accurate]),
            more_accurate,
        )

    def test_publication_expectations_bind_every_input_hash(self) -> None:
        from diffusion_lm_nano.adaptive_study import (
            AdaptiveStudyConfig,
            _verify_manifest_expectations,
        )
        from diffusion_lm_nano.artifacts import sha256

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "path": "artifacts/publication-checkpoint.pt",
                                "status": "measured",
                                "sha256": sha256(checkpoint),
                            },
                            {
                                "path": "artifacts/adaptive.json",
                                "status": "pending",
                                "provenance": {
                                    "immutable_input_expectations": {
                                        "checkpoint": {
                                            "path": "artifacts/publication-checkpoint.pt",
                                            "sha256": "resolved from measured entry",
                                            "status": "measured",
                                        },
                                        "corpus": {
                                            "path": "data/train.bin",
                                            "sha256": sha256(corpus),
                                        },
                                        "training_config": {
                                            "path": "configs/publication.toml",
                                            "sha256": sha256(training_config),
                                        },
                                        "study_config": {
                                            "path": "configs/adaptive.toml",
                                            "sha256": sha256(study_config),
                                        },
                                    }
                                },
                            },
                        ]
                    }
                ),
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
                policies=(
                    "fixed_order",
                    "fixed_fraction",
                    "confidence",
                    "adaptive",
                ),
                adaptive_min_passes=(2,),
                adaptive_entropy_thresholds=(0.0,),
                adaptive_confidence_deltas=(0.0,),
                adaptive_confidence_patiences=(0,),
            )
            verified = _verify_manifest_expectations(config, study_config)
            self.assertEqual(
                verified["checkpoint"]["observed_sha256"],
                sha256(checkpoint),
            )
            self.assertEqual(
                verified["checkpoint"]["sha256"],
                sha256(checkpoint),
            )
            self.assertEqual(
                _verify_manifest_expectations(config, study_config),
                verified,
            )
            corpus.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "corpus hash"):
                _verify_manifest_expectations(config, study_config)


if __name__ == "__main__":
    unittest.main()
