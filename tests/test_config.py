from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from diffusion_lm_nano.config import DiffusionConfig, ModelConfig, ProjectConfig


class ConfigTests(unittest.TestCase):
    def test_smoke_config_is_valid(self) -> None:
        config = ProjectConfig.from_toml(
            Path(__file__).parents[1] / "configs" / "smoke.toml"
        )
        self.assertEqual(config.model.d_model, 32)
        self.assertEqual(config.diffusion.strategy, "confidence")

    def test_decoding_study_config_is_valid(self) -> None:
        config = ProjectConfig.from_toml(
            Path(__file__).parents[1] / "configs" / "decoding_study.toml"
        )
        self.assertEqual(config.diffusion.sampling_steps, 12)

    def test_decoding_signal_config_is_valid(self) -> None:
        config = ProjectConfig.from_toml(
            Path(__file__).parents[1] / "configs" / "decoding_signal.toml"
        )
        self.assertEqual(config.train.corpus_path, "artifacts/decoding-copy-corpus.jsonl")

    def test_invalid_head_divisibility_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(d_model=30, n_heads=8).validate()

    def test_invalid_strategy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strategy"):
            DiffusionConfig(strategy="unknown").validate()  # type: ignore[arg-type]

    def test_special_tokens_must_differ(self) -> None:
        with self.assertRaisesRegex(ValueError, "differ"):
            ModelConfig(mask_token_id=256, pad_token_id=256).validate()


if __name__ == "__main__":
    unittest.main()
