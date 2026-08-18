from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


if torch is not None:
    class ToyModel(torch.nn.Module):
        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, tokens, t):
            batch, length = tokens.shape
            positions = torch.arange(length)[None, :, None]
            vocabulary = torch.arange(self.vocab_size)[None, None, :]
            preferred = (positions % (self.vocab_size - 2)).expand(batch, -1, -1)
            return -(vocabulary - preferred).abs().float()

    class StaticLogitModel(torch.nn.Module):
        def __init__(self, vocab_size: int, peak: float = 20.0) -> None:
            super().__init__()
            self.vocab_size = vocab_size
            self.peak = peak

        def forward(self, tokens, t):
            batch, length = tokens.shape
            logits = torch.full(
                (batch, length, self.vocab_size),
                -self.peak,
                device=tokens.device,
            )
            positions = torch.arange(length, device=tokens.device)
            preferred = positions.remainder(self.vocab_size - 2)
            logits[:, positions, preferred] = self.peak
            return logits


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SamplerTests(unittest.TestCase):
    def test_fixed_strategy_ignores_inapplicable_adaptive_minimum(self) -> None:
        from diffusion_lm_nano.config import AdaptiveDecodeConfig
        from diffusion_lm_nano.sampler import sample_with_metadata

        model = StaticLogitModel(vocab_size=8)
        result = sample_with_metadata(
            model,
            batch_size=1,
            length=4,
            mask_token_id=7,
            pad_token_id=6,
            vocab_size=8,
            steps=2,
            strategy="confidence",
            temperature=0.0,
            device=torch.device("cpu"),
            adaptive_config=AdaptiveDecodeConfig(min_passes=4),
        )
        self.assertEqual(result.forward_passes, 2)

    def _sample(self, seed: int, strategy: str = "confidence"):
        from diffusion_lm_nano.sampler import sample

        return sample(
            ToyModel(12),
            batch_size=2,
            length=9,
            mask_token_id=11,
            pad_token_id=10,
            vocab_size=12,
            steps=4,
            strategy=strategy,
            temperature=0.7,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_sampler_terminates_for_both_strategies(self) -> None:
        for strategy in ("fixed_fraction", "confidence", "fixed_order"):
            result = self._sample(3, strategy)
            self.assertEqual(tuple(result.shape), (2, 9))
            self.assertFalse(bool(result.eq(11).any()))

    def test_sampling_is_deterministic_for_a_seed(self) -> None:
        self.assertTrue(torch.equal(self._sample(42), self._sample(42)))

    def test_callback_observes_final_unmasked_state(self) -> None:
        from diffusion_lm_nano.sampler import sample

        states = []
        sample(
            ToyModel(8),
            batch_size=1,
            length=5,
            mask_token_id=7,
            pad_token_id=6,
            vocab_size=8,
            steps=3,
            temperature=0,
            callback=lambda step, tokens, confidence: states.append(tokens),
        )
        self.assertTrue(states)
        self.assertFalse(bool(states[-1].eq(7).any()))

    def test_normalized_entropy_has_expected_endpoints(self) -> None:
        from diffusion_lm_nano.sampler import normalized_entropy

        uniform = torch.full((1, 4), 0.25)
        certain = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(normalized_entropy(uniform, 4)), 1.0, places=6)
        self.assertEqual(float(normalized_entropy(certain, 4)), 0.0)

    def test_entropy_stopping_commits_all_remaining_predictions(self) -> None:
        from diffusion_lm_nano.config import AdaptiveDecodeConfig
        from diffusion_lm_nano.sampler import sample_with_metadata

        result = sample_with_metadata(
            StaticLogitModel(10),
            batch_size=1,
            length=12,
            mask_token_id=9,
            pad_token_id=8,
            vocab_size=10,
            steps=6,
            strategy="adaptive",
            temperature=0.0,
            adaptive_config=AdaptiveDecodeConfig(
                min_passes=2,
                entropy_threshold=0.05,
            ),
        )

        self.assertEqual(result.forward_passes, 2)
        self.assertEqual(result.stopping_reason, "entropy_threshold")
        self.assertTrue(result.trajectory[-1].adaptive_finalization)
        self.assertEqual(result.trajectory[-1].remaining_count, 0)
        self.assertFalse(bool(result.tokens.eq(9).any()))

    def test_confidence_stall_stops_after_configured_patience(self) -> None:
        from diffusion_lm_nano.config import AdaptiveDecodeConfig
        from diffusion_lm_nano.sampler import sample_with_metadata

        result = sample_with_metadata(
            StaticLogitModel(10),
            batch_size=1,
            length=12,
            mask_token_id=9,
            pad_token_id=8,
            vocab_size=10,
            steps=6,
            strategy="adaptive",
            temperature=0.0,
            adaptive_config=AdaptiveDecodeConfig(
                min_passes=1,
                confidence_delta=0.01,
                confidence_patience=2,
            ),
        )

        self.assertEqual(result.forward_passes, 3)
        self.assertEqual(result.stopping_reason, "confidence_stall")
        self.assertAlmostEqual(
            result.trajectory[-1].confidence_improvement or 0.0,
            0.0,
            places=6,
        )

    def test_adaptive_maximum_pass_fallback_is_complete(self) -> None:
        from diffusion_lm_nano.config import AdaptiveDecodeConfig
        from diffusion_lm_nano.sampler import sample_with_metadata

        result = sample_with_metadata(
            ToyModel(12),
            batch_size=2,
            length=9,
            mask_token_id=11,
            pad_token_id=10,
            vocab_size=12,
            steps=4,
            strategy="adaptive",
            temperature=0.0,
            adaptive_config=AdaptiveDecodeConfig(),
        )

        self.assertEqual(result.forward_passes, 4)
        self.assertEqual(result.stopping_reason, "maximum_passes")
        self.assertFalse(bool(result.tokens.eq(11).any()))

    def test_adaptive_trajectory_accounts_for_every_reveal(self) -> None:
        from diffusion_lm_nano.sampler import sample_with_metadata

        result = sample_with_metadata(
            ToyModel(12),
            batch_size=1,
            length=9,
            mask_token_id=11,
            pad_token_id=10,
            vocab_size=12,
            steps=4,
            strategy="adaptive",
            temperature=0.0,
        )

        self.assertEqual(len(result.trajectory), result.forward_passes)
        self.assertEqual(sum(step.reveal_count for step in result.trajectory), 9)
        previous_remaining = 9
        for step in result.trajectory:
            self.assertEqual(step.unresolved_count, previous_remaining)
            self.assertEqual(
                step.remaining_count,
                step.unresolved_count - step.reveal_count,
            )
            previous_remaining = step.remaining_count
        self.assertEqual(previous_remaining, 0)

    def test_adaptive_stopping_is_seed_deterministic_at_both_temperatures(self) -> None:
        from diffusion_lm_nano.config import AdaptiveDecodeConfig
        from diffusion_lm_nano.sampler import sample_with_metadata

        for temperature in (0.0, 0.7):
            def run():
                return sample_with_metadata(
                    StaticLogitModel(10),
                    batch_size=1,
                    length=10,
                    mask_token_id=9,
                    pad_token_id=8,
                    vocab_size=10,
                    steps=5,
                    strategy="adaptive",
                    temperature=temperature,
                    generator=torch.Generator().manual_seed(123),
                    adaptive_config=AdaptiveDecodeConfig(
                        min_passes=2,
                        entropy_threshold=0.05,
                    ),
                )

            first = run()
            second = run()
            self.assertTrue(torch.equal(first.tokens, second.tokens))
            self.assertEqual(first.stopping_reason, second.stopping_reason)
            self.assertEqual(first.trajectory, second.trajectory)


if __name__ == "__main__":
    unittest.main()
