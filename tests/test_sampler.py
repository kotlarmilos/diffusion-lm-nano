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


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SamplerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
