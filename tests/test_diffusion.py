from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DiffusionTests(unittest.TestCase):
    def test_schedules_are_monotonic_with_exact_endpoints(self) -> None:
        from diffusion_lm_nano.schedules import make_schedule

        times = torch.linspace(0, 1, 101)
        for name in ("linear", "cosine", "power"):
            values = make_schedule(name).probability(times)
            self.assertEqual(float(values[0]), 0.0)
            self.assertAlmostEqual(float(values[-1]), 1.0, places=6)
            self.assertTrue(bool((values[1:] >= values[:-1]).all()))

    def test_corruption_endpoints(self) -> None:
        from diffusion_lm_nano.corruption import IGNORE_INDEX, corrupt_tokens
        from diffusion_lm_nano.schedules import LinearSchedule

        clean = torch.tensor([[1, 2, 3], [4, 5, 6]])
        uncorrupted, targets, selected = corrupt_tokens(
            clean, torch.zeros(2), LinearSchedule(), 9
        )
        self.assertTrue(torch.equal(uncorrupted, clean))
        self.assertFalse(bool(selected.any()))
        self.assertTrue(bool(targets.eq(IGNORE_INDEX).all()))

        corrupted, targets, selected = corrupt_tokens(
            clean, torch.ones(2), LinearSchedule(), 9
        )
        self.assertTrue(bool(corrupted.eq(9).all()))
        self.assertTrue(torch.equal(targets, clean))
        self.assertTrue(bool(selected.all()))

    def test_padding_is_never_corrupted(self) -> None:
        from diffusion_lm_nano.corruption import corrupt_tokens
        from diffusion_lm_nano.schedules import LinearSchedule

        clean = torch.tensor([[1, 2, 0]])
        attention = torch.tensor([[True, True, False]])
        corrupted, _, selected = corrupt_tokens(
            clean, torch.ones(1), LinearSchedule(), 9, attention
        )
        self.assertEqual(int(corrupted[0, 2]), 0)
        self.assertFalse(bool(selected[0, 2]))

    def test_no_loss_on_uncorrupted_tokens(self) -> None:
        from diffusion_lm_nano.corruption import IGNORE_INDEX
        from diffusion_lm_nano.objective import masked_cross_entropy

        logits = torch.randn(2, 3, 10, requires_grad=True)
        targets = torch.full((2, 3), IGNORE_INDEX)
        loss = masked_cross_entropy(logits, targets)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(bool(logits.grad.eq(0).all()))


if __name__ == "__main__":
    unittest.main()

