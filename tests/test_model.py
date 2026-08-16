from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        from diffusion_lm_nano.config import ModelConfig
        from diffusion_lm_nano.model import MaskedDiffusionTransformer

        config = ModelConfig(
            vocab_size=18,
            max_length=7,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            mask_token_id=17,
            pad_token_id=16,
        )
        model = MaskedDiffusionTransformer(config)
        output = model(torch.randint(0, 16, (3, 7)), torch.rand(3))
        self.assertEqual(tuple(output.shape), (3, 7, 18))

    def test_timestep_shape_is_validated(self) -> None:
        from diffusion_lm_nano.config import ModelConfig
        from diffusion_lm_nano.model import MaskedDiffusionTransformer

        model = MaskedDiffusionTransformer(
            ModelConfig(
                vocab_size=10,
                max_length=4,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                mask_token_id=9,
                pad_token_id=8,
            )
        )
        with self.assertRaisesRegex(ValueError, "t must"):
            model(torch.ones((2, 4), dtype=torch.long), torch.ones(3))


if __name__ == "__main__":
    unittest.main()

