from pathlib import Path
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
