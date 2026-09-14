"""D1-only model wiring checks without loading the CUDA model dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

SOURCE = Path(__file__).parents[2] / 'cosmos_framework/model/generator/umift_depth_model.py'

class DepthModelWiringTest(unittest.TestCase):
    def _make_positive_weight_model(self, compute_depth_aux_loss):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

        class Parent:
            def __init__(self, config):
                self.config = config
                self.parallel_dims = None
                self.tokenizer_vision_gen = SimpleNamespace(decode_with_grad=lambda latent: latent)

            def training_step(self, batch, iteration):
                latent = torch.zeros(1, 1, 6, 1, 1)
                output = {
                    'x0': [latent], 'xt': [latent], 'model_pred': [latent],
                    'sigma': torch.ones(1, 6),
                    'condition_mask_vision': [torch.tensor([1, 0, 0, 0, 0, 0])[:, None, None]],
                }
                return output, torch.tensor(2.0)

        scope = dict(OmniMoTModel=Parent, torch=torch, compute_depth_aux_loss=compute_depth_aux_loss)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), scope)
        return scope['UMIFTDepthModel'](None, depth_aux_weight=0.1)

    @staticmethod
    def _valid_batch():
        canvas = torch.zeros(1, 3, 21, 1, 512)
        depth = torch.full((1, 21, 1, 256), 0.25)
        return {
            'video': [canvas],
            'depth_m': [depth],
            'depth_metric_mask': [(depth > 0) & (depth < 0.5)],
        }

    def test_zero_weight_preserves_original_step(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        class Parent:
            def __init__(self, config): self.config = config
            def training_step(self, batch, iteration): return batch, iteration
        scope = dict(OmniMoTModel=Parent, torch=torch)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), 'exec'), scope)
        model = scope['UMIFTDepthModel'](None, depth_aux_weight=0.0)
        batch = {'untouched': object()}
        output, loss = model.training_step(batch, 9)
        self.assertIs(output, batch)
        self.assertEqual(loss, 9)
        with self.assertRaises(ValueError): scope['UMIFTDepthModel'](None, depth_aux_weight=-1)

    def test_nonfinite_aux_is_rejected_before_returning_training_loss(self):
        result = SimpleNamespace(
            loss=torch.tensor(float('nan')),
            has_empty_support=torch.tensor(False),
        )
        model = self._make_positive_weight_model(lambda **kwargs: result)

        with self.assertRaisesRegex(ValueError, 'synchronized invalid depth batch'):
            model.training_step(self._valid_batch(), 1)

    def test_runtime_aux_failure_uses_synchronized_error_contract(self):
        def fail(**kwargs):
            raise RuntimeError('decode cache is invalid')

        model = self._make_positive_weight_model(fail)

        with self.assertRaisesRegex(ValueError, 'synchronized invalid depth batch'):
            model.training_step(self._valid_batch(), 1)

if __name__ == '__main__': unittest.main()
