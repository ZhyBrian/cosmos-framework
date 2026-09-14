"""D1-only model wiring checks without loading the CUDA model dependencies."""
import ast
from pathlib import Path
import unittest
import torch

SOURCE = Path(__file__).parents[2] / 'cosmos_framework/model/generator/umift_depth_model.py'

class DepthModelWiringTest(unittest.TestCase):
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

if __name__ == '__main__': unittest.main()
