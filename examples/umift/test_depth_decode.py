#!/usr/bin/env python3
"""CPU-only contract tests for the differentiable Wan depth decode path."""

import sys
import types
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# These dependencies are unrelated to VAE decoding and are unavailable in the
# lightweight CPU test environment.
log_module = types.ModuleType("cosmos_framework.utils.log")
log_module.info = lambda *args, **kwargs: None
sys.modules.setdefault("cosmos_framework.utils.log", log_module)

distributed_module = types.ModuleType("cosmos_framework.utils.distributed")
distributed_module.get_rank = lambda: 0
distributed_module.sync_model_states = lambda module: module
sys.modules.setdefault("cosmos_framework.utils.distributed", distributed_module)

easy_io_module = types.ModuleType("cosmos_framework.utils.easy_io")
easy_io_module.easy_io = object()
sys.modules.setdefault("cosmos_framework.utils.easy_io", easy_io_module)

interface_module = types.ModuleType("cosmos_framework.model.generator.tokenizers.interface")
interface_module.VideoTokenizerInterface = object
sys.modules.setdefault("cosmos_framework.model.generator.tokenizers.interface", interface_module)

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (  # noqa: E402
    Wan2pt2VAEInterface,
    WanVAE,
    WanVAE_,
)


class _RecordingDecoder(torch.nn.Module):
    def __init__(self, fail_on_call: int | None = None):
        super().__init__()
        self.calls = 0
        self.fail_on_call = fail_on_call

    def forward(self, x, feat_cache, first_chunk):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("injected decoder failure")
        feat_cache[0] = x.clone()
        return x.repeat(1, 12, 1, 1, 1)


def _make_low_level_decoder(fail_on_call: int | None = None) -> WanVAE_:
    model = WanVAE_.__new__(WanVAE_)
    torch.nn.Module.__init__(model)
    model.z_dim = 1
    model.conv2 = torch.nn.Identity()
    model.decoder = _RecordingDecoder(fail_on_call)
    model._dec_conv_num = 1
    model._dec_cache = [torch.ones(1)]

    def compiled_decoder(*args, **kwargs):
        raise AssertionError("the differentiable path must use the eager decoder")

    model._compiled_decoder_forward = compiled_decoder
    return model


class DifferentiableWanDecodeTest(unittest.TestCase):
    def test_low_level_decode_is_differentiable_eager_and_fresh(self):
        model = _make_low_level_decoder()
        latent = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        scale = (torch.zeros(1), torch.ones(1))

        decoded = model.decode_with_grad(latent, scale)
        decoded.sum().backward()

        self.assertEqual(decoded.shape, (1, 3, 2, 4, 4))
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)
        self.assertEqual(model.decoder.calls, 2)
        self.assertEqual(model._dec_cache, [None])

    def test_low_level_decode_clears_cache_after_failure(self):
        model = _make_low_level_decoder(fail_on_call=2)
        latent = torch.randn(1, 1, 2, 2, 2, requires_grad=True)
        scale = (torch.zeros(1), torch.ones(1))

        with self.assertRaisesRegex(RuntimeError, "injected decoder failure"):
            model.decode_with_grad(latent, scale)

        self.assertEqual(model._dec_cache, [None])

    def test_wan_wrapper_preserves_input_dtype_and_gradient(self):
        class InnerModel:
            def decode_with_grad(self, latent, scale):
                self.received_dtype = latent.dtype
                return latent * scale[1].view(1, 1, 1, 1, 1)

        wrapper = WanVAE.__new__(WanVAE)
        wrapper.dtype = torch.float64
        wrapper.scale = (torch.zeros(1, dtype=torch.float64), torch.full((1,), 2.0, dtype=torch.float64))
        wrapper.model = InnerModel()
        latent = torch.ones(1, 1, 1, 1, 1, dtype=torch.float32, requires_grad=True)

        decoded = wrapper.decode_with_grad(latent)
        decoded.sum().backward()

        self.assertEqual(wrapper.model.received_dtype, torch.float64)
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertEqual(latent.grad.item(), 2.0)

    def test_interface_rejects_decoder_override_and_cached_scope(self):
        interface = Wan2pt2VAEInterface.__new__(Wan2pt2VAEInterface)
        interface.model = object()
        interface._decoder_override = object()
        interface._keep_decoder_cache = False
        latent = torch.zeros(1)

        with self.assertRaisesRegex(RuntimeError, "decoder override"):
            interface.decode_with_grad(latent)

        interface._decoder_override = None
        interface._keep_decoder_cache = True
        with self.assertRaisesRegex(RuntimeError, "cached decoder"):
            interface.decode_with_grad(latent)

    def test_interface_forwards_to_native_differentiable_decoder(self):
        class NativeWan:
            def decode_with_grad(self, latent):
                return latent.square()

        interface = Wan2pt2VAEInterface.__new__(Wan2pt2VAEInterface)
        interface.model = NativeWan()
        interface._decoder_override = None
        interface._keep_decoder_cache = False
        latent = torch.tensor([3.0], requires_grad=True)

        decoded = interface.decode_with_grad(latent)
        decoded.sum().backward()

        self.assertEqual(decoded.item(), 9.0)
        self.assertEqual(latent.grad.item(), 6.0)


if __name__ == "__main__":
    unittest.main()
