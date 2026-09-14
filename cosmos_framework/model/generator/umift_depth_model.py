"""D1 metric-depth auxiliary supervision; the original E3 model is unchanged."""

import torch

from cosmos_framework.algorithm.loss.umift_depth_aux import compute_depth_aux_loss
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class UMIFTDepthModel(OmniMoTModel):
    def __init__(self, config, depth_aux_weight=0.0):
        if not 0 <= float(depth_aux_weight) < float('inf'):
            raise ValueError('depth_aux_weight must be finite and nonnegative')
        super().__init__(config)
        self.depth_aux_weight = float(depth_aux_weight)

    def training_step(self, data_batch, iteration):
        if self.depth_aux_weight == 0:
            return super().training_step(data_batch, iteration)
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            raise ValueError('D1 depth sidecars currently require CP disabled')
        output, fm_loss = super().training_step(data_batch, iteration)
        device = fm_loss.device
        results = []
        errors = []
        try:
            n = len(output['model_pred'])
            if n != 1 or len(data_batch['video']) != n:
                raise ValueError('D1 requires one RGBD item per microbatch')
            def sidecar(key):
                value = data_batch[key]
                while isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError('D1 sidecar must match the single visual item')
                    value = value[0]
                return value.to(device)
            depth = sidecar('depth_m')
            metric_mask = sidecar('depth_metric_mask')
            # Identity check against the training canvas, never an inference condition.
            canvas = data_batch['video'][0]
            while canvas.ndim > 4 and canvas.shape[0] == 1:
                canvas = canvas[0]
            expected = (canvas[:, :, :, 256:].float().mean(0) + 1) / 4
            if depth.numel() != expected.numel() or not torch.allclose(
                depth.reshape_as(expected).float(), expected, atol=1e-6, rtol=0
            ):
                raise ValueError('depth sidecar does not match training RGBD canvas')
            if not torch.equal(metric_mask, (depth > 0) & (depth < 0.5)):
                raise ValueError('depth metric mask disagrees with raw ground truth')
            results.append(compute_depth_aux_loss(
                x0=output['x0'][0], xt=output['xt'][0], pred=output['model_pred'][0],
                sigma=output['sigma'][0], condition_mask=output['condition_mask_vision'][0],
                decoder=self.tokenizer_vision_gen.decode_with_grad,
                depth_m=depth, depth_metric_mask=metric_mask, error_on_empty_support=False,
            ))
            if bool(results[0].has_empty_support):
                raise ValueError('future depth frame has no valid support')
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
        # All ranks enter the same check before any backward/FSDP gradient collective.
        bad = torch.tensor(int(bool(errors)), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
        if bad.item():
            detail = dict(errors=errors, episode=str(data_batch.get('episode_id')),
                          start=str(data_batch.get('window_start')))
            gathered = [None] * torch.distributed.get_world_size() if torch.distributed.is_initialized() else [detail]
            if torch.distributed.is_initialized():
                torch.distributed.all_gather_object(gathered, detail)
            raise ValueError(f'D1 synchronized invalid depth batch: {gathered}')
        aux = results[0].loss
        loss = fm_loss + self.depth_aux_weight * aux
        output.update(depth_aux_loss=aux.detach(), depth_aux_mae_m=aux.detach() * 0.5,
                      depth_aux_weight=torch.tensor(self.depth_aux_weight, device=device),
                      loss_fm_before_depth=fm_loss.detach())
        return output, loss
