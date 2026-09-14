"""Real Wan D1 autograd gate using training episode zero only."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import get_umift_rgbd_sft_dataset
    from cosmos_framework.algorithm.loss.umift_depth_aux import compute_depth_aux_loss
    config = load_experiment_from_toml('examples/toml/sft_config/action_fd_umift_edge_rgbd_h5.toml')
    tokenizer = instantiate(config.model.config.tokenizer)
    tokenizer.reset_dtype()
    ds = get_umift_rgbd_sft_dataset(os.environ['DATASET_PATH'], split='refit_train', stage='e1',
        tokenizer_config=config.model.config.vlm_config.tokenizer,
        max_action_dim=int(config.model.config.max_action_dim))
    sample = ds.get_window(0, 64)
    video = sample['video'].unsqueeze(0).cuda()
    with torch.no_grad():
        latent = tokenizer.encode(video)
        reference = tokenizer.decode(latent)
    rows = []
    for repeat in range(2):
        torch.cuda.reset_peak_memory_stats()
        z = latent.detach().clone().requires_grad_(True)
        decoded = tokenizer.decode_with_grad(z)
        forward_error = float((decoded.detach() - reference).abs().max())
        depth = (decoded[:, :, 5:21, :, 256:].float().mean(1) + 1) / 4
        gt = sample['depth_m'][None, 5:21].cuda()
        mask = sample['depth_metric_mask'][None, 5:21].cuda()
        loss = ((depth - gt).abs() * mask).sum((-2, -1)).div(mask.sum((-2, -1))).mean()
        loss.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().max() > 0
        assert all(p.grad is None and not p.requires_grad for p in tokenizer.model.model.parameters())
        rows.append(dict(repeat=repeat, forward_max_abs=forward_error, loss_m=loss.item(),
            gradient_norm=z.grad.norm().item(), shape=list(decoded.shape),
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
        del decoded, depth, loss, z
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(passed=True, episode=0, start=64, rows=rows), indent=2)+'\n')
    print(args.output.read_text(), flush=True)


if __name__ == '__main__':
    main()
