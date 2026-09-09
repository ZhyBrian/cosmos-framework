"""Four-rank E3 correctness gate on training windows, including both future modalities."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('zarr','sft-toml','checkpoint','output'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1,2,3' or os.environ.get('WORLD_SIZE')!='4':
        raise ValueError('E3 model gate requires four A40 ranks on GPU0-3')
    if args.output.exists():raise FileExistsError(args.output)
    import torch
    from cosmos_framework.inference.common.init import init_script
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import get_umift_rgbd_sft_dataset
    from cosmos_framework.data.generator.action.datasets.umift_history_dataset import get_umift_history_sft_dataset
    from examples.umift.rgbd_infer import build_rgbd_batch,load_rgbd_model,run_rgbd_prediction,split_prediction
    from examples.umift.infer import _move_batch_to_cuda,validate_independent_parallelism
    from examples.umift.long_rollout import array_sha
    init_script()
    try:
        model,resolved,evidence=load_rgbd_model(args.sft_toml,args.checkpoint)
        validate_independent_parallelism(model.parallel_dims)
        rank=torch.distributed.get_rank()
        kwargs=dict(split='refit_train',stage='e1',seed=42,resolution='256',fps=15.,history_frames=5,
                    tokenizer_config=resolved.model.config.vlm_config.tokenizer,
                    max_action_dim=int(resolved.model.config.max_action_dim))
        dataset=get_umift_rgbd_sft_dataset(str(args.zarr),**kwargs)
        old=get_umift_history_sft_dataset(str(args.zarr),**kwargs)
        from cosmos_framework.utils.lazy_config import instantiate
        loader=instantiate(resolved.dataloader_train)
        packed=next(iter(loader))
        assert packed['is_preprocessed'] is True
        assert tuple(packed['video'][0].shape)==(1,3,21,256,512)
        assert packed['video'][0].dtype==torch.float32
        def scalar(value):
            while isinstance(value,(list,tuple)):value=value[0]
            return int(value.item() if hasattr(value,'item') else value)
        packed_ep=scalar(packed['episode_id']);packed_start=scalar(packed['window_start'])
        assert packed_ep not in (13,43,49)
        packed_reference=dataset.get_window(packed_ep,packed_start)
        assert torch.equal(packed['video'][0][0],packed_reference['video'])
        del loader,packed,packed_reference
        sample=dataset.get_window(rank,64)
        legacy=old.get_window(rank,64)
        assert torch.equal(sample['action'],legacy['action'])
        assert torch.equal(sample['physical_action'],legacy['physical_action'])
        assert torch.equal(sample['video'][:,:,:,:256],legacy['video'].float()/127.5-1)
        assert torch.equal(sample['video_source_indices'],legacy['video_source_indices'])
        original=sample['video'].clone()
        batch=build_rgbd_batch(sample)
        changed=sample.copy();changed['video']=original.clone();changed['video'][:,5:]=-original[:,5:]
        assert torch.equal(batch['video'][0],build_rgbd_batch(changed)['video'][0])
        with torch.inference_mode():
            latent=model.encode(original.unsqueeze(0).cuda())
            perturb=model.encode(changed['video'].unsqueeze(0).cuda())
            prefix=model.encode(original[:,:5].unsqueeze(0).cuda())
            delta=float((latent[:,:,:2]-perturb[:,:,:2]).abs().max())
            assert delta==0,delta
            assert tuple(latent.shape)==(1,48,6,16,32),latent.shape
            prediction=run_rgbd_prediction(model,_move_batch_to_cuda(batch),noise_seed=0,num_steps=30)
            rgb,depth=split_prediction(prediction)
        report=dict(rank=rank,episode_id=rank,start=64,load_evidence=evidence,
                    future_rgbd_prefix_max_abs=delta,
                    history_only_prefix_max_abs=float((latent[:,:,:2]-prefix).abs().max()),
                    latent_shape=list(latent.shape),canvas_shape=list(original.shape),
                    action_shape=list(sample['action'].shape),action_offset=5,clean_latents=[0,1],
                    legacy_action_exact=True,legacy_rgb_exact=True,future_batch_identical=True,
                    formal_packed_batch_verified=True,packed_episode=packed_ep,packed_start=packed_start,
                    input_dtype=str(original.dtype),prediction_dtype=str(prediction.dtype),
                    prediction_finite=bool(np.isfinite(prediction).all()),
                    depth_raw_min_m=float(depth.min()),depth_raw_max_m=float(depth.max()),
                    raw_out_of_range_fraction=float(((depth<0)|(depth>.5)).mean()),
                    physical_action_sha256=array_sha(sample['physical_action'].numpy()),
                    reserved_gib=torch.cuda.max_memory_reserved()/1024**3,passed=True)
        if rank==0:args.output.mkdir(parents=True)
        torch.distributed.barrier()
        np.savez(args.output/f'rank_{rank}_prediction.npz',rgb=rgb,depth_m=depth)
        (args.output/f'rank_{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        torch.distributed.barrier()
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()


if __name__=='__main__':main()
