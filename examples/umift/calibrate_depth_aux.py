"""Calibrate D1 weight from 32 training windows; no parameter updates or test data."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--draws', type=int, default=8)
    args = ap.parse_args()
    import torch
    import torch.distributed as dist
    import numpy as np
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml
    from cosmos_framework.utils import distributed, misc
    from cosmos_framework.utils.context_managers import data_loader_init, distributed_init, model_init
    from cosmos_framework.utils.lazy_config import instantiate
    from examples.umift.probe_loss import reset_probe_rng, tensor_tree_sha256, _first_int
    cfg = load_experiment_from_toml('examples/toml/sft_config/action_fd_umift_edge_rgbd_d1.toml',
                                   extra_overrides=['job.name=d1_calibration'])
    cfg.validate(); cfg.freeze()
    trainer = cfg.trainer.type(cfg)
    keys, source = trainer.checkpointer.keys_to_resume_during_load()
    if source is None or not source.warm_start or set(keys) != {'model'}:
        raise ValueError('calibration requires model-only E3 warmstart')
    with distributed_init(): distributed.init()
    if dist.get_world_size() != 4: raise ValueError('four A40 ranks required')
    rank = dist.get_rank()
    with model_init(): model = instantiate(cfg.model)
    with data_loader_init(): loader = instantiate(cfg.dataloader_train)
    model = model.to('cuda', memory_format=cfg.trainer.memory_format)
    model.on_train_start(cfg.trainer.memory_format)
    trainer.checkpointer.load(model)
    optimizer, scheduler = model.init_optimizer_scheduler(cfg.optimizer, cfg.scheduler)
    params = [p for group in optimizer.param_groups for p in group['params']]
    loader.set_start_iteration(0); iterator = iter(loader); model.train()
    rows = []
    def local(t): return t.to_local() if hasattr(t, 'to_local') else t
    for draw in range(args.draws):
        batch = misc.to(next(iterator), device='cuda')
        ep = _first_int(batch['episode_id']); start = _first_int(batch['window_start'])
        if ep in (13,43,49): raise ValueError('heldout entered calibration')
        fm_grads = []
        for pass_index, weight in enumerate((0.0, 1.0)):
            optimizer.zero_grad(set_to_none=True)
            reset_probe_rng(draw, rank)
            model.depth_aux_weight = weight
            output, loss = model.training_step(batch, draw)
            identity = tensor_tree_sha256([output['x0'],output['xt'],output['sigma']])
            if pass_index == 0: first_identity = identity
            elif identity != first_identity: raise ValueError('calibration paired latent/noise mismatch')
            loss.backward()
            if pass_index == 0:
                fm_value = float(loss.detach())
                fm_grads = [None if p.grad is None else local(p.grad).detach().float().cpu().clone() for p in params]
            else:
                fm_sq = aux_sq = dot = 0.0
                for p,g0 in zip(params,fm_grads,strict=True):
                    if p.grad is None:
                        if g0 is not None: raise ValueError('gradient presence changed')
                        continue
                    g1 = local(p.grad).detach().float().cpu()
                    if g0 is None: g0 = torch.zeros_like(g1)
                    ga = g1-g0
                    fm_sq += float(g0.double().square().sum())
                    aux_sq += float(ga.double().square().sum())
                    dot += float((g0.double()*ga.double()).sum())
                sums = torch.tensor([fm_sq,aux_sq,dot],dtype=torch.float64,device='cuda')
                dist.all_reduce(sums)
                f,a,d=sums.cpu().tolist()
                if not np.isfinite([f,a,d]).all() or min(f,a)<=0: raise ValueError('invalid calibration gradients')
                row=dict(draw=draw,rank=rank,episode=ep,start=start,identity=identity,
                         fm_loss=fm_value,aux_loss=float(output['depth_aux_loss']),
                         fm_grad_norm=f**0.5,aux_grad_norm=a**0.5,cosine=d/(f*a)**0.5,
                         peak_gib=torch.cuda.max_memory_allocated()/2**30)
                rows.append(row); print(json.dumps(row),flush=True)
        del output,loss,fm_grads
    all_rows=[None]*4; dist.all_gather_object(all_rows,rows)
    if rank==0:
        f=np.median([r['fm_grad_norm'] for r in rows]); a=np.median([r['aux_grad_norm'] for r in rows])
        result=dict(protocol='d1-train-gradient-calibration-v1',windows=args.draws*4,
                    lambda_depth=float(.1*f/a),target_initial_gradient_ratio=.1,
                    parameter_updates=0,checkpoint=os.environ['BASE_CHECKPOINT_PATH'],
                    rows=[r for rr in all_rows for r in rr])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__=='__main__': main()
