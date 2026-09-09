"""Frozen E3-Dout selection; RGB and depth criteria are explicit protocol fields."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from examples.umift.long_rollout import EPISODES, array_sha, file_sha
from examples.umift.protocol import derive_noise_seed
from examples.umift.rgbd_metrics import depth_video_metrics, joint_selection_score

ITERATIONS = (500, 1000, 1500, 2000, 2500, 3000)
# Frozen canonical dataset identities, independently audited before E3 implementation.
HELD_OUT = {13: (2695, "20260730135340388"), 43: (1816, "20260730171255401"),
            49: (1806, "20260730173714606")}


def expected_windows() -> list[dict]:
    windows = []
    for ep, (length, session) in HELD_OUT.items():
        for start in np.rint(np.linspace(0, length - 33, 8)).astype(int):
            wid = f"episode_{ep}:s={start}"
            windows.append(dict(episode_id=ep, start=int(start), raw_session=session,
                                window_id=wid, noise_seed=derive_noise_seed(wid, 0)))
    return windows



def validate_candidate_identity(checkpoint: Path, iteration: int) -> None:
    if (checkpoint.name != 'model' or checkpoint.parent.name != f'iter_{iteration:09d}'
            or 'action_fd_umift_edge_rgbd_h5' not in checkpoint.parts):
        raise ValueError('checkpoint is not the requested E3-Dout H5 iteration')


def freeze(zarr_path: Path, output: Path, rgb_weight: float) -> None:
    import zarr
    if output.exists():raise FileExistsError(output)
    joint_selection_score(1,1,1,1,rgb_weight=rgb_weight)
    root=zarr.open_group(str(zarr_path),mode='r')
    windows=[]
    for ep in EPISODES:
        g=root['data'][f'episode_{ep}'];last=int(g['rgb_0'].shape[0])-33
        starts=np.rint(np.linspace(0,last,8)).astype(int)
        if len(set(starts.tolist()))!=8 or last<0:raise ValueError('insufficient legal windows')
        for start in starts:
            wid=f'episode_{ep}:s={start}'
            windows.append(dict(episode_id=ep,start=int(start),raw_session=str(g.attrs['src']).split('#')[0],
                                window_id=wid,noise_seed=derive_noise_seed(wid,0)))
    if windows != expected_windows():raise ValueError('source differs from the audited fixed 24 windows')
    report=dict(protocol='e3-rgbd-selection-v1',experiment_id='E3-Dout',history_frames=5,
        iterations=list(ITERATIONS),rgb_weight=rgb_weight,
        selection_metric='weighted_ratio_of_session_equal_lpips_and_depth_mae_to_persistence',
        aggregation_order='mean future frames, mean windows per session, mean sessions, then ratio to P',
        depth_mask='finite(GT) & 0<GT<0.5; never predicted-mask intersection',depth_units='metres',
        prediction_clamp_for_depth_metrics=False,tie_break='earlier_iteration',num_steps=30,
        selection_uses_test_episodes=True,zarr_path=str(zarr_path),windows=windows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')


def load_protocol(path: Path) -> tuple[dict,str]:
    p=json.loads(path.read_text())
    if (p.get('protocol')!='e3-rgbd-selection-v1' or p.get('history_frames')!=5
            or p.get('selection_uses_test_episodes') is not True
            or p.get('iterations')!=list(ITERATIONS) or len(p.get('windows',[]))!=24
            or p.get('prediction_clamp_for_depth_metrics') is not False
            or p.get('experiment_id') != 'E3-Dout' or p.get('num_steps') != 30
            or p.get('windows') != expected_windows()
            or p.get('depth_units') != 'metres'
            or p.get('selection_metric') != 'weighted_ratio_of_session_equal_lpips_and_depth_mae_to_persistence'
            or p.get('aggregation_order') != 'mean future frames, mean windows per session, mean sessions, then ratio to P'
            or p.get('depth_mask') != 'finite(GT) & 0<GT<0.5; never predicted-mask intersection'
            or p.get('tie_break') != 'earlier_iteration'):
        raise ValueError('invalid frozen E3-Dout protocol')
    joint_selection_score(1,1,1,1,rgb_weight=p['rgb_weight'])
    return p,file_sha(path)


def infer(args):
    import torch
    from cosmos_framework.data.generator.action.datasets.umift_rgbd_dataset import get_umift_rgbd_sft_dataset
    from cosmos_framework.inference.common.init import init_script
    from examples.umift.rgbd_infer import build_rgbd_batch,load_rgbd_model,run_rgbd_prediction,split_prediction
    from examples.umift.infer import _move_batch_to_cuda,validate_independent_parallelism,validate_launch_environment
    validate_launch_environment(os.environ);init_script()
    p,sha=load_protocol(args.protocol)
    if args.iteration not in ITERATIONS:raise ValueError('candidate not preregistered')
    validate_candidate_identity(args.checkpoint,args.iteration)
    model,resolved,evidence=load_rgbd_model(args.sft_toml,args.checkpoint)
    validate_independent_parallelism(model.parallel_dims)
    ds=get_umift_rgbd_sft_dataset(p['zarr_path'],split='history',history_frames=5,
        tokenizer_config=resolved.model.config.vlm_config.tokenizer,
        max_action_dim=int(resolved.model.config.max_action_dim))
    rank=torch.distributed.get_rank()
    if rank==0:args.output.mkdir(parents=True,exist_ok=False)
    torch.distributed.barrier();rows=[]
    with torch.inference_mode():
        for i,w in enumerate(p['windows']):
            if i%4!=rank:continue
            s=ds.get_window(w['episode_id'],w['start'])
            truth=((s['video'][:,:,:,:256].permute(1,2,3,0).numpy()[4:]+1)/2).astype(np.float32)
            depth_truth=s['depth_m'].numpy()[4:].astype(np.float32)
            full=run_rgbd_prediction(model,_move_batch_to_cuda(build_rgbd_batch(s)),
                                     noise_seed=w['noise_seed'],num_steps=p['num_steps'])
            rgb,dep=split_prediction(full)
            prediction=np.concatenate((truth[:1],rgb[5:]),axis=0)
            depth_prediction=np.concatenate((depth_truth[:1],dep[5:]),axis=0)
            dest=args.output/f'window_{i:02d}.npz'
            np.savez(dest,truth=truth,prediction=prediction,depth_truth=depth_truth,depth_prediction=depth_prediction)
            rows.append({**w,'window_index':i,'file':dest.name,'sha256':file_sha(dest),
                         'physical_action_sha256':array_sha(s['physical_action'].numpy()),
                         'history_source_indices':s['history_source_indices'].tolist(),
                         'history_padding_count':s['history_padding_count']})
            print(json.dumps(dict(window_done=i,iteration=args.iteration)),flush=True)
    if file_sha(args.protocol)!=sha:raise ValueError('protocol changed during inference')
    (args.output/f'rank_{rank}.json').write_text(json.dumps(dict(history_frames=5,experiment_id='E3-Dout',
        iteration=args.iteration,protocol_sha256=sha,load_evidence=evidence,rank=rank,rows=rows,complete=True),indent=2)+'\n')
    torch.distributed.barrier();torch.distributed.destroy_process_group()


def _flat(rgb_metrics,depth_metrics):
    return {**rgb_metrics['mean'],'temporal_l1':rgb_metrics['temporal']['mean_l1'],
            'depth_mae_m':depth_metrics['mean']['depth_mae_m'],
            'depth_rmse_m':depth_metrics['mean']['depth_rmse_m'],
            'depth_valid_fraction':depth_metrics['mean']['depth_valid_fraction'],
            'depth_out_of_range_fraction':depth_metrics['mean']['depth_out_of_range_fraction']}


def score(args):
    from examples.umift.evaluate import aggregate_session_equal,evaluate_video_pair
    from examples.umift.score_refit_reference import _lpips_metric
    p,sha=load_protocol(args.protocol)
    ranks=[json.loads((args.input/f'rank_{r}.json').read_text()) for r in range(4)]
    identity={(r['iteration'],r['load_evidence']['checkpoint']) for r in ranks}
    if len(identity)!=1 or any(r['protocol_sha256']!=sha or not r['complete'] for r in ranks):
        raise ValueError('rank identities or completion disagree')
    rows=sorted([w for r in ranks for w in r['rows']],key=lambda w:w['window_index'])
    if [r['window_index'] for r in rows]!=list(range(24)):raise ValueError('missing/duplicate window')
    metric=_lpips_metric();results=[];model_flat=[];p_flat=[]
    for row,w in zip(rows,p['windows'],strict=True):
        if any(row[k]!=v for k,v in w.items()):raise ValueError('window identity changed')
        src=args.input/row['file']
        if file_sha(src)!=row['sha256']:raise ValueError('prediction file changed')
        with np.load(src,allow_pickle=False) as d:
            rgb=evaluate_video_pair(d['truth'],d['prediction'],include_lpips=True,lpips_metric=metric)
            dep=depth_video_metrics(d['depth_truth'],d['depth_prediction'])
            prgb=evaluate_video_pair(d['truth'],np.repeat(d['truth'][:1],17,axis=0),include_lpips=True,lpips_metric=metric)
            pdep=depth_video_metrics(d['depth_truth'],np.repeat(d['depth_truth'][:1],17,axis=0))
        model_flat.append(dict(raw_session=row['raw_session'],metrics=_flat(rgb,dep)))
        p_flat.append(dict(raw_session=row['raw_session'],metrics=_flat(prgb,pdep)))
        results.append({**row,'rgb_metrics':rgb,'depth_metrics':dep,'persistence_rgb_metrics':prgb,'persistence_depth_metrics':pdep})
    agg=aggregate_session_equal(model_flat);pagg=aggregate_session_equal(p_flat)
    overall,baseline=agg['overall'],pagg['overall']
    value=joint_selection_score(overall['lpips'],overall['depth_mae_m'],baseline['lpips'],baseline['depth_mae_m'],rgb_weight=p['rgb_weight'])
    iteration,checkpoint=next(iter(identity));validate_candidate_identity(Path(checkpoint),iteration)
    report=dict(experiment_id='E3-Dout',history_frames=5,iteration=iteration,checkpoint=checkpoint,
        protocol_sha256=sha,rgb_weight=p['rgb_weight'],selection_uses_test_episodes=True,
        session_equal_aggregate=agg,persistence_session_equal_aggregate=pagg,joint_score=value,windows=results)
    out=args.input/'metrics.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(iteration=iteration,joint_score=value,overall=overall,persistence=baseline)))


def choose_candidate(reports: list[dict], protocol_sha: str, rgb_weight: float) -> dict:
    if sorted(r['iteration'] for r in reports)!=list(ITERATIONS):raise ValueError('need six distinct candidates')
    baseline=None;scores={}
    for r in reports:
        validate_candidate_identity(Path(r['checkpoint']),r['iteration'])
        if (r['protocol_sha256']!=protocol_sha or r['history_frames']!=5 or r['experiment_id']!='E3-Dout'
                or r['rgb_weight']!=rgb_weight):raise ValueError('candidate protocol or weight differs')
        a,b=r['session_equal_aggregate']['overall'],r['persistence_session_equal_aggregate']['overall']
        bp=(b['lpips'],b['depth_mae_m'])
        if baseline is None:baseline=bp
        elif baseline!=bp:raise ValueError('persistence reference changed between candidates')
        value=joint_selection_score(a['lpips'],a['depth_mae_m'],*bp,rgb_weight=rgb_weight)
        if 'joint_score' in r and not np.isclose(r['joint_score'],value,atol=1e-12,rtol=0):
            raise ValueError('stored joint score does not match frozen formula')
        scores[r['iteration']]=value
    return min(reports,key=lambda r:(scores[r['iteration']],r['iteration']))


def choose(args):
    p,sha=load_protocol(args.protocol)
    paths=[args.input/f'iter_{s:09d}'/'metrics.json' for s in ITERATIONS]
    reports=[json.loads(x.read_text()) for x in paths]
    best=choose_candidate(reports,sha,p['rgb_weight'])
    result=dict(experiment_id='E3-Dout',history_frames=5,iteration=best['iteration'],checkpoint=best['checkpoint'],
        protocol_file=str(args.protocol.resolve()),protocol_sha256=sha,rgb_weight=p['rgb_weight'],
        selection_uses_test_episodes=True,joint_score=best['joint_score'],
        candidates=[dict(iteration=r['iteration'],metrics=r['session_equal_aggregate']['overall'],
                         joint_score=r['joint_score'],file=str(f),sha256=file_sha(f)) for r,f in zip(reports,paths,strict=True)])
    out=args.input/'selected.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--zarr',type=Path,required=True);f.add_argument('--output',type=Path,required=True)
    f.add_argument('--rgb-weight',type=float,required=True,help='User-agreed RGB weight, frozen before candidate evaluation')
    i=sub.add_parser('infer')
    for k in ('protocol','checkpoint','sft-toml','output'):i.add_argument('--'+k,type=Path,required=True)
    i.add_argument('--iteration',type=int,required=True)
    for name in ('score','choose'):
        s=sub.add_parser(name);s.add_argument('--protocol',type=Path,required=True);s.add_argument('--input',type=Path,required=True)
    a=ap.parse_args()
    if a.command=='freeze':freeze(a.zarr,a.output,a.rgb_weight)
    elif a.command=='infer':infer(a)
    elif a.command=='score':score(a)
    else:choose(a)


if __name__=='__main__':main()
