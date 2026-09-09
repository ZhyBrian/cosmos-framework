"""Failure-critical contracts for E3 RGBD inference and physical metrics."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def sample():
    return dict(history_frames=5, mode="forward_dynamics", ai_caption="",
                video=torch.linspace(-1, 1, 3*21*256*512).reshape(3,21,256,512),
                action=torch.zeros(16,64), conditioning_fps=torch.tensor(15.),
                image_size=torch.tensor([256,512,256,512]), domain_id=torch.tensor(6),
                raw_action_dim=10, is_preprocessed=True,
                sequence_plan=SimpleNamespace(condition_frame_indexes_vision=[0,1],
                    condition_frame_indexes_action=list(range(16)),action_start_frame_offset=5,
                    has_text=True,has_vision=True,has_action=True))


def test_future_rgb_depth_and_sidecars_cannot_enter_inference():
    from examples.umift.rgbd_infer import build_rgbd_batch
    s=sample(); old=s['video'].clone()
    a=build_rgbd_batch(s)
    s['video'][:,5:]=torch.rand_like(s['video'][:,5:])*2-1
    s['depth_m']=torch.rand(21,256,256)
    b=build_rgbd_batch(s)
    assert a['is_preprocessed'] is True
    assert a['video'][0].shape == (1,3,21,256,512)
    assert torch.equal(a['video'][0],b['video'][0])
    assert torch.equal(a['video'][0][0,:,:5],old[:,:5])
    assert not a['video'][0][:,:,5:].count_nonzero()
    assert 'depth_m' not in b


def test_uint8_and_wrong_action_offset_rejected():
    from examples.umift.rgbd_infer import build_rgbd_batch
    s=sample();s['video']=torch.zeros_like(s['video'],dtype=torch.uint8)
    with pytest.raises(ValueError,match='float'):build_rgbd_batch(s)
    s=sample();s['sequence_plan'].action_start_frame_offset=1
    with pytest.raises(ValueError,match='offset'):build_rgbd_batch(s)


def test_depth_error_uses_gt_strict_support_and_raw_prediction():
    from examples.umift.rgbd_metrics import depth_video_metrics
    gt=np.array([[[0,.1,.5,.2]],[[0,.1,.5,.2]]],dtype=np.float32)
    pred=np.array([[[0,.1,.5,.2]],[[-.2,.12,.45,.6]]],dtype=np.float32)
    r=depth_video_metrics(gt,pred)
    assert r['mean']['depth_mae_m']==pytest.approx(.21)
    assert r['mean']['depth_rmse_m']==pytest.approx(np.sqrt((.02**2+.4**2)/2))
    assert r['mean']['depth_cap_underprediction_m']==pytest.approx(.05)
    assert r['mean']['depth_out_of_range_fraction']==pytest.approx(.5)
    assert r['mean']['depth_valid_fraction']==.5
    assert r['mean']['depth_zero_region_mean_m']==pytest.approx(-.2)


def test_depth_nonfinite_or_invalid_truth_rejected():
    from examples.umift.rgbd_metrics import depth_video_metrics
    gt=np.full((2,2,2),.2,np.float32);p=gt.copy();p[1,0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):depth_video_metrics(gt,p)
    gt[1,0,0]=.6
    with pytest.raises(ValueError,match='truth'):depth_video_metrics(gt,np.zeros_like(gt))


def test_joint_score_units_and_explicit_weight():
    from examples.umift.rgbd_metrics import joint_selection_score
    assert joint_selection_score(.2,.01,.4,.02,rgb_weight=.5)==pytest.approx(.5)
    assert joint_selection_score(.2,.01,.4,.02,rgb_weight=0)==pytest.approx(.5)
    with pytest.raises(ValueError):joint_selection_score(.2,.01,0,.02,rgb_weight=.5)


def test_joint_canvas_loss_mask_and_action_time_positions():
    from cosmos_framework.data.generator.sequence_packing.sequence import PackedSequenceBuilder
    b=PackedSequenceBuilder();b.begin_sample(0)
    b.pack_vision_tokens(input_vision_tokens=torch.zeros(1,48,6,16,32),
        condition_frame_indexes_vision=[0,1],input_timestep=.5,latent_patch_size=2,
        vision_fps=15.,enable_fps_modulation=True,base_fps=24.,temporal_compression_factor=4,
        vision_temporal_positions=None,temporal_position_period=None)
    b.pack_action_tokens(input_action_tokens=torch.zeros(16,64),
        condition_frame_indexes_action=list(range(16)),input_timestep=.5,action_temporal_offset=0,
        enable_fps_modulation=True,base_fps=24.,action_fps=15.,base_temporal_compression_factor=4,
        action_start_frame_offset=5)
    assert b.vision.condition_mask[0].flatten().tolist()==[1.,1.,0.,0.,0.,0.]
    assert len(b.vision.mse_loss_indexes)==4*8*16
    assert b.action.mse_loss_indexes==[]
    pos=torch.cat(b.position_ids,dim=1)
    assert pos[0,-16:].tolist()==pytest.approx([.4*i for i in range(5,21)])
