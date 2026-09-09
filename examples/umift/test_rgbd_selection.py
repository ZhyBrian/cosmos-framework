import copy
import pytest


def reports():
    return [dict(iteration=s,history_frames=5,experiment_id='E3-Dout',protocol_sha256='frozen',
        checkpoint=f'/data/cosmos_runs/E3/action_fd_umift_edge_rgbd_h5/checkpoints/iter_{s:09d}/model',
        rgb_weight=.5,session_equal_aggregate={'overall':{'lpips':v,'depth_mae_m':.01}},
        persistence_session_equal_aggregate={'overall':{'lpips':.4,'depth_mae_m':.02}})
        for s,v in zip((500,1000,1500,2000,2500,3000),(.3,.2,.1,.1,.2,.3))]


def test_rgbd_selection_uses_frozen_components_and_early_tie():
    from examples.umift.rgbd_selection import choose_candidate
    assert choose_candidate(reports(),'frozen',.5)['iteration']==1500


@pytest.mark.parametrize('change',['missing','protocol','baseline','weight','checkpoint','nan'])
def test_rgbd_selection_rejects_incompatible_candidates(change):
    from examples.umift.rgbd_selection import choose_candidate
    r=copy.deepcopy(reports())
    if change=='missing':r.pop()
    elif change=='protocol':r[0]['protocol_sha256']='other'
    elif change=='baseline':r[0]['persistence_session_equal_aggregate']['overall']['depth_mae_m']=.03
    elif change=='weight':r[0]['rgb_weight']=.7
    elif change=='checkpoint':r[0]['checkpoint']=r[0]['checkpoint'].replace('rgbd_h5','h5')
    else:r[0]['session_equal_aggregate']['overall']['depth_mae_m']=float('nan')
    with pytest.raises(ValueError):choose_candidate(r,'frozen',.5)


def test_protocol_rejects_changed_executable_semantics(tmp_path):
    import json
    from examples.umift.rgbd_selection import load_protocol
    p=dict(protocol='e3-rgbd-selection-v1',history_frames=5,selection_uses_test_episodes=True,
           iterations=[500,1000,1500,2000,2500,3000],rgb_weight=.5,
           prediction_clamp_for_depth_metrics=False,num_steps=2,windows=[{}]*24)
    path=tmp_path/'protocol.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError):load_protocol(path)
