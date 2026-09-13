"""Estimate spline target statistics on training episodes only and test geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from lerobot.policies.smolvla_apollo.configuration_smolvla_apollo import SmolVLAApolloConfig
from lerobot.policies.smolvla_apollo.modeling_smolvla_apollo import apollo_raw_targets, SmolVLAApolloPolicy
from lerobot.policies.smolvla_spline.event_targets import make_event_operator_banks
from lerobot.policies.smolvla_spline.modeling_smolvla_spline import SmolVLASplinePolicy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    args = p.parse_args()
    cfg = SmolVLAApolloConfig(device="cpu")
    pm, bl, pg, bp = make_event_operator_banks(n_ctrl=cfg.n_ctrl, degree=cfg.spline_degree,
                                               min_seg=cfg.min_seg, horizon_max=cfg.horizon_max)
    export = json.loads((args.dataset/'export_manifest.json').read_text())
    t = pq.read_table(args.dataset/'data/chunk-000/file-000.parquet', columns=['action','episode_index'])
    actions = np.asarray(t['action'].to_pylist(), dtype=np.float32)
    ids = np.asarray(t['episode_index'].to_pylist())
    collected, durations, expected_ends, input_windows = [], [], [], []
    for ep in range(export['train_episodes']):
        a = torch.from_numpy(actions[ids == ep])
        for start in range(0, len(a), 256):
            ix = torch.arange(start, min(start+256, len(a)))[:, None] + torch.arange(cfg.horizon_max)[None, :]
            pad = ix >= len(a)
            windows = a[ix.clamp(max=len(a)-1)]
            raw, duration = apollo_raw_targets(windows, pad, cfg, pm, bl, pg)
            collected.append(raw); durations.append(duration)
            mask = (torch.arange(cfg.horizon_max)[None, :] < duration[:,None]) & ~pad
            expected_ends.append((windows[..., :6]*mask[...,None]).sum(1))
            input_windows.append(windows)
    raw, duration = torch.cat(collected), torch.cat(durations)
    means = raw.mean(0); stds = raw.std(0, correction=0).clamp_min(1e-6)
    payload = {'stats_contract_version':1, 'n_ctrl':cfg.n_ctrl, 'degree':cfg.spline_degree,
               'horizon_max':cfg.horizon_max, 'min_seg':cfg.min_seg, 'pause_frac':cfg.pause_frac,
               'fit_end_weight':None, 'action_layout':'eef7', 'pose_dims':list(range(6)),
               'grip_idx':6, 'pass_dims':[], 'boundary_semantics':'production_event_index_k_exclusive',
               'apollo_gripper_event_threshold':cfg.gripper_event_threshold,
               'apollo_gripper_decode':'continuous_open_fraction_0_to_1',
               'pose_ctrl_mean':means[:,:6].tolist(), 'pose_ctrl_std':stds[:,:6].tolist(),
               'grip_ctrl_mean':means[:,6].tolist(), 'grip_ctrl_std':stds[:,6].tolist(),
               'logT_mean':float(raw[:,0,7].mean()),
               'logT_std':float(raw[:,0,7].std(correction=0).clamp_min(1e-6)),
               'train_windows':len(raw), 'train_episodes':export['train_episodes']}
    cfg.embedded_spline_stats = payload
    # Initialize only the spline operators, avoiding a VLM allocation on CPU.
    class Holder(torch.nn.Module):
        _POSE_DIMS = 6
        _N_OUT = 8
    holder = Holder(); holder.config = cfg
    SmolVLASplinePolicy._init_spline_operators(holder, cfg)
    tokens = (raw - holder._tgt_mean) / holder._tgt_std
    endpoint_error = 0.
    pose_sq, grip_sq, count = 0., 0., 0
    windows_all = torch.cat(input_windows)
    expected = torch.cat(expected_ends)
    # Use the actual unbound decoder on a tiny module with the same operators.
    # super() requires an ApolloPolicy instance, so borrow __new__ without a VLM.
    obj = SmolVLAApolloPolicy.__new__(SmolVLAApolloPolicy)
    torch.nn.Module.__init__(obj)
    obj.config=cfg; obj._N_OUT=8; obj._POSE_DIMS=6
    obj.register_buffer('_tgt_mean', holder._tgt_mean)
    obj.register_buffer('_tgt_std', holder._tgt_std)
    for length in range(cfg.min_seg,cfg.horizon_max+1):
        select = duration==length
        if not select.any(): continue
        dec = obj._decode_tokens(tokens[select], length)
        endpoint_error=max(endpoint_error,float((dec[...,:6].sum(1)-expected[select]).abs().max()))
        assert torch.all(dec[...,7:14]==0) and torch.all(dec[...,14]==1) and torch.all(dec[...,15]==0)
        assert torch.all((dec[...,6]>=0)&(dec[...,6]<=1))
        pose_sq += float((dec[...,:6]-windows_all[select,:length,:6]).square().sum())
        grip_sq += float((dec[...,6]-windows_all[select,:length,6]).square().sum())
        count += int(select.sum())*length
    if endpoint_error > 1e-6:
        raise ValueError(f'Endpoint reconstruction error {endpoint_error}')
    report={'windows':len(raw),'endpoint_maxabs':endpoint_error,
            'pose_component_rmse_native_units':(pose_sq/(count*6))**.5,
            'gripper_open_fraction_rmse':(grip_sq/count)**.5,
            'duration_histogram':{str(i):int((duration==i).sum()) for i in range(cfg.min_seg,cfg.horizon_max+1)},
            'geometry':'PASS', 'inactive_outputs':'PASS'}
    (args.dataset/'apollo_spline_stats.json').write_text(json.dumps(payload,indent=2)+'\n')
    (args.dataset/'apollo_spline_reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__=='__main__': main()
