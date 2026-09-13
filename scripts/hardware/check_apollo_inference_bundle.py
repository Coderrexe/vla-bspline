"""GPU prediction/transfer check only. No Dora or robot API is imported."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

from apollo_predictor import ApolloPredictor
from apollo_legacy_state import CheckpointStateAdapter, training_to_current_tcp_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.bundle/'handoff_manifest.json').read_text())
    if manifest['status'] != 'OFFLINE_VERIFIED' or manifest['not_ready']:
        raise ValueError('Incomplete bundle')
    report = {'robot_connected': False, 'dora_connected': False, 'models': [],
              'gpu': torch.cuda.get_device_name(torch.device(args.device)),
              'packages': {k: importlib.metadata.version(k) for k in
                           ['torch', 'torchvision', 'transformers', 'dora-rs', 'numpy', 'scipy']}}
    torch.set_num_threads(4)
    for entry in manifest['verified_models']:
        path = args.bundle/entry['relative_path']
        with (path/'model.safetensors').open('rb') as f:
            actual_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual_hash != entry['model_sha256']:
            raise ValueError(f'Weight hash mismatch: {path}')
        model = ApolloPredictor(path, device=args.device)
        with np.load(path/'prediction_fixture.npz') as f:
            state, view, grip = (f[k].copy() for k in ['state', 'view_wrist', 'grip_wrist'])
            expected, seed = f['expected_action'].copy(), int(f['seed'])
        latencies = []
        for _ in range(6):
            model.reset()
            torch.manual_seed(seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actual = model.predict_chunk(state, view, grip)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter()-start)
        if actual.shape != (8, 16) or not np.isfinite(actual).all():
            raise ValueError('Invalid action prefix')
        reconstructed = CheckpointStateAdapter(path)(training_to_current_tcp_state(state))
        model.reset()
        torch.manual_seed(seed)
        adapted = model.predict_chunk(reconstructed, view, grip)
        # A numerical reproducibility gate, NOT a robot safety bound.
        matches = bool(np.allclose(actual, expected, atol=1e-5, rtol=5e-3))
        adapter_matches = bool(np.allclose(adapted, actual, atol=1e-5, rtol=5e-3))
        row = {'model': path.name, 'weight_sha256': actual_hash, 'strict_load': 'PASS',
               'reference_prediction_matches': matches,
               'max_reference_difference_by_channel': np.abs(actual-expected).max(0).tolist(),
               'state_adapter_roundtrip_max_difference': float(np.max(abs(reconstructed-state))),
               'adapter_prediction_matches': adapter_matches,
               'max_adapter_prediction_difference_by_channel': np.abs(adapted-actual).max(0).tolist(),
               'latency_ms_median': float(1000*np.median(latencies[1:])),
               'latency_ms_max': float(1000*np.max(latencies[1:]))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output.with_name(args.output.stem+'_'+path.name+'.npz'),
                            expected=expected, actual=actual, adapted=adapted)
        report['models'].append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report['status'] = ('PASS' if all(r['reference_prediction_matches'] and
                        r['adapter_prediction_matches'] for r in report['models']) else 'REVIEW_REQUIRED')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if report['status'] != 'PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
