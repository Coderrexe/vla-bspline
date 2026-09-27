"""Offline input sensitivity: alter only gripper state at recorded grasp onset.

Perturbed states are deliberately counterfactual to the images. This diagnoses
model input dependence, not a valid physical rollout. Never use them live.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_apollo_first_grasp import video_frames


def main():
    import torch
    from apollo_predictor import ApolloPredictor
    from apollo_legacy_state import CheckpointStateAdapter, align_quaternion_hemisphere
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--models', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    with np.load(args.inputs/'inputs.npz', allow_pickle=False) as data:
        states, targets = data['states'], data['targets']
    selected = [(i, s) for i, s in enumerate(manifest['samples'])
                if s['split'] == 'validation' and s['offset'] == 0]
    if len(selected) != 5:
        raise ValueError('Expected the same five held-out grasp-boundary inputs')
    images, video_hashes = {}, {}
    for i, sample in selected:
        for camera in ('view_wrist', 'grip_wrist'):
            path = args.videos/'episodes'/sample['episode']/'video'/f'{camera}.mp4'
            video_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            images[i, camera] = video_frames(path, [sample['frame']])[0]
    rows, predictions = [], []
    torch.set_num_threads(4)
    for checkpoint in args.models:
        provenance = json.loads((checkpoint/'dataset_provenance.json').read_text())
        mapping = {e['source_episode_id']: e for e in provenance['episode_mapping']}
        if provenance['task'] != 'Drawer Assembling' or any(
                mapping[s['episode']]['split'] != 'validation'
                or mapping[s['episode']]['source_parquet_sha256'] != s['original_parquet_sha256']
                for _, s in selected):
            raise ValueError('Model split/hash mismatch')
        adapter = CheckpointStateAdapter(checkpoint)
        model = ApolloPredictor(checkpoint, device='cuda:0')
        for i, sample in selected:
            state = align_quaternion_hemisphere(adapter.stabilize_training_state(states[i]), adapter.mean)
            for label, opening in [('recorded', float(state[7])), ('1.0', 1.), ('0.9', .9),
                                   ('0.8', .8), ('0.7', .7), ('0.5', .5)]:
                perturbed = state.copy(); perturbed[7] = opening
                draws = []
                for seed in manifest['seeds']:
                    model.reset(); torch.manual_seed(seed)
                    draws.append(model.predict_chunk(perturbed, images[i, 'view_wrist'], images[i, 'grip_wrist']))
                draws = np.stack(draws); predictions.append(draws)
                rows.append({'model': checkpoint.name, 'episode': sample['episode'], 'frame': sample['frame'],
                             'condition': label, 'recorded_opening_mm': float(state[7]*84),
                             'input_opening_mm': opening*84,
                             'recorded_target_mean8_mm': float(targets[i, :, 6].mean()*84),
                             'predicted_mean8_mm_by_seed': (draws[:, :, 6].mean(1)*84).tolist(),
                             'predicted_xyz_sum_mm_by_seed': (draws[:, :, :3].sum(1)*1000).tolist()})
        del model; gc.collect(); torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'predictions.npz', predictions=np.stack(predictions))
    report = {'purpose': 'OFFLINE_COUNTERFACTUAL_INPUT_SENSITIVITY_ONLY', 'actions_published': 0,
              'robot_api_calls': 0, 'video_sha256': video_hashes, 'seeds': manifest['seeds'],
              'warning': 'Images and gripper measurements intentionally disagree in perturbed cases. Not evidence of successful physical control; never spoof live telemetry.',
              'rows': rows}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
