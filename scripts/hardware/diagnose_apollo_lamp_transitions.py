"""Offline lamp closure/release validation, not physical task success.

Uses original corrected-TCP frames. Five held-out episodes and the first five
training episodes are reported separately, with three matched inference seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_apollo_first_grasp import OFFSETS, SEEDS, video_frames


def prepare(args):
    import pyarrow.parquet as pq
    provenance = json.loads(args.provenance.read_text())
    if provenance['task'] != 'Lamp Assembling':
        raise ValueError('This diagnostic requires the lamp task')
    selected = []
    for split in ('validation', 'train'):
        selected.extend([x for x in provenance['episode_mapping'] if x['split'] == split][:5])
    if len(selected) != 10:
        raise ValueError('Expected five held-out and five training episodes')
    samples, states, targets = [], [], []
    for entry in selected:
        episode = entry['source_episode_id']
        path = args.dataset/'episodes'/episode/'frames.parquet'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['source_parquet_sha256']:
            raise ValueError('Unexpected recording revision')
        table = pq.read_table(path, columns=['observation.state', 'action'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        closing = np.flatnonzero((action[1:, 6] < .5) & (action[:-1, 6] >= .5))+1
        release = np.flatnonzero((action[1:, 6] >= .5) & (action[:-1, 6] < .5))+1
        if len(closing) != 1 or len(release) != 1 or closing[0] >= release[0]:
            raise ValueError('Expected one closure then one release')
        for phase, boundary in [('closure', int(closing[0])), ('release', int(release[0]))]:
            for offset in OFFSETS:
                frame = boundary+offset
                if frame < 0 or frame+8 > len(state):
                    raise ValueError('Phase window outside recording')
                samples.append({'episode': episode, 'split': entry['split'], 'phase': phase,
                                'frame': frame, 'boundary_frame': boundary, 'offset': offset,
                                'original_parquet_sha256': digest})
                states.append(state[frame]); targets.append(action[frame:frame+8])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'inputs.npz', states=np.stack(states), targets=np.stack(targets))
    (args.output/'manifest.json').write_text(json.dumps({
        'purpose': 'OFFLINE_TEACHER_FORCED_LAMP_TRANSITIONS_NOT_TASK_SUCCESS',
        'task': provenance['task'], 'state_pose_convention': 'corrected_tcp',
        'phases': ['closure', 'release'], 'offsets': OFFSETS, 'seeds': SEEDS,
        'samples': samples}, indent=2)+'\n')
    print(json.dumps({'samples': len(samples), 'episodes': 10, 'actions_published': 0}))


def evaluate(args):
    import torch
    from apollo_predictor import ApolloPredictor
    from apollo_legacy_state import CheckpointStateAdapter, align_quaternion_hemisphere
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    with np.load(args.inputs/'inputs.npz', allow_pickle=False) as data:
        states, targets = data['states'], data['targets']
    if states.shape != (120, 32) or targets.shape != (120, 8, 16):
        raise ValueError('Prepared lamp transition shape mismatch')
    samples = manifest['samples']
    provenance = json.loads((args.checkpoint/'dataset_provenance.json').read_text())
    mapping = {x['source_episode_id']: x for x in provenance['episode_mapping']}
    if provenance['task'] != manifest['task'] or any(
            mapping.get(s['episode'], {}).get('source_parquet_sha256') != s['original_parquet_sha256']
            or mapping[s['episode']]['split'] != s['split'] for s in samples):
        raise ValueError('Checkpoint data split or hashes differ')
    adapter = CheckpointStateAdapter(args.checkpoint)
    if adapter.pose_convention != manifest['state_pose_convention']:
        raise ValueError('Lamp checkpoint requires corrected TCP inputs')
    images, hashes = {}, {}
    for episode in sorted({s['episode'] for s in samples}):
        indices = sorted({s['frame'] for s in samples if s['episode'] == episode})
        for camera in ('view_wrist', 'grip_wrist'):
            path = args.videos/'episodes'/episode/'video'/f'{camera}.mp4'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for index, frame in zip(indices, video_frames(path, indices)):
                images[episode, index, camera] = frame
    torch.set_num_threads(4)
    model = ApolloPredictor(args.checkpoint, device='cuda:0')
    rows, outputs = [], []
    for i, sample in enumerate(samples):
        state = align_quaternion_hemisphere(adapter.stabilize_training_state(states[i]), adapter.mean)
        predictions = []
        for seed in manifest['seeds']:
            model.reset(); torch.manual_seed(seed)
            predictions.append(model.predict_chunk(
                state, images[sample['episode'], sample['frame'], 'view_wrist'],
                images[sample['episode'], sample['frame'], 'grip_wrist']))
        predictions = np.stack(predictions)
        outputs.append(predictions)
        target = targets[i]
        rows.append({**sample, 'measured_opening_mm': float(states[i, 7]*84),
                     'target_mean8_mm': float(target[:, 6].mean()*84),
                     'predicted_mean8_mm': float(predictions[:, :, 6].mean()*84),
                     'gripper_mae_mm': float(np.abs(predictions[:, :, 6]-target[:, 6]).mean()*84),
                     'xyz_sum_rmse_mm': float(np.sqrt(np.mean(
                         (predictions[:, :, :3].sum(1)-target[:, :3].sum(0))**2))*1000)})
    groups = []
    for split in ('validation', 'train'):
        for phase in manifest['phases']:
            for offset in manifest['offsets']:
                subset = [x for x in rows if (x['split'], x['phase'], x['offset']) == (split, phase, offset)]
                if len(subset) != 5:
                    raise ValueError('Unbalanced phase diagnostic')
                groups.append({'split': split, 'phase': phase, 'offset': offset, 'observations': 5,
                               **{k: float(np.mean([x[k] for x in subset])) for k in
                                  ('measured_opening_mm', 'target_mean8_mm', 'predicted_mean8_mm',
                                   'gripper_mae_mm', 'xyz_sum_rmse_mm')}})
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'predictions.npz', predictions=np.stack(outputs), target=targets)
    report = {'purpose': manifest['purpose'], 'checkpoint': str(args.checkpoint),
              'checkpoint_sha256': hashlib.sha256((args.checkpoint/'model.safetensors').read_bytes()).hexdigest(),
              'actions_published': 0, 'video_sha256': hashes, 'samples': rows, 'groups': groups,
              'inference_seeds': manifest['seeds'], 'independent_held_out_episodes': 5}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'checkpoint': str(args.checkpoint), 'groups': groups}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)
