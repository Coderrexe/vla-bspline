"""Offline first-grasp diagnostics. No Dora, robot API, or command publisher.

Prepare original legacy state/action samples locally; evaluate against the same
episode videos on the lab GPU. Held-out and training episodes remain separate.
Teacher-forced predictions are NOT closed-loop hardware success measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

OFFSETS = (-32, -16, -8, 0, 8, 24)
SEEDS = (20260912, 20260913, 20260914)


def prepare(args):
    import pyarrow.parquet as pq
    provenance = json.loads(args.provenance.read_text())
    if provenance['task'] != 'Drawer Assembling':
        raise ValueError('This diagnostic is for drawer assembling')
    selected = []
    for split in ('validation', 'train'):
        entries = [x for x in provenance['episode_mapping'] if x['split'] == split]
        selected.extend(entries[:5])
    if len(selected) != 10:
        raise ValueError('Expected five held-out and five training episodes')
    args.output.mkdir(parents=True, exist_ok=False)
    samples, states, targets = [], [], []
    for entry in selected:
        episode = entry['source_episode_id']
        path = args.dataset/'episodes'/episode/'frames.parquet'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['source_parquet_sha256']:
            raise ValueError(f'Not the original training recording: {episode}')
        table = pq.read_table(path, columns=['observation.state', 'action', 'timestamp'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
        if action.shape != (len(state), 16) or state.shape[1:] != (32,):
            raise ValueError('Unexpected original feature schema')
        candidates = np.flatnonzero(action[:, 6] < .5)
        if not len(candidates):
            raise ValueError(f'No first half-open target: {episode}')
        boundary = int(candidates[0])
        for offset in OFFSETS:
            frame = boundary+offset
            if frame < 0 or frame+8 > len(state):
                raise ValueError('Phase window outside episode')
            samples.append({'episode': episode, 'split': entry['split'],
                'frame': frame, 'timestamp_s': float(table['timestamp'][frame].as_py()),
                'first_half_open_frame': boundary, 'offset': offset,
                'original_parquet_sha256': digest})
            states.append(state[frame]); targets.append(action[frame:frame+8])
    np.savez_compressed(args.output/'inputs.npz', states=np.stack(states), targets=np.stack(targets))
    (args.output/'manifest.json').write_text(json.dumps({
        'purpose': 'OFFLINE_TEACHER_FORCED_GRASP_TRANSITION_NOT_TASK_SUCCESS',
        'sample_selection': 'first target below half-open; six fixed offsets; first five episodes per split in saved provenance',
        'offsets': OFFSETS, 'seeds': SEEDS, 'samples': samples}, indent=2)+'\n')
    print(json.dumps({'samples': len(samples), 'episodes': len(selected), 'actions_published': 0}))


def video_frames(path, indices):
    expression = 'select='+ '+'.join(f'eq(n\\,{j})' for j in indices)
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-threads', '1',
        '-i', str(path), '-vf', expression, '-vsync', '0', '-frames:v', str(len(indices)),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], check=True, capture_output=True)
    expected = len(indices)*480*640*3
    if len(result.stdout) != expected:
        raise ValueError(f'Video frame-count/shape mismatch: {path}')
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, 480, 640, 3).copy()


def evaluate(args):
    import gc
    import torch
    from apollo_predictor import ApolloPredictor
    from apollo_legacy_state import CheckpointStateAdapter, align_quaternion_hemisphere
    manifest = json.loads((args.inputs/'manifest.json').read_text())
    with np.load(args.inputs/'inputs.npz', allow_pickle=False) as data:
        states, targets = data['states'], data['targets']
    samples = manifest['samples']
    if (states.shape != (60, 32) or targets.shape != (60, 8, 16)
            or len(samples) != 60 or not np.isfinite(states).all()
            or not np.isfinite(targets).all()):
        raise ValueError('Prepared diagnostic input mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    images, video_hashes = {}, {}
    for episode in sorted({x['episode'] for x in samples}):
        entries = [(i, s['frame']) for i, s in enumerate(samples) if s['episode'] == episode]
        indices = [j for _, j in entries]
        if indices != sorted(set(indices)):
            raise ValueError('Expected unique ordered frame indices')
        for camera in ('view_wrist', 'grip_wrist'):
            path = args.videos/'episodes'/episode/'video'/f'{camera}.mp4'
            video_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            frames = video_frames(path, indices)
            for (i, _), frame in zip(entries, frames):
                images[i, camera] = frame
    report = {'purpose': manifest['purpose'], 'actions_published': 0,
              'runtime_api_writes': 0, 'video_sha256': video_hashes, 'models': []}
    arrays = {}
    torch.set_num_threads(4)
    for checkpoint in args.models:
        provenance = json.loads((checkpoint/'dataset_provenance.json').read_text())
        entries = {x['source_episode_id']: x for x in provenance['episode_mapping']}
        if provenance['task'] != 'Drawer Assembling' or any(
                s['episode'] not in entries or entries[s['episode']]['split'] != s['split']
                or entries[s['episode']]['source_parquet_sha256'] != s['original_parquet_sha256']
                for s in samples):
            raise ValueError('Checkpoint training/validation provenance does not match the diagnostic')
        model = ApolloPredictor(checkpoint, device='cuda:0')
        adapter = CheckpointStateAdapter(checkpoint)
        rows, outputs = [], []
        for i, sample in enumerate(samples):
            # These are original legacy features, NOT current TCP observations.
            state = align_quaternion_hemisphere(
                adapter.stabilize_training_state(states[i]), adapter.mean)
            predictions = []
            for seed in manifest['seeds']:
                model.reset(); torch.manual_seed(seed)
                predictions.append(model.predict_chunk(state, images[i, 'view_wrist'], images[i, 'grip_wrist']))
            predictions = np.stack(predictions)
            outputs.append(predictions)
            target = targets[i]
            rows.append({**sample, 'measured_opening_mm': float(states[i, 7]*84),
                'target_first_mm': float(target[0, 6]*84),
                'target_mean8_mm': float(target[:, 6].mean()*84),
                'predicted_first_mm': (predictions[:, 0, 6]*84).tolist(),
                'predicted_mean8_mm': (predictions[:, :, 6].mean(1)*84).tolist(),
                'gripper_mae_mm': float(np.abs(predictions[:, :, 6]-target[:, 6]).mean()*84),
                'xyz_sum_rmse_mm': float(np.sqrt(np.mean(
                    (predictions[:, :, :3].sum(1)-target[:, :3].sum(0))**2))*1000)})
        arrays[checkpoint.name] = np.stack(outputs)
        groups = []
        for split in ('validation', 'train'):
            for offset in manifest['offsets']:
                subset = [x for x in rows if x['split'] == split and x['offset'] == offset]
                groups.append({'split': split, 'offset': offset, 'observations': len(subset),
                    **{k: float(np.mean([x[k] for x in subset])) for k in (
                        'measured_opening_mm', 'target_mean8_mm', 'predicted_mean8_mm',
                        'gripper_mae_mm', 'xyz_sum_rmse_mm')}})
        item = {'model': checkpoint.name, 'samples': rows, 'groups': groups}
        report['models'].append(item)
        (args.output/f'{checkpoint.name}.json').write_text(json.dumps(item, indent=2)+'\n')
        print(json.dumps({'model': checkpoint.name, 'groups': groups}), flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    np.savez_compressed(args.output/'predictions.npz', **arrays, target=targets)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--provenance', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('evaluate')
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--models', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
