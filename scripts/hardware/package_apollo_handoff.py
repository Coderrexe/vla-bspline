"""Collect only completed, verified hardware policies into a portable directory."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--run', action='append', required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'verified_models': [], 'not_ready': []}
    for name in args.run:
        run = args.root / 'outputs' / name
        validation = run / 'offline_validation.json'
        checkpoint = run / 'checkpoints/last/pretrained_model'
        if not validation.is_file() or not (checkpoint / 'hardware_training_manifest.json').is_file():
            report['not_ready'].append({'run': name, 'reason': 'Training or prediction verification did not finish'})
            continue
        result = json.loads(validation.read_text())
        training = json.loads((checkpoint / 'hardware_training_manifest.json').read_text())
        if result.get('load_predict') != 'PASS' or training.get('initialization_tensor_check', {}).get('status') != 'PASS':
            report['not_ready'].append({'run': name, 'reason': 'Verification is not PASS'})
            continue
        target = args.output / 'models' / f"{training['task']}_{training['head']}"
        target.mkdir(parents=True)
        for source in checkpoint.iterdir():
            if source.is_file():
                # Checkpoints are complete and will not be modified. Hard links
                # avoid doubling storage on scratch; rsync/tar copies real bytes.
                os.link(source.resolve(), target / source.name)
        shutil.copy2(validation, target / 'offline_validation.json')
        if validation.with_suffix('.npz').exists():
            shutil.copy2(validation.with_suffix('.npz'), target / 'offline_validation.npz')
        first_episode = sorted((args.root / 'raw' / training['task'] / 'episodes').glob('*/episode.json'))[0]
        episode_metadata = json.loads(first_episode.read_text())
        reference = {'source_episode_id': first_episode.parent.name,
                     'reference_only_not_an_execution_command': True,
                     **{k: episode_metadata[k] for k in ['profile_snapshot', 'arm_bases', 'extrinsics', 'session_id']}}
        (target / 'apollo_collection_reference.json').write_text(json.dumps(reference, indent=2)+'\n')
        # A recorded observation and its already-verified prediction let the
        # mentor test the input pipeline without connecting to a physical robot.
        sample = result['samples'][0]
        episode = args.root / 'raw' / training['task'] / 'episodes' / sample['source_episode_id']
        state_table = pq.read_table(episode / 'frames.parquet', columns=['observation.state'])
        fixture = {'state': np.asarray(state_table['observation.state'][sample['frame']].as_py(), dtype=np.float32),
                   'seed': np.asarray(result['seed'], dtype=np.int64),
                   'source_episode_id': np.asarray(sample['source_episode_id']),
                   'source_frame': np.asarray(sample['frame'], dtype=np.int64)}
        for camera in ['view_wrist', 'grip_wrist']:
            with av.open(str(episode / 'video' / f'{camera}.mp4')) as video:
                for index, frame in enumerate(video.decode(video=0)):
                    if index == sample['frame']:
                        fixture[camera] = frame.to_ndarray(format='rgb24')
                        break
            if camera not in fixture:
                raise RuntimeError(f'Missing fixture frame: {episode}, {camera}')
        with np.load(validation.with_suffix('.npz')) as values:
            fixture['expected_action'] = values['predicted'][0]
        np.savez_compressed(target / 'prediction_fixture.npz', **fixture)
        report['verified_models'].append({
            'run': name, 'relative_path': str(target.relative_to(args.output)),
            'model_sha256': digest(target / 'model.safetensors'),
            'offline_metrics': result['policy'], 'training_steps': training['steps'],
            'task': training['task'], 'head': training['head'],
        })
    for name in ['apollo_predictor.py', 'verify_apollo_checkpoint.py']:
        shutil.copy2(args.root / 'scripts' / name, args.output / name)
    for name in ['training_source_v1.tar.gz', 'base_initialization.json', 'HARDWARE_TRAINING_2026-09-12.md']:
        shutil.copy2(args.root / 'manifests' / name, args.output / name)
    # Cache the already-used VLM config/tokenizer files, not its pretrained
    # weights. The complete fine-tuned weights are in the policy checkpoints.
    hf = Path(os.environ['HF_HOME']) / 'hub'
    repo = 'models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct'
    revision = (hf / repo / 'refs/main').read_text().strip()
    report['vlm_config_tokenizer_revision'] = revision
    cache = args.output / 'hf_cache/hub' / repo
    (cache / 'refs').mkdir(parents=True)
    shutil.copy2(hf / repo / 'refs/main', cache / 'refs/main')
    snapshot = hf / repo / 'snapshots' / revision
    for source in snapshot.rglob('*'):
        if not source.is_file() or source.suffix in {'.safetensors', '.bin', '.pt', '.pth'}:
            continue
        target = cache / 'snapshots' / revision / source.relative_to(snapshot)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    report['status'] = 'OFFLINE_VERIFIED' if not report['not_ready'] else 'PARTIAL'
    report['purpose'] = ('integration_smoke' if any(m['training_steps'] < 5000 for m in report['verified_models'])
                         else 'post_training_handoff')
    report['physical_robot_evaluation'] = 'Not yet performed; mentor must validate the command interface and safety'
    (args.output / 'handoff_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    readme = '''# Apollo hardware policy handoff

See `handoff_manifest.json` for the models that actually finished training and
offline verification. No closed-loop robot success is claimed here.
If its purpose is `integration_smoke`, the bundle is only an interface test:
do not use those minimally trained weights for physical execution.

Unpack `training_source_v1.tar.gz` into a separate `source` directory. Use a
separate compatible Python environment (versions in `base_initialization.json`)
and put `source/src` and this handoff folder on `PYTHONPATH`. Do not change the
working Apollo robot environment to match the cluster. A prediction-only service
in a separate process can bridge them.

The included small `hf_cache` carries the VLM config and tokenizer used by the
model. From this handoff directory, after activating the separate compatible
inference environment:

```bash
mkdir source  # first setup only; do not overwrite an existing source directory
tar -xzf training_source_v1.tar.gz -C source
export APOLLO_HANDOFF="$PWD"
export PYTHONPATH="$APOLLO_HANDOFF/source/src:$APOLLO_HANDOFF${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$APOLLO_HANDOFF/hf_cache"
export HF_HUB_OFFLINE=1
```

The checkpoint itself contains all model weights. The archived source and small
cache avoid dependence on the training dataset or mutable remote model configs.

```python
from apollo_predictor import ApolloPredictor
model = ApolloPredictor("models/cabinet_assembling_spline", device="cuda")
model.reset()
actions = model.predict_chunk(state, view_rgb, grip_rgb)
```

Input state uses the exact 32-field ordering in `apollo_interface.json`; both
images are RGB uint8 HWC 480x640. The output is at most eight raw 16-field Apollo
commands. Nominal control rate is 25 Hz. The adapter never connects to a robot.
Keep the camera arm and rails parked as during demonstration collection.

Before enabling movement, the mentor must verify translation units, rotation
composition, camera order, control timing, reset/queue behavior, and robot-side
safety limits. Do not repeat stale deltas if inference stalls. Consult the
training document for the one quarantined cabinet telemetry episode.
Use the collection-time Apollo action-application path: the rotation convention
cannot be safely reconstructed from the labels alone. The included
`apollo_collection_reference.json` records initial profiles and camera/arm
references; it is metadata, not a request to move to those joint positions.

Each model also includes `prediction_fixture.npz`: one actual recorded state,
the two RGB images, the random seed, and the previously verified model output.
Use it to test the new inference environment before connecting to the robot:

```python
import numpy as np
import torch
fixture = np.load("models/cabinet_assembling_spline/prediction_fixture.npz")
model.reset()
torch.manual_seed(int(fixture["seed"]))
prediction = model.predict_chunk(fixture["state"], fixture["view_wrist"], fixture["grip_wrist"])
print("maximum difference from reference:", np.max(np.abs(prediction - fixture["expected_action"])))
```

Small numerical differences across GPU/software versions are possible. This is
an offline integration fixture, not a physical safety or task-success test.
'''
    (args.output / 'README.md').write_text(readme)
    print(json.dumps(report, indent=2), flush=True)
    if report['not_ready']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
