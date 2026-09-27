"""Task-specific Apollo training through LeRobot's standard training loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--task', choices=['cabinet_assembling','drawer_assembling','lamp_assembling'], required=True)
    parser.add_argument('--head', choices=['waypoint','spline'], required=True)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--tag', default='v1')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--verify-initialization', action='store_true')
    parser.add_argument('--dataset-version', default='v1')
    parser.add_argument('--save-every', type=int, default=5000)
    args = parser.parse_args()
    if args.steps <= 0 or args.save_every <= 0:
        parser.error('Training and checkpoint intervals must be positive')
    sys.argv = [sys.argv[0]]  # LeRobot config validation also inspects CLI flags.
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla_apollo.configuration_smolvla_apollo import SmolVLAApolloConfig
    import lerobot.scripts.lerobot_train as training
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this GPU training job')
    base = json.loads((args.root/'manifests/base_initialization.json').read_text())
    dataset = args.root/'datasets'/f'{args.task}_{args.dataset_version}'
    export = json.loads((dataset/'export_manifest.json').read_text())
    output = args.root/'outputs'/f'{args.task}_{args.head}_{args.tag}_{os.environ.get("SLURM_JOB_ID","local")}'
    common = dict(device='cuda', push_to_hub=False, pretrained_path=Path(base['snapshot']),
                  load_vlm_weights=False, n_action_steps=8, optimizer_lr=1e-4,
                  prefix_length=0, pad_language_to='max_length',
                  scheduler_warmup_steps=min(500,max(1,args.steps//10)),
                  scheduler_decay_steps=args.steps, scheduler_decay_lr=1e-5,
                  input_features={
                      'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(32,)),
                      'observation.images.view_wrist':PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640)),
                      'observation.images.grip_wrist':PolicyFeature(type=FeatureType.VISUAL,shape=(3,480,640)),
                  }, output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(16,))})
    if args.head=='waypoint':
        policy = SmolVLAConfig(chunk_size=24, **common)
    else:
        policy = SmolVLAApolloConfig(embedded_spline_stats=json.loads((dataset/'apollo_spline_stats.json').read_text()), **common)
    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id='bc_demo/'+args.task, root=str(dataset),
                                                   eval_split=.09, video_backend='pyav'),
                              policy=policy, output_dir=output, seed=1000,
                              num_workers=6, batch_size=args.batch_size, steps=args.steps,
                              save_freq=min(args.save_every,args.steps), log_freq=100,
                              eval_steps=min(2500,args.steps), max_eval_samples=512,
                              env_eval_freq=0, prefetch_factor=2,
                              wandb=WandBConfig(enable=False))
    record = {'task':args.task,'head':args.head,'output':str(output),'steps':args.steps,
              'dataset':str(dataset),
              'base_initialization':base,'train_episodes':export['train_episodes'],
              'validation_episodes':5,'batch_size':args.batch_size,'seed':1000,
              'gpu':torch.cuda.get_device_name(),'camera_order':list(policy.image_features),
              'source_sha256':{}}
    source = args.root/'source/src/lerobot'
    for relative in ['policies/smolvla_apollo/configuration_smolvla_apollo.py',
                     'policies/smolvla_apollo/modeling_smolvla_apollo.py',
                     'policies/smolvla_spline/modeling_smolvla_spline.py',
                     'policies/smolvla_spline/event_targets.py',
                     'policies/smolvla/modeling_smolvla.py']:
        record['source_sha256'][relative]=hashlib.sha256((source/relative).read_bytes()).hexdigest()
    manifest = args.root/'manifests'/f'{output.name}.json'
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)
    if args.verify_initialization:
        # Check every serialized pretrained tensor before the first update.
        # This also catches an accidentally incompatible model architecture.
        from safetensors import safe_open
        original_make_policy = training.make_policy

        def checked_make_policy(*positional, **keyword):
            model = original_make_policy(*positional, **keyword)
            state = model.state_dict()
            checked = 0
            with safe_open(str(Path(base['snapshot'])/'model.safetensors'), framework='pt', device='cpu') as f:
                for key in f.keys():
                    if key not in state or not torch.equal(state[key].detach().cpu(), f.get_tensor(key)):
                        raise RuntimeError(f'Pretrained initialization mismatch: {key}')
                    checked += 1
            record['initialization_tensor_check'] = {'status':'PASS','tensors':checked}
            manifest.write_text(json.dumps(record,indent=2)+'\n')
            print('PRETRAINED_INITIALIZATION_PASS',checked,flush=True)
            return model

        training.make_policy = checked_make_policy
    training.train(cfg)
    checkpoint=output/'checkpoints/last/pretrained_model'
    if not (checkpoint/'model.safetensors').is_file():
        raise RuntimeError('Training did not produce a complete checkpoint')
    # Annotate each retained checkpoint with its actual update count, so an
    # earlier validation-selected model cannot be mistaken for the final model.
    for saved in sorted((output/'checkpoints').iterdir()):
        if not saved.name.isdigit():
            continue
        target = saved/'pretrained_model'
        if not (target/'model.safetensors').is_file():
            raise RuntimeError(f'Incomplete saved checkpoint: {target}')
        shutil.copy2(dataset/'apollo_manifest.json',target/'apollo_interface.json')
        shutil.copy2(dataset/'export_manifest.json',target/'dataset_provenance.json')
        item = dict(record, requested_training_steps=args.steps, steps=int(saved.name))
        (target/'hardware_training_manifest.json').write_text(json.dumps(item,indent=2)+'\n')
        if (dataset/'apollo_observation_contract.json').is_file():
            shutil.copy2(dataset/'apollo_observation_contract.json',target/'apollo_observation_contract.json')
    print('HARDWARE_TRAINING_COMPLETE',str(checkpoint),flush=True)


if __name__=='__main__': main()
