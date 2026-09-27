"""Create new offline evaluation views of immutable intermediate checkpoints.

Weights/processors/configs link to their original checkpoint. Dataset interface
metadata comes from the final checkpoint of that same run. Does not overwrite
source files, change model tensors, or authorize hardware motion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',required=True)
    p.add_argument('--steps',type=int,nargs='+',default=[5000,10000,15000,20000])
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    final=args.run/'checkpoints/last/pretrained_model'
    metadata=['apollo_interface.json','dataset_provenance.json']
    for name in metadata:
        if not (final/name).is_file():
            raise FileNotFoundError(final/name)
    args.output.mkdir(parents=True)
    records=[]
    for step in args.steps:
        source=(args.run/f'checkpoints/{step:06d}/pretrained_model').resolve()
        if not (source/'model.safetensors').is_file():
            raise FileNotFoundError(source)
        target=args.output/f'{args.prefix}_{step:06d}'
        target.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in metadata+['hardware_training_manifest.json']:
                os.symlink(path,target/path.name)
        for name in metadata:
            shutil.copy2(final/name,target/name)
        record=json.loads((final/'hardware_training_manifest.json').read_text())
        record.update(steps=step,requested_training_steps=record['steps'],
                      immutable_source_checkpoint=str(source),offline_evaluation_view=True)
        (target/'hardware_training_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
        digest=hashlib.sha256()
        with (source/'model.safetensors').open('rb') as f:
            for block in iter(lambda:f.read(4<<20),b''):
                digest.update(block)
        records.append({'checkpoint':str(target),'source':str(source),'steps':step,'model_sha256':digest.hexdigest()})
    (args.output/'views.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
