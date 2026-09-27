"""Read-only inspection of our flushed telemetry and captured RGB frames.

No network, runtime imports, action publisher, or session changes. Intended for
visual supervision alongside the on-site operator, not as a collision detector.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def last_complete(path):
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-131072))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('trial', 'policy', 'video', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    args = p.parse_args()
    if args.output.with_suffix('.json').exists() or args.output.with_suffix('.png').exists():
        raise FileExistsError('Use a new snapshot output')
    telemetry = last_complete(args.trial/'telemetry.jsonl')
    prediction = last_complete(args.policy/'predictions.jsonl')
    refusal = last_complete(args.policy/'refusals.jsonl')
    report = {'read_only': True, 'time_mono': time.monotonic(), 'refusal': refusal}
    if telemetry:
        report.update(telemetry_age_s=time.monotonic()-telemetry['ts'],
                      session=telemetry.get('session'), arms=[])
        for arm in telemetry.get('arms', []):
            report['arms'].append({k:arm.get(k) for k in
                ('arm_id','ee_pose','gripper_open_frac','error_code','stale','recovering')})
    if prediction:
        actions = np.asarray(prediction['actions_grip'])
        report['prediction'] = {k:prediction.get(k) for k in
            ('prediction','publication_requested','gripper_target_path_used_before')}
        report['prediction'].update(net_xyz_mm=(actions[:,:3].sum(0)*1000).tolist(),
            gripper_target_mm=(actions[:,6]*84).tolist())
    canvas = Image.new('RGB', (1280,504), 'white')
    draw = ImageDraw.Draw(canvas)
    report['frames'] = {}
    for column, camera in enumerate(('view_wrist','grip_wrist')):
        frames = sorted((args.video/camera).glob('*.jpg'))
        if len(frames) < 2:
            raise ValueError('Not enough recorded frames for both views')
        frame = frames[-2]  # latest file may still be being written
        age = time.time()-frame.stat().st_mtime
        with Image.open(frame) as im:
            if im.size != (640,480):
                raise ValueError('Unexpected camera resolution')
            canvas.paste(im, (640*column,24))
        draw.text((640*column+5,5), f'{camera} {frame.name}, file age {age:.2f}s', fill='black')
        report['frames'][camera] = {'path':str(frame), 'file_age_s':age}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output.with_suffix('.png'))
    with args.output.with_suffix('.json').open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
