"""Render original recorded frames with offline prediction annotations only."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    samples = sorted([s for s in report['samples'] if s['episode'] == args.episode], key=lambda x: x['frame'])
    if not samples or len(samples) > 12:
        raise ValueError('Expected one episode and at most twelve samples')
    needed = {s['frame'] for s in samples}
    images = {}
    for camera in ('view_wrist', 'grip_wrist'):
        with av.open(str(args.dataset/'episodes'/args.episode/'video'/f'{camera}.mp4')) as video:
            for i, frame in enumerate(video.decode(video=0)):
                if i in needed:
                    images[i, camera] = frame.to_image()
                if i >= max(needed):
                    break
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 23)
    canvas = Image.new('RGB', (1280, 90+len(samples)*545), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), f'{args.episode} | {report.get("model", "checkpoint")}', font=font, fill='black')
    draw.text((16, 42), 'Recorded demonstrations + offline predictions; NOT a learned robot rollout', font=font, fill='black')
    for i, sample in enumerate(samples):
        top = 90+i*545
        predicted = np.asarray(sample['predicted_mean8_mm'])
        label = (f'frame {sample["frame"]} / offset {sample["offset"]:+d} | '
                 f'measured {sample["measured_opening_mm"]:.1f} mm | '
                 f'target {sample["target_mean8_mm"]:.1f} mm | '
                 f'predicted {predicted.mean():.1f} mm (8-row means)')
        draw.text((16, top+4), label, font=font, fill='black')
        for j, camera in enumerate(('view_wrist', 'grip_wrist')):
            canvas.paste(images[sample['frame'], camera].resize((640, 480)), (j*640, top+35))
            draw.text((j*640+16, top+517), camera, font=font, fill='black')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    canvas.save(args.output)


if __name__ == '__main__':
    main()
