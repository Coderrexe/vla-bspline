#!/usr/bin/env python3
"""Export a small, traceable public showcase from existing research artifacts.

Run from the repository root with Pillow, NumPy and imageio-ffmpeg installed.
Source videos are real recorded rollouts. The spline illustrations are schematics.
No training code or experiment artifacts are changed.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'website/public'
MEDIA = PUBLIC / 'media'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MEDIA.mkdir(parents=True, exist_ok=True)


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame(path, seconds=0, size=None):
    raw = subprocess.check_output([FFMPEG, '-loglevel', 'error', '-ss', str(seconds), '-i', str(path), '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-threads', '1', '-'])
    image = Image.open(io.BytesIO(raw)).convert('RGB')
    return image.resize(size, Image.Resampling.LANCZOS) if size else image


def font(size, serif=False):
    path = '/System/Library/Fonts/Supplemental/Georgia.ttf' if serif else '/System/Library/Fonts/Supplemental/Arial.ttf'
    return ImageFont.truetype(path, size)


def export_video(name, source, kind, seconds, artifact=None, scene_seed=None):
    path = ROOT / source
    out = MEDIA / f'{name}.mp4'
    if kind == 'human teleoperation':
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(path), '-an', '-c:v', 'libx264', '-crf', '25', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)], check=True)
    else:
        # Remux only; recorded pixels and frame timing are preserved.
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', str(path), '-an', '-c:v', 'copy', '-movflags', '+faststart', str(out)], check=True)
    frame(path, seconds).save(MEDIA / f'{name}.webp', quality=92)
    record = {'file': f'media/{name}.mp4', 'kind': kind, 'source': source, 'source_sha256': sha(path), 'export_sha256': sha(out)}
    if artifact:
        data = load(artifact)
        row = next(e for e in data['episodes'] if e['seed'] == scene_seed)
        record.update({'result_source': artifact, 'result_sha256': sha(ROOT / artifact), 'scene_seed': scene_seed, 'steps': row['steps'], 'success': bool(row['extrema']['success']), 'initial_observation_sha256': row['initial_observation_sha256'], 'instructions': row['instruction_sentences'], 'switches': row['instruction_switches'], 'environment_steps_per_video_second': 20})
    return record


def main():
    evidence = load('outputs/language_clause_exact/locked_3seed_factorial_2327539.json')
    assert evidence['claim_ready']
    language = []
    for key, label in [('A_fixed_clock', 'Waypoint'), ('C_fixed_clock', 'Event spline')]:
        cell = evidence['comparisons'][key]
        language.append({'head': label, 'before': cell['pooled']['original_successes'], 'after': cell['pooled']['clause_successes'], 'n': cell['n_paired_states'], 'gain': 100 * cell['training_seed_mean_effect'], 'seeds': cell['per_seed']})
    interaction = evidence['primary_matched_clock_head_interaction']
    calvin = load('scripts/figures/data/calvin_seeded_final.json')
    rc = load('outputs/robocasa_confirm_n40/official_phase_50scene_stats.json')
    assert rc['aggregate']['original']['successes'] == 4
    assert rc['aggregate']['official']['successes'] == 16
    results = {
        'updated': '2026-09-26',
        'language': {'rows': language, 'interaction_pp': 100 * interaction['mean_interaction'], 'interaction_ci': [100*x for x in interaction['hierarchical_seed_state_bootstrap_95_ci']], 'protocol': 'LIBERO-Long t0/t4. Three training seeds; 100 rollouts per seed and condition. Identical fixed clause schedule at inference. Only training language labels change within each head.', 'source': 'paper/results.md#executable-language-programs-on-libero-long'},
        'calvin': {'native': float(np.mean(calvin['C']['per_seed'])), 'retimed': float(np.mean(calvin['U']['per_seed'])), 'pace': calvin['realized_U'], 'delta': calvin['U-C']['mean'], 'native_seeds': calvin['C']['per_seed'], 'retimed_seeds': calvin['U']['per_seed'], 'source': 'paper/results.md#2-calvin-human-teleop-dead-time-rich--the-flagship'},
        'robocasa': {'rows': [{'seed':1000, 'kettle':[2,6,50], 'rinse':[2,10,50], 'before':4, 'after':16, 'n':100}, {'seed':1001, 'kettle':[2,1,20], 'rinse':[1,8,20], 'before':3, 'after':9, 'n':40}], 'source': 'paper/results.md#robocasa365-target-scene-execution-and-language-steering'},
        'queries': {'rows': [{'suite':'LIBERO Object','before':31.3,'after':11.7,'success_before':93,'success_after':93}, {'suite':'LIBERO Long','before':73.8,'after':33.0,'success_before':61,'success_after':62}], 'n':100, 'source':'docs/RESULTS.md'},
        'evidence_sources': [{'path':p,'sha256':sha(ROOT/p)} for p in ['outputs/language_clause_exact/locked_3seed_factorial_2327539.json','scripts/figures/data/calvin_seeded_final.json','outputs/robocasa_confirm_n40/official_phase_50scene_stats.json','paper/results.md']]
    }
    (PUBLIC/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    videos = []
    for name, src, result, seed, poster in [
        ('kettle', 'kettle_official_success_seed1011', 'confirmK_official_s1000_kettle_official_fixed_target_s1010_n40_2333000', 1011, 10),
        ('kettle-baseline', 'kettle_original_fail_seed1011', 'confirmK_original_s1000_kettle_official_fixed_target_s1010_n40_2332999', 1011, 10),
        ('rinse', 'rinse_official_success_seed1015', 'confirmR_official_s1000_rinse_official_fixed_target_s1010_n40_2332998', 1015, 13),
        ('rinse-baseline', 'rinse_original_fail_seed1015', 'confirmR_original_s1000_rinse_official_fixed_target_s1010_n40_2332997', 1015, 13),
    ]:
        videos.append(export_video(name, f'outputs/video_forensics/robocasa_confirm/{src}.mp4', 'learned policy in simulation', poster, f'outputs/robocasa_confirm_n40/{result}.json', seed))
    for i in (0,2):
        assert videos[i]['initial_observation_sha256'] == videos[i+1]['initial_observation_sha256']
        assert videos[i]['success'] is True and videos[i+1]['success'] is False
    videos.append(export_video('lamp-teleop', 'bc_demo/lamp_assembling/episodes/20260911T214721.850Z-7b6028/video/view_wrist.mp4', 'human teleoperation', 8))
    (PUBLIC/'media-manifest.json').write_text(json.dumps({'videos':videos, 'note':'Simulation recordings save every fifth environment step at 4 fps. Hardware video is a human demonstration, not an autonomous policy success.'},indent=2)+'\n')

    # Animated README: two successful recorded simulation rollouts, 2x their encoded playback.
    images=[]
    streams=[]
    for name in ['kettle','rinse']:
        streams.append(imageio_ffmpeg.read_frames(str(MEDIA/f'{name}.mp4'), pix_fmt='rgb24'))
    meta=[next(s) for s in streams]
    raw_frames=[[Image.frombytes('RGB',m['size'],b) for b in s] for s,m in zip(streams,meta)]
    for index in range(max(map(len,raw_frames))):
        canvas=Image.new('RGB',(640,402),'#f6f5ef'); d=ImageDraw.Draw(canvas)
        d.text((24,17),'ROBOCASA365  /  LEARNED POLICY ROLLOUTS',font=font(15),fill='#314943')
        for col, (frames,name) in enumerate(zip(raw_frames,['Kettle boiling','Rinsing the basin'])):
            x=24+col*308
            canvas.paste(frames[min(index,len(frames)-1)].resize((284,284),Image.Resampling.LANCZOS),(x,51))
            d.text((x,347),name,font=font(19,True),fill='#182d2b')
        d.text((24,379),'Selected successful episodes · scene seeds 1011 / 1015 · 2× recording playback',font=font(12),fill='#5c655f')
        images.append(canvas)
    # GIF delays have 10 ms precision: alternate 120/130 ms to retain 2x playback.
    delays=[120 if i % 2 == 0 else 130 for i in range(len(images))]
    images[0].save(MEDIA/'rollouts.gif',save_all=True,append_images=images[1:],duration=delays,loop=0,optimize=True)
    frame(MEDIA/'kettle.mp4',0).save(MEDIA/'kettle-start.webp',quality=92)

    # A clean GitHub / sharing cover, using actual rollout pixels.
    cover=Image.new('RGB',(1600,840),'#f6f5ef'); d=ImageDraw.Draw(cover)
    d.text((72,49),'V I S I O N   ·   L A N G U A G E   ·   A C T I O N',font=font(20),fill='#54716a')
    d.text((68,108),'VLA B-Spline',font=font(112,True),fill='#173c3a')
    d.text((74,255),'Language in steps. Motion in curves.',font=font(40,True),fill='#45544e')
    d.text((74,323),'Simba Shi  ·  Quinten Jin  ·  Xiatao Sun',font=font(25),fill='#173c3a')
    d.text((74,365),'In collaboration with Yale APOLLO Lab',font=font(22),fill='#657069')
    for i,(metric,caption) in enumerate([('+43.7 pp','Clause execution gain¹'),('1.42×','Realized pace on CALVIN²'),('2.68×','Fewer policy queries³')]):
        x=74+i*298
        d.line((x,520,x+260,520),fill='#ccd2c5',width=2)
        d.text((x,546),metric,font=font(53,True),fill='#173c3a')
        d.text((x,616),caption,font=font(20),fill='#45544e')
    d.text((74,730),'¹ LIBERO-Long t0/t4, 3 seeds   ² vs. native spline   ³ LIBERO Object, 93% success',font=font(18),fill='#626b64')
    for i,t in enumerate([3,10,24]):
        im=frame(MEDIA/'kettle.mp4',t,(204,204))
        cover.paste(im,(1288,54+i*248))
        d.text((1288,265+i*248),['01  PICK','02  PLACE','03  ACTUATE'][i],font=font(15),fill='#36554c')
    d.line((1215,54,1215,766),fill='#ccd2c5',width=2)
    cover.save(MEDIA/'cover.png',optimize=True)
    cover.resize((1200,630),Image.Resampling.LANCZOS).save(MEDIA/'social.jpg',quality=92)

    # Readable static chart for GitHub, generated from the exact same public numbers.
    bg='#f6f5ef'; ink='#173c3a'; muted='#657069'; line='#dde1d7'; orange='#be6b46'
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 460" role="img" aria-label="Clause training gains across three seeds"><rect width="1200" height="460" fill="{bg}"/><g font-family="Arial, sans-serif" fill="{ink}">', '<text x="52" y="48" font-size="23">Executable language improves across three training seeds</text>', f'<text x="52" y="78" font-size="15" fill="{muted}">Gain from clause labels with the same scheduled clause input at inference · LIBERO-Long t0/t4</text>']
    for tick in range(0,51,10):
        y=362-tick*4.4
        svg.append(f'<path d="M88 {y}H1150" stroke="{line}"/><text x="67" y="{y+5}" text-anchor="end" font-size="14" fill="{muted}">{tick}</text>')
    for i,seed in enumerate([1000,1001,1002]):
        for j, row in enumerate(language):
            value=row['seeds'][i]['clause_minus_original']*100
            x=202+i*336+j*84; h=value*4.4; color=orange if j==0 else ink
            svg.append(f'<rect x="{x}" y="{362-h}" width="66" height="{h}" fill="{color}"/><text x="{x+33}" y="{350-h}" text-anchor="middle" font-size="19">+{value:.0f}</text>')
        svg.append(f'<text x="{277+i*336}" y="393" text-anchor="middle" font-size="16">Seed {seed}</text>')
    svg.append(f'<rect x="390" y="422" width="12" height="12" fill="{orange}"/><text x="410" y="434" font-size="14">Waypoint</text><rect x="568" y="422" width="12" height="12" fill="{ink}"/><text x="588" y="434" font-size="14">Event spline</text><text x="1147" y="434" text-anchor="end" font-size="13" fill="{muted}">100 rollouts / seed / condition</text></g></svg>')
    (MEDIA/'clause-results.svg').write_text(''.join(svg))
    print('Exported public results, five videos, posters, README animation, cover and chart.')


if __name__ == '__main__':
    main()
