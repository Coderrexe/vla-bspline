"""Extract frame strips from RoboCasa eval videos so failures can be READ,
not guessed (Xiatao: "genuinely investigate the videos"). For each requested
eval run: pick success + failure episodes, sample K frames each, tile into one
PNG strip per episode. Output goes to a local-syncable dir.

  python rc_watch.py --runs rc_eval_A_TurnOnElectricKettle rc_eval_C_TurnOnElectricKettle \
      --n_succ 2 --n_fail 3 --frames 8 --out ~/scratch/vla_bspline/outputs/watch
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np


def frames_of(video, k):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video],
                       capture_output=True, text=True, timeout=120)
    n = int(r.stdout.strip())
    idxs = np.linspace(0, n - 1, k).round().astype(int)
    sel = "+".join(f"eq(n\\,{i})" for i in idxs)
    out = video + ".strip.png"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video,
                    "-vf", f"select='{sel}',tile={k}x1", "-vsync", "0", out],
                   check=True, timeout=300)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--base", default="~/scratch/vla_bspline/outputs")
    ap.add_argument("--n_succ", type=int, default=2)
    ap.add_argument("--n_fail", type=int, default=3)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = os.path.expanduser(args.base)
    outd = os.path.expanduser(args.out)
    os.makedirs(outd, exist_ok=True)

    for run in args.runs:
        info = json.load(open(os.path.join(base, run, "eval_info.json")))
        succ = info["per_task"][0]["metrics"]["successes"]
        vids = sorted(glob.glob(os.path.join(base, run, "videos", "*", "eval_episode_*.mp4")),
                      key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        picks = []
        for i, v in enumerate(vids):
            ok = succ[i] if i < len(succ) else None
            picks.append((i, v, ok))
        chosen = [p for p in picks if p[2]][: args.n_succ] + \
                 [p for p in picks if p[2] is False][: args.n_fail]
        for i, v, ok in chosen:
            strip, n = frames_of(v, args.frames)
            tag = "succ" if ok else "fail"
            dst = os.path.join(outd, f"{run}_ep{i:02d}_{tag}_{n}steps.png")
            os.replace(strip, dst)
            print("wrote", dst)


if __name__ == "__main__":
    main()
