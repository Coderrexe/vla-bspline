"""Label v2 event segments with Molmo 2 (Xiatao T2: granular language).

Two stages, two conda envs (lerobot's transformers pin conflicts with
Molmo 2's processor — same isolation pattern as the CALVIN evaluator):

  STAGE 1 --extract  (lerobot env, CPU ok): reproduce the SAME event
    segmentation the spline head trains on (gripper toggle | pause | cap),
    dump per-segment key frames as PNGs + a segments.jsonl manifest.
  STAGE 2 --label    (molmo env, GPU): load Molmo2-8B, caption each segment
    from its frames + a video-level context prompt built from the episode's
    task annotation (Xiatao's anti-hallucination guidance: name the task and
    objects once at video level), append labels -> seg_labels.jsonl.

  # stage 1 (lerobot env)
  python molmo_segment_labels.py --extract --dataset libero --episodes 2 \
      --workdir ~/scratch/vla_bspline/molmo_libero
  # stage 2 (molmo env)
  python molmo_segment_labels.py --label --workdir ~/scratch/vla_bspline/molmo_libero --smoke

The later dataset-build step swaps per-segment labels into training task
strings ("C-lang" arm).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

MODEL_ID = "allenai/Molmo2-8B"

SEGMENT_PROMPT = (
    "Context for the whole video: the robot arm is performing the task: '{task}'. "
    "You are shown {n} frames spanning ONE short segment of that video. "
    "Describe ONLY what the arm does in this segment, as a single short "
    "imperative clause (max 10 words), e.g. 'move toward the red block', "
    "'close the gripper on the bowl', 'lift and carry it left'. "
    "Do not mention objects that are not visible. Answer with the clause only."
)

DATASETS = {
    # root=None -> hub cache (HF_HOME), which is where training already put libero
    "libero": ("HuggingFaceVLA/libero", None, "observation.images.image"),
    "calvin": ("fywang/calvin-task-ABCD-D-lerobot", "~/scratch/vla_bspline/calvin_v30",
               "observation.images.top"),
}


def v2_segments(actions: np.ndarray, horizon_max: int, min_seg: int, pause_frac: float):
    """Mirror of the head's event segmentation: walk the episode, cut at the
    first gripper flip or sustained pause after min_seg, else at horizon_max."""
    grip = actions[:, 6]
    speed = np.linalg.norm(actions[:, :6], axis=1)
    thr = pause_frac * np.median(speed[speed > 1e-8]) if (speed > 1e-8).any() else 0.0
    segs, start, L = [], 0, len(actions)
    while start < L:
        end = min(start + horizon_max, L)
        cut = end
        for k in range(start + min_seg, end):
            if k > 0 and np.sign(grip[k]) != np.sign(grip[k - 1]):
                cut = k + 1
                break
            if thr > 0 and k + 1 < L and speed[k] < thr and speed[k + 1] < thr:
                cut = k + 1
                break
        segs.append((start, cut))
        start = cut
    return segs


def stage_extract(args):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo, root, cam = DATASETS[args.dataset]
    cam = args.camera or cam
    ds = LeRobotDataset(repo, root=os.path.expanduser(root) if root else None)
    ep_from = ds.meta.episodes["dataset_from_index"]
    ep_to = ds.meta.episodes["dataset_to_index"]
    frames_dir = os.path.join(args.workdir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    n_eps = ds.meta.total_episodes
    eps = range(min(args.episodes, n_eps) if args.episodes else n_eps)
    manifest = os.path.join(args.workdir, "segments.jsonl")
    n_seg = 0
    with open(manifest, "w") as f:
        for ep in eps:
            lo, hi = int(ep_from[ep]), int(ep_to[ep])
            ti = int(np.asarray(ds.hf_dataset[lo]["task_index"]))
            task = str(ds.meta.tasks.index[ti]) if hasattr(ds.meta.tasks, "index") else str(ti)
            if args.task_filter and args.task_filter not in task:
                continue
            acts = np.stack([np.asarray(ds.hf_dataset[i]["action"]) for i in range(lo, hi)])
            for (s, e) in v2_segments(acts, args.horizon_max, args.min_seg, args.pause_frac):
                idxs = np.unique(np.linspace(lo + s, lo + max(s, e - 1),
                                             args.frames_per_seg).round().astype(int))
                paths = []
                for j, gi in enumerate(idxs):
                    img = ds[int(gi)][cam]                     # CHW float [0,1]
                    arr = (np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8)
                    p = os.path.join(frames_dir, f"ep{ep:05d}_s{s:03d}_{j}.png")
                    Image.fromarray(arr).save(p)
                    paths.append(p)
                f.write(json.dumps({"episode_index": int(ep), "seg_start": int(s),
                                    "seg_end": int(e), "task": task, "frames": paths}) + "\n")
                n_seg += 1
    print(f"EXTRACTED {n_seg} segments from {len(list(eps))} episodes -> {manifest}")


def stage_label(args):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    manifest = os.path.join(args.workdir, "segments.jsonl")
    out_path = os.path.join(args.workdir, "seg_labels.jsonl")
    done = set()
    if os.path.exists(out_path):                # resumable
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["episode_index"], r["seg_start"]))
    n = 0
    with open(out_path, "a") as out:
        for line in open(manifest):
            rec = json.loads(line)
            if (rec["episode_index"], rec["seg_start"]) in done:
                continue
            imgs = [Image.open(p).convert("RGB") for p in rec["frames"]]
            text = SEGMENT_PROMPT.format(task=rec["task"], n=len(imgs))
            # Molmo2's processor requires image placeholders via its chat template;
            # this transformers version takes the PIL images inside the content
            messages = [{"role": "user",
                         "content": [{"type": "image", "image": im} for im in imgs] +
                                    [{"type": "text", "text": text}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            ans = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip().rstrip(".").lower()
            rec["label"] = ans
            rec.pop("frames")
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            if args.smoke:
                print(f"ep {rec['episode_index']} [{rec['seg_start']:3d},{rec['seg_end']:3d}) -> {ans}",
                      flush=True)
    print(f"LABELED {n} new segments -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--dataset", choices=list(DATASETS), default="libero")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--horizon_max", type=int, default=24)
    ap.add_argument("--min_seg", type=int, default=8)
    ap.add_argument("--pause_frac", type=float, default=0.15)
    ap.add_argument("--frames_per_seg", type=int, default=3)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--task_filter", default=None,
                    help="extract only episodes whose task string contains this substring")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.workdir = os.path.expanduser(args.workdir)
    os.makedirs(args.workdir, exist_ok=True)

    if args.extract:
        stage_extract(args)
    if args.label:
        stage_label(args)
    if not (args.extract or args.label):
        raise SystemExit("pass --extract and/or --label")


if __name__ == "__main__":
    main()
