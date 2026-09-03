"""Label v2 event segments with Molmo 2 (Xiatao T2: granular language).

Two stages, two conda envs (lerobot's transformers pin conflicts with
Molmo 2's processor — same isolation pattern as the CALVIN evaluator):

  STAGE 1 --extract  (lerobot env, CPU ok): partition each episode into
    non-overlapping language-label segments.  At each segment start, the next
    boundary is computed by the SAME fixed-window event kernel used by the
    spline head (raw gripper-value change | pause | episode end | cap).  Dump
    per-segment key frames as PNGs + a segments.jsonl manifest.
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
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

MODEL_ID = "allenai/Molmo2-8B"

PROMPT_VERSION = "molmo2_dual_view_atomic_v2"

SEGMENT_PROMPT = (
    "Context for the whole video: the robot arm is performing the task: '{task}'. "
    "You are shown {n} chronological views spanning ONE short event segment. "
    "Infer the motion between the first and last time points. Describe ONLY the "
    "single atomic action actually underway in this segment, as a 2-8 word "
    "imperative clause. Use only objects named in the task; do not describe a "
    "completed earlier action or a future action. Answer with the clause only."
)

# For the two multi-object LIBERO-Long cells that motivated the language rescue,
# constrain Molmo to a task-grounded atomic phase vocabulary.  The old free-form
# labels hallucinated objects and often described three actions at once.  This
# turns Molmo into a visual phase classifier while retaining natural-language
# training strings and makes every accepted label auditable.
PHASE_BANKS = {
    "put the white mug on the left plate and put the yellow and white mug on the right plate": [
        "move toward the white mug",
        "grasp the white mug",
        "lift the white mug",
        "carry the white mug to the left plate",
        "lower the white mug onto the left plate",
        "release the white mug",
        "move toward the yellow and white mug",
        "grasp the yellow and white mug",
        "lift the yellow and white mug",
        "carry the yellow and white mug to the right plate",
        "lower the yellow and white mug onto the right plate",
        "release the yellow and white mug",
    ],
    "put both the alphabet soup and the cream cheese box in the basket": [
        "move toward the alphabet soup can",
        "grasp the alphabet soup can",
        "lift the alphabet soup can",
        "carry the alphabet soup can to the basket",
        "lower the alphabet soup can into the basket",
        "release the alphabet soup can",
        "move toward the cream cheese box",
        "grasp the cream cheese box",
        "lift the cream cheese box",
        "carry the cream cheese box to the basket",
        "lower the cream cheese box into the basket",
        "release the cream cheese box",
    ],
}

PHASE_PROMPT = (
    "Whole-video task: '{task}'. The chronological external and wrist-camera "
    "views show exactly ONE short event segment. Select the ONE candidate that "
    "best describes the arm motion underway between the first and last time "
    "points. Do not select a completed earlier phase or a future phase.\n"
    "Candidates:\n{candidates}\n"
    "Answer with exactly one candidate ID, such as P03."
)

DATASETS = {
    # root=None -> hub cache (HF_HOME), which is where training already put libero
    "libero": ("HuggingFaceVLA/libero", None, "observation.images.image"),
    "calvin": ("fywang/calvin-task-ABCD-D-lerobot", "~/scratch/vla_bspline/calvin_v30",
               "observation.images.top"),
}


BOUNDARY_VERSION = "event_targets_k_exclusive_window_quantile_v1"


class LanguageSegment(NamedTuple):
    """A half-open episode interval and the event that ended it."""

    start: int
    end: int
    event_type: str


def _load_event_targets_module():
    """Load the dependency-light event module without importing policy models.

    Importing ``policy.smolvla_spline`` executes its package ``__init__``, which
    eagerly imports the full LeRobot/model stack.  Stage 1 only needs the shared
    boundary kernel, so loading the source module directly keeps extraction
    usable in the lightweight CPU environment while retaining one implementation
    of the production semantics.
    """

    module_path = (
        Path(__file__).resolve().parents[2]
        / "policy"
        / "smolvla_spline"
        / "event_targets.py"
    )
    spec = importlib.util.spec_from_file_location("molmo_event_targets", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load production event targets from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EVENT_TARGETS = _load_event_targets_module()


def language_segments(
    actions: np.ndarray,
    horizon_max: int,
    min_seg: int,
    pause_frac: float,
) -> list[LanguageSegment]:
    """Partition an episode using the production boundary kernel.

    This is deliberately a *language-label partition*, not the collection of
    normalization/training windows.  Training constructs an overlapping
    ``horizon_max`` window at every dataset index; a label manifest must instead
    assign every episode action to exactly one half-open interval.  We therefore
    call the production first-event routine repeatedly at the preceding cut.

    Boundary membership is identical to production at those starts: an event at
    local action index ``k`` yields ``T=k``, so the event action is excluded from
    ``[start, start + k)`` and becomes the first action of the next label segment.
    A final remainder shorter than ``min_seg`` is clipped to the real episode end
    rather than inventing label frames for the padded actions used in training.
    """

    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape (T, D), got {actions.shape}")
    if actions.shape[1] < 7:
        raise ValueError(f"expected at least 7 action dimensions, got {actions.shape[1]}")
    if not 1 <= min_seg <= horizon_max:
        raise ValueError(f"min_seg={min_seg} must lie in [1, {horizon_max}]")
    if pause_frac < 0:
        raise ValueError(f"pause_frac must be nonnegative, got {pause_frac}")
    if len(actions) == 0:
        return []

    # Production registers/operates on float32 tensors.  Match that dtype here
    # so threshold edge cases do not silently change in the offline extractor.
    actions = actions.astype(np.float32, copy=False)
    segments: list[LanguageSegment] = []
    start = 0
    while start < len(actions):
        real_length = min(horizon_max, len(actions) - start)
        real_window = actions[start : start + real_length]

        # LeRobot action queries edge-pad short tail windows and mark the padded
        # entries.  Event detection computes its speed quantile on that complete
        # window before padding is zeroed for spline fitting, so reproduce both
        # the values and the mask here.
        if real_length < horizon_max:
            tail = np.repeat(real_window[-1:], horizon_max - real_length, axis=0)
            window = np.concatenate([real_window, tail], axis=0)
        else:
            window = real_window
        pad = torch.zeros(1, horizon_max, dtype=torch.bool)
        pad[:, real_length:] = True
        window_tensor = torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0)

        duration = int(
            _EVENT_TARGETS.first_event_indices(
                window_tensor,
                pad,
                pose_lo=0,
                grip_idx=6,
                min_seg=min_seg,
                horizon_max=horizon_max,
                pause_frac=pause_frac,
            ).item()
        )
        event_type = _EVENT_TARGETS.first_event_types(
            window_tensor,
            pad,
            pose_lo=0,
            grip_idx=6,
            min_seg=min_seg,
            horizon_max=horizon_max,
            pause_frac=pause_frac,
        )[0]
        end = min(start + duration, len(actions))
        if end <= start:  # defensive: min_seg validation should make this impossible
            raise RuntimeError(f"non-progressing language segment at action {start}")
        segments.append(LanguageSegment(start, end, event_type))
        start = end
    return segments


def v2_segments(actions: np.ndarray, horizon_max: int, min_seg: int, pause_frac: float):
    """Backward-compatible pair-only view of :func:`language_segments`."""

    return [
        (segment.start, segment.end)
        for segment in language_segments(actions, horizon_max, min_seg, pause_frac)
    ]


def _validate_boundary_records(path: str) -> None:
    """Refuse to mix legacy and corrected segment artifacts."""

    with open(path) as records:
        for line_number, line in enumerate(records, start=1):
            record = json.loads(line)
            found = record.get("boundary_version", "legacy_or_unspecified")
            if found != BOUNDARY_VERSION:
                raise RuntimeError(
                    f"{path}:{line_number} has boundary_version={found!r}; expected "
                    f"{BOUNDARY_VERSION!r}. Re-extract into a new workdir before labeling."
                )


def _selected_cameras(args, default_camera: str) -> list[str]:
    """Resolve repeated/comma-separated camera arguments without duplicates."""

    raw = args.camera or [default_camera]
    cameras = []
    for item in raw:
        for camera in item.split(","):
            camera = camera.strip()
            if camera and camera not in cameras:
                cameras.append(camera)
    if not cameras:
        raise ValueError("at least one camera is required")
    return cameras


def _phase_prompt(task: str) -> tuple[str, list[str]]:
    if task not in PHASE_BANKS:
        raise ValueError(
            f"no constrained phase bank for task {task!r}; use --label_mode free "
            "or add a reviewed task-specific phase bank"
        )
    candidates = PHASE_BANKS[task]
    listing = "\n".join(f"P{i:02d}: {clause}" for i, clause in enumerate(candidates))
    return PHASE_PROMPT.format(task=task, candidates=listing), candidates


def _parse_phase_id(text: str, candidate_count: int) -> int | None:
    """Parse an exact constrained phase response; never guess from prose."""

    match = re.fullmatch(r"\s*[Pp](\d{1,2})[.\s]*", text)
    if match is None:
        return None
    phase = int(match.group(1))
    return phase if 0 <= phase < candidate_count else None


def stage_extract(args):
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo, root, default_camera = DATASETS[args.dataset]
    cameras = _selected_cameras(args, default_camera)
    ds = LeRobotDataset(repo, root=os.path.expanduser(root) if root else None)
    ep_from = ds.meta.episodes["dataset_from_index"]
    ep_to = ds.meta.episodes["dataset_to_index"]
    frames_dir = os.path.join(args.workdir, "frames")

    n_eps = ds.meta.total_episodes
    eps = range(min(args.episodes, n_eps) if args.episodes else n_eps)
    manifest = os.path.join(args.workdir, "segments.jsonl")
    if os.path.exists(manifest) or os.path.exists(frames_dir):
        raise FileExistsError(
            f"refusing to overwrite extraction artifact in {args.workdir}; "
            "use a new immutable workdir"
        )
    os.makedirs(frames_dir)
    n_seg = 0
    n_selected_eps = 0
    with open(manifest, "w") as f:
        for ep in eps:
            lo, hi = int(ep_from[ep]), int(ep_to[ep])
            ti = int(np.asarray(ds.hf_dataset[lo]["task_index"]))
            task = str(ds.meta.tasks.index[ti]) if hasattr(ds.meta.tasks, "index") else str(ti)
            if args.task_id and ti not in args.task_id:
                continue
            if args.task_filter and args.task_filter not in task:
                continue
            n_selected_eps += 1
            acts = np.stack([np.asarray(ds.hf_dataset[i]["action"]) for i in range(lo, hi)])
            for segment in language_segments(
                acts, args.horizon_max, args.min_seg, args.pause_frac
            ):
                s, e = segment.start, segment.end
                idxs = np.unique(np.linspace(lo + s, lo + max(s, e - 1),
                                             args.frames_per_seg).round().astype(int))
                paths = []
                views = []
                for j, gi in enumerate(idxs):
                    item = ds[int(gi)]
                    for camera_index, camera in enumerate(cameras):
                        img = item[camera]                     # CHW float [0,1]
                        arr = (np.asarray(img).transpose(1, 2, 0) * 255).astype(np.uint8)
                        p = os.path.join(
                            frames_dir,
                            f"ep{ep:05d}_s{s:03d}_t{j}_c{camera_index}.png",
                        )
                        Image.fromarray(arr).save(p)
                        paths.append(p)
                        views.append(
                            {"time_index": int(j), "global_index": int(gi), "camera": camera}
                        )
                f.write(json.dumps({"episode_index": int(ep), "seg_start": int(s),
                                    "seg_end": int(e), "event_type": segment.event_type,
                                    "boundary_version": BOUNDARY_VERSION,
                                    "boundary_config": {
                                        "horizon_max": args.horizon_max,
                                        "min_seg": args.min_seg,
                                        "pause_frac": args.pause_frac,
                                        "pose_lo": 0,
                                        "grip_idx": 6,
                                    },
                                    "task_index": ti, "task": task,
                                    "cameras": cameras, "frames": paths,
                                    "frame_views": views}) + "\n")
                n_seg += 1
    print(
        f"EXTRACTED {n_seg} segments from {n_selected_eps} selected episodes "
        f"using {cameras} -> {manifest}"
    )


def stage_label(args):
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    manifest = os.path.join(args.workdir, "segments.jsonl")
    out_path = os.path.join(args.workdir, "seg_labels.jsonl")
    _validate_boundary_records(manifest)
    if os.path.exists(out_path):
        _validate_boundary_records(out_path)

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

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
            if args.label_mode == "phase_bank":
                text, candidates = _phase_prompt(rec["task"])
            else:
                text = SEGMENT_PROMPT.format(task=rec["task"], n=len(imgs))
                candidates = []
            # Molmo2's processor requires image placeholders via its chat template;
            # this transformers version takes the PIL images inside the content
            content = [{"type": "text", "text": text}]
            views = rec.get("frame_views", [{} for _ in imgs])
            for index, (image, view) in enumerate(zip(imgs, views)):
                description = (
                    f"Time {view.get('time_index', index)}, "
                    f"camera {view.get('camera', 'unspecified')}:"
                )
                content.extend(
                    [{"type": "text", "text": description}, {"type": "image", "image": image}]
                )
            messages = [{"role": "user", "content": content}]

            def generate(messages_to_run, max_new_tokens):
                inputs = processor.apply_chat_template(
                    messages_to_run, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                return processor.decode(
                    generated[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()

            raw_answer = generate(messages, 12 if candidates else 24)
            if candidates:
                phase_id = _parse_phase_id(raw_answer, len(candidates))
                if phase_id is None:
                    retry = messages + [
                        {"role": "assistant", "content": [{"type": "text", "text": raw_answer}]},
                        {"role": "user", "content": [{"type": "text", "text":
                            "Invalid format. Return exactly one candidate ID (P00, P01, ...), nothing else."}]},
                    ]
                    retry_answer = generate(retry, 6)
                    phase_id = _parse_phase_id(retry_answer, len(candidates))
                    if phase_id is None:
                        raise RuntimeError(
                            f"Molmo did not return a valid phase ID for episode "
                            f"{rec['episode_index']} segment {rec['seg_start']}: "
                            f"first={raw_answer!r}, retry={retry_answer!r}"
                        )
                    raw_answer = retry_answer
                ans = candidates[phase_id]
                rec["phase_id"] = f"P{phase_id:02d}"
            else:
                ans = raw_answer.strip().rstrip(".").lower()
            rec["label"] = ans
            rec["raw_model_output"] = raw_answer
            rec["label_mode"] = args.label_mode
            rec["prompt_version"] = PROMPT_VERSION
            rec["model_id"] = MODEL_ID
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
    ap.add_argument("--camera", action="append", default=None,
                    help="camera feature (repeat or comma-separate for multi-view labels)")
    ap.add_argument("--task_id", type=int, action="append", default=None,
                    help="extract only this source task_index (repeatable)")
    ap.add_argument("--task_filter", default=None,
                    help="extract only episodes whose task string contains this substring")
    ap.add_argument("--label_mode", choices=("free", "phase_bank"), default="free")
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
