"""Audit native Apollo demonstrations without modifying them.

Read every Parquet row and decode every video frame. Produce one CSV/JSON
inventory and contact sheets with matched views at six points per episode.
Run locally; on a cluster run under a CPU allocation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit_task(task: Path, out: Path) -> dict:
    manifest = json.loads((task / "manifest.json").read_text())
    rows, all_actions, all_states, all_dt = [], [], [], []
    issues = []
    sheets = out / task.name
    sheets.mkdir(parents=True, exist_ok=True)
    ep_paths = sorted((task / "episodes").glob("*/episode.json"))
    tile_w, tile_h, label_h = 240, 180, 22
    sheet = None
    for ep_i, meta_path in enumerate(ep_paths):
        ep_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        table = pq.read_table(ep_dir / "frames.parquet")
        a = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        s = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        ts = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        wall = np.asarray(table["wallclock_ns"].to_pylist(), dtype=np.int64)
        fi = np.asarray(table["frame_index"].to_pylist())
        dt = np.diff(wall) / 1e9
        n = len(table)
        problems = []
        if a.shape != (n, 16) or s.shape != (n, 32):
            problems.append("unexpected action/state dimensions")
        if not (np.isfinite(a).all() and np.isfinite(s).all()):
            problems.append("nonfinite action/state")
        if not np.array_equal(fi, np.arange(n)):
            problems.append("noncontiguous frame index")
        if not np.allclose(ts, np.arange(n) / manifest["fps"], atol=1e-4):
            problems.append("nonuniform encoded timestamp")
        if np.any(dt <= 0):
            problems.append("nonmonotonic wall clock")
        if n != meta["length"]:
            problems.append("metadata/parquet length mismatch")
        all_actions.append(a)
        all_states.append(s)
        all_dt.append(dt)
        # Illustrative transitions only: analog gripper commands require
        # separate review before deriving executable-clause annotations.
        closed = a[:, 6] < 0.5
        edges = np.flatnonzero(closed[1:] != closed[:-1]) + 1
        sample_indices = np.linspace(0, n - 1, 6, dtype=int)
        if ep_i % 4 == 0:
            sheet = Image.new("RGB", (6 * tile_w, 4 * (2 * (tile_h + label_h) + 28)), "white")
        ybase = (ep_i % 4) * (2 * (tile_h + label_h) + 28)
        draw = ImageDraw.Draw(sheet)
        draw.text((4, ybase + 5), f"{task.name} #{ep_i:02d} {ep_dir.name}  frames={n}", fill="black")
        video_records = {}
        for camera_i, camera in enumerate(["view_wrist", "grip_wrist"]):
            video_path = ep_dir / "video" / f"{camera}.mp4"
            record = {"sha256": digest(video_path)}
            frames = {}
            pts = []
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                record.update(width=stream.width, height=stream.height,
                              average_rate=str(stream.average_rate), header_frames=stream.frames)
                for frame_i, frame in enumerate(container.decode(stream)):
                    pts.append(float(frame.time) if frame.time is not None else None)
                    if frame_i in sample_indices:
                        frames[frame_i] = frame.to_image().resize((tile_w, tile_h))
            record["decoded_frames"] = len(pts)
            record["pts_monotonic"] = all(x is not None and y is not None and y > x for x, y in zip(pts, pts[1:]))
            if len(pts) != n or not record["pts_monotonic"]:
                problems.append(f"{camera}: decoded frames/PTS mismatch")
            video_records[camera] = record
            for column, frame_i in enumerate(sample_indices):
                x = column * tile_w
                y = ybase + 28 + camera_i * (tile_h + label_h)
                if frame_i in frames:
                    sheet.paste(frames[frame_i], (x, y + label_h))
                draw.text((x + 3, y + 4), f"{camera} f{frame_i} {ts[frame_i]:.1f}s", fill="black")
        row = {
            "task": task.name, "episode_index": ep_i, "episode_id": ep_dir.name,
            "frames": n, "encoded_duration_s": n / manifest["fps"],
            "wall_duration_s": (wall[-1] - wall[0]) / 1e9,
            "dt_median_s": float(np.median(dt)), "dt_max_s": float(dt.max()),
            "gap_fraction_gt_60ms": float(np.mean(dt > 0.06)),
            "gripper_edges_at_half": edges.tolist(),
            "gripper_values": np.unique(a[:, 6]).tolist(),
            "initial_grip_xyz": s[0, 9:12].tolist(),
            "final_grip_xyz": s[-1, 9:12].tolist(),
            "initial_rail_m": float(s[0, 8]),
            "rail_action_maxabs": float(np.abs(a[:, [7, 15]]).max()),
            "view_pose_action_maxabs": float(np.abs(a[:, 8:14]).max()),
            "action_source": dict(Counter(map(int, table["action_source"].to_pylist()))),
            "tasks": sorted(set(table["task"].to_pylist())),
            "success_metadata": meta.get("success"),
            "export_ok": meta.get("export_ok"),
            "filter_frames_skipped": meta.get("filter", {}).get("frames_skipped"),
            "parquet_sha256": digest(ep_dir / "frames.parquet"),
            "metadata_sha256": digest(meta_path), "videos": video_records,
            "issues": problems,
        }
        rows.append(row)
        issues.extend(f"{ep_dir.name}: {problem}" for problem in problems)
        if ep_i % 4 == 3 or ep_i == len(ep_paths) - 1:
            sheet.save(sheets / f"contact_{ep_i // 4:02d}.jpg", quality=93)
        print(json.dumps({"task": task.name, "episode": ep_i, "frames": n, "issues": problems}), flush=True)
    actions, states, dts = np.concatenate(all_actions), np.concatenate(all_states), np.concatenate(all_dt)
    def statistics(values, names):
        return {name: {"min": float(values[:, i].min()), "max": float(values[:, i].max()),
                       "mean": float(values[:, i].mean()), "std": float(values[:, i].std()),
                       "abs_q99": float(np.quantile(np.abs(values[:, i]), .99))}
                for i, name in enumerate(names)}
    result = {
        "task": task.name, "episode_count": len(rows), "frame_count": len(actions),
        "manifest_episode_count": manifest["episodes"], "manifest_frame_count": manifest["frames"],
        "fps": manifest["fps"], "issues": issues, "episodes": rows,
        "action_statistics": statistics(actions, manifest["features"]["action"]["names"]),
        "state_statistics": statistics(states, manifest["features"]["observation.state"]["names"]),
        "wall_dt_quantiles_s": dict(zip(["min", "q50", "q90", "q99", "max"],
                                        map(float, np.quantile(dts, [0, .5, .9, .99, 1])))),
    }
    (out / f"{task.name}_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame([{k:v for k,v in r.items() if k not in ["videos", "gripper_values"]} for r in rows]).to_csv(out / f"{task.name}_episodes.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("bc_demo"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for task in sorted(args.root.iterdir()):
        if (task / "manifest.json").is_file():
            audit_task(task, args.out)


if __name__ == "__main__":
    main()
