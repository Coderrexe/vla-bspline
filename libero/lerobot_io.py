"""
Lightweight reader for the HuggingFaceVLA/libero LeRobot v3.0 dataset.

We avoid installing the full `lerobot` package locally: for B-spline fitting we
only need the NUMERIC columns (observation.state, action, timestamp). Images live
in separate video files (~500 MB) that we never touch on the laptop. Only the data
parquet (~100 MB total) is downloaded, cached in the standard HuggingFace cache
(~/.cache/huggingface), NOT in this OneDrive folder.

NOTE on format: the episodes-meta `data/file_index` does NOT map 1:1 to the actual
parquet filenames, and `dataset_from/to_index` are GLOBAL row indices. So instead of
trusting that mapping we download all data parquet once and index episodes directly.

State layout (8-dim):  [0:3] eef xyz | [3:6] axis-angle | [6:8] gripper finger qpos
Action layout (7-dim): [0:6] delta eef pose (OSC_POSE, in [-1,1]) | [6] gripper (+/-1)
"""
from __future__ import annotations

import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "HuggingFaceVLA/libero"
REPO_TYPE = "dataset"

STATE_POS = slice(0, 3)
STATE_ROT = slice(3, 6)
STATE_GRIP = slice(6, 8)
ACT_DPOSE = slice(0, 6)
ACT_GRIP = 6

NUMERIC_COLS = ["episode_index", "frame_index", "index", "timestamp",
                "observation.state", "action", "task_index"]


@lru_cache(maxsize=1)
def local_root() -> str:
    """Download (once) all numeric data + metadata; return local snapshot dir."""
    return snapshot_download(
        REPO_ID, repo_type=REPO_TYPE,
        allow_patterns=["data/**/*.parquet", "meta/**/*.parquet", "meta/*.json"],
    )


@lru_cache(maxsize=1)
def load_info() -> dict:
    return json.load(open(os.path.join(local_root(), "meta", "info.json")))


@lru_cache(maxsize=1)
def load_tasks() -> pd.DataFrame:
    return pd.read_parquet(os.path.join(local_root(), "meta", "tasks.parquet"))


@lru_cache(maxsize=1)
def load_episodes_meta() -> pd.DataFrame:
    root = local_root()
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"), recursive=True))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


@lru_cache(maxsize=1)
def _all_numeric() -> pd.DataFrame:
    """Concatenate the numeric columns of every data parquet, globally sorted."""
    root = local_root()
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    dfs = [pd.read_parquet(f, columns=NUMERIC_COLS) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


def episode_indices() -> np.ndarray:
    return np.sort(_all_numeric()["episode_index"].unique())


def episode_arrays(episode_index: int):
    """Return (t, state, action) for one episode, ordered by frame.
    t: (L,) seconds; state: (L, 8); action: (L, 7)."""
    df = _all_numeric()
    ep = df.loc[df["episode_index"] == episode_index].sort_values("frame_index")
    if len(ep) == 0:
        raise KeyError(f"episode {episode_index} not found")
    state = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
    action = np.stack(ep["action"].to_numpy()).astype(np.float64)
    t = np.asarray(ep["timestamp"].to_numpy(), dtype=np.float64).reshape(-1)
    return t, state, action


if __name__ == "__main__":
    info = load_info()
    print("codebase", info["codebase_version"], "| fps", info["fps"],
          "| episodes", info["total_episodes"], "| frames", info["total_frames"],
          "| tasks", info["total_tasks"])
    eps = episode_indices()
    print("indexed episodes:", len(eps), "| first/last:", eps[0], eps[-1])
    t, s, a = episode_arrays(0)
    print("episode 0: L =", len(t), "| dur =", round(float(t[-1]), 2), "s |",
          "state", s.shape, "action", a.shape)
