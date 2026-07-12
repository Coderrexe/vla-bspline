"""CALVIN episode IO — mirrors lerobot_io's episode_arrays() interface for the
LeRobot-v2.1 conversion `fywang/calvin-task-ABCD-D-lerobot` (24,053 episodes,
64-step labeled segments, 10 fps, rel-action convention: dims 0-5 scaled EEF
deltas in ~[-1,1] (pos x50, rot x20), dim 6 gripper +/-1 — verified July 12).

Parquet files are fetched on demand and cached under HF_HOME.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

REPO = "fywang/calvin-task-ABCD-D-lerobot"
N_EPISODES = 24053


def episode_indices(n: int | None = None):
    return list(range(N_EPISODES if n is None else min(n, N_EPISODES)))


@lru_cache(maxsize=4096)
def _episode_df(ep: int):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                        repo_type="dataset")
    return pd.read_parquet(p, columns=["action", "observation.state"])


def episode_arrays(ep: int):
    """Returns (state, None, action) — action is (L, 7) float64, LIBERO-like
    (rel deltas + gripper sign). State is (L, 15)."""
    df = _episode_df(ep)
    action = np.stack(df["action"].values).astype(np.float64)
    state = np.stack(df["observation.state"].values).astype(np.float64)
    return state, None, action
