"""Generative-dispersion probe: is the spline head's spatial deficit caused by
flow-matching prediction noise amplifying differently in control-point space?

For each of N real observations, draw K action chunks from the SAME observation
and measure the dispersion (std across samples) of the decoded cumulative
POSITION path, at the start / middle / end of the executed window. Waypoint
noise is per-step-local; a noisy interior control point spreads over many steps.

  python probe_dispersion.py --ckpts A=<path> C=<path> --task libero_spatial --n_obs 12 --k 16
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--task", default="libero_spatial")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--n_obs", type=int, default=12)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--warm_steps", type=int, default=60)
    args = ap.parse_args()

    env_cfg = LiberoEnvCfg(task=args.task, task_ids=[args.task_id], control_mode="relative")
    envs = make_env(env_cfg, n_envs=1)
    suite = list(envs.keys())[0]
    env = envs[suite][args.task_id] if isinstance(envs[suite], dict) else envs[suite]

    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        policy.eval()
        preproc, postproc = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

        disp_start, disp_mid, disp_end = [], [], []
        for ep in range(args.n_obs):
            obs, _ = env.reset(seed=3000 + ep)
            policy.reset()
            # roll the policy in for warm_steps so observations include contact
            # approach, not just the episode start
            steps = int(args.warm_steps * (0.3 + 0.7 * (ep % 4) / 3))
            for _ in range(steps):
                observation = preprocess_observation(obs)
                try:
                    observation["task"] = list(env.call("task_description"))
                except Exception:
                    observation["task"] = [""]
                observation = env_pre(observation)
                observation = preproc(observation)
                with torch.inference_mode():
                    a = policy.select_action(observation)
                a = env_post({ACTION: postproc(a)})[ACTION].cpu().numpy()[0]
                obs, *_ = env.step(a[None, :] if a.ndim == 1 else a)

            observation = preprocess_observation(obs)
            try:
                observation["task"] = list(env.call("task_description"))
            except Exception:
                observation["task"] = [""]
            observation = env_pre(observation)
            observation = preproc(observation)

            paths = []
            with torch.inference_mode():
                for _ in range(args.k):
                    policy.reset()
                    chunk = []
                    a0 = policy.select_action(dict(observation))
                    chunk.append(env_post({ACTION: postproc(a0)})[ACTION].cpu().numpy()[0])
                    # drain the rest of this generated chunk from the queue
                    while len(policy._queues[ACTION]) > 0:
                        a = policy.select_action(dict(observation))
                        chunk.append(env_post({ACTION: postproc(a)})[ACTION].cpu().numpy()[0])
                    chunk = np.stack(chunk)[:, :6]
                    paths.append(np.cumsum(chunk, axis=0))
            L = min(p.shape[0] for p in paths)
            P = np.stack([p[:L] for p in paths])            # (K, L, 6)
            std_t = np.linalg.norm(P.std(axis=0), axis=-1)  # (L,) dispersion over samples
            disp_start.append(std_t)                        # per-step curve

        # dispersion at FIXED step indices -> horizon-matched across heads
        Lmin = min(len(s) for s in disp_start)
        curves = np.stack([s[:Lmin] for s in disp_start])   # (n_obs, Lmin)
        fixed = [j for j in (2, 4, 9, 14, 19) if j < Lmin]
        vals = "  ".join(f"s{j+1}={curves[:, j].mean():.4f}" for j in fixed)
        print(f"DISPERSION {name}: {vals}  (chunk_len={Lmin}, K={args.k}, "
              f"n_obs={args.n_obs}, task={args.task})", flush=True)
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
