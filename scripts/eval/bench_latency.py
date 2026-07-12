"""Per-call inference latency: waypoint SmolVLA (50 action tokens) vs spline head
(6-8 spline tokens) — the compute half of the cost-efficiency story.

Times the full chunk-generation call (vision+language encode + flow-matching expert
denoise) on identical observations, CUDA-synchronized, warmup excluded.

  python bench_latency.py --ckpts A=<path> C=<path> --task libero_object --n 50
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies import make_policy, make_pre_post_processors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="NAME=path pairs")
    ap.add_argument("--task", default="libero_object")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    env_cfg = LiberoEnvCfg(task=args.task, task_ids=[0], control_mode="relative")
    envs = make_env(env_cfg, n_envs=1)
    suite = list(envs.keys())[0]
    env = envs[suite][0] if isinstance(envs[suite], dict) else envs[suite]
    obs, _ = env.reset(seed=0)

    for pair in args.ckpts:
        name, path = pair.split("=", 1)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        policy.eval()
        preproc, _ = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
        )
        env_pre, _ = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

        observation = preprocess_observation(obs)
        try:
            observation["task"] = list(env.call("task_description"))
        except Exception:
            observation["task"] = [""]
        observation = env_pre(observation)
        observation = preproc(observation)

        lat = []
        with torch.inference_mode():
            for i in range(args.n + 10):
                policy.reset()  # force a fresh chunk generation every call
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                policy.select_action(dict(observation))
                torch.cuda.synchronize()
                if i >= 10:  # warmup excluded
                    lat.append((time.perf_counter() - t0) * 1000)
        lat = np.array(lat)
        print(f"LATENCY {name}: median {np.median(lat):.1f} ms  p10 {np.percentile(lat,10):.1f}"
              f"  p90 {np.percentile(lat,90):.1f}  (n={len(lat)}, chunk tokens="
              f"{getattr(policy.config, 'chunk_size', '?')})", flush=True)
        del policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
