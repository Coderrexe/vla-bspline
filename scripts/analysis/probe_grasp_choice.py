"""Grasp-choice steering probe (EVAL_DESIGN 8b, discrete metric).

Two-object LIBERO scene (libero_10 task 7: alphabet soup + cream cheese).
Command "move toward the X, close gripper, grasp" for X in {soup, cheese};
at the first gripper-close, read which object the EEF is nearer (sim body
positions). Steering = P(chosen == commanded). Control policy (Cn8, never
trained on clauses) should ignore the clause; ClangAdv should follow it.

  python probe_grasp_choice.py --ckpt <variant_dir> --episodes 10
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

PROMPTS = {
    "soup": "move toward the alphabet soup can, close gripper, grasp can",
    "cheese": "move toward the cream cheese box, close gripper, grasp box",
}
BODIES = {"soup": "alphabet_soup_1_main", "cheese": "cream_cheese_1_main"}


def inner_sim(env):
    base = env.envs[0] if hasattr(env, "envs") else env
    u = base
    for _ in range(8):
        if type(u).__name__ == "LiberoEnv":
            return u._env.sim
        u = getattr(u, "env", getattr(u, "unwrapped", u))
    raise RuntimeError("sim not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--seed_base", type=int, default=7000)
    args = ap.parse_args()

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    env_cfg = LiberoEnvCfg(task="libero_10", task_ids=[7], control_mode="relative")
    envs = make_env(env_cfg, n_envs=1)
    suite = list(envs.keys())[0]
    env = envs[suite][7] if isinstance(envs[suite], dict) else envs[suite]

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preproc, postproc = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    results = {k: [] for k in PROMPTS}
    for target, prompt in PROMPTS.items():
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed_base + ep)
            # _env (and its sim) is built lazily on reset — resolve fresh each episode
            sim = inner_sim(env)
            bid = {k: sim.model.body_name2id(v) for k, v in BODIES.items()}
            torch.manual_seed(args.seed_base + ep)
            policy.reset()
            choice = None
            for step in range(args.max_steps):
                observation = preprocess_observation(obs)
                observation["task"] = [prompt]
                observation = env_pre(observation)
                eef = np.asarray(observation["observation.state"][0])[:3] \
                    if not torch.is_tensor(observation["observation.state"]) \
                    else observation["observation.state"][0][:3].cpu().numpy()
                observation = preproc(observation)
                with torch.inference_mode():
                    action = policy.select_action(observation)
                action = postproc(action)
                tr = env_post({ACTION: action})
                a = tr[ACTION].cpu().numpy()[0]
                if a[6] > 0 and choice is None:      # first gripper close
                    d = {k: float(np.linalg.norm(eef - sim.data.body_xpos[b]))
                         for k, b in bid.items()}
                    choice = min(d, key=d.get)
                    break
                obs, *_ = env.step(a[None, :])
            results[target].append(choice)
        n_ok = sum(1 for c in results[target] if c == target)
        n_alt = sum(1 for c in results[target] if c is not None and c != target)
        n_none = sum(1 for c in results[target] if c is None)
        print(f"commanded={target:7s} -> chose commanded {n_ok}/{args.episodes}, "
              f"other {n_alt}, no-grasp {n_none}", flush=True)

    total_ok = sum(sum(1 for c in results[t] if c == t) for t in PROMPTS)
    total_dec = sum(sum(1 for c in results[t] if c is not None) for t in PROMPTS)
    print(f"STEERING ACCURACY: {total_ok}/{total_dec} decided grasps followed the command")


if __name__ == "__main__":
    main()
