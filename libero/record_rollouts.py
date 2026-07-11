"""Minimal rollout recorder for smoothness analysis: run a policy in LIBERO
(single task-suite task), record per-step commanded actions + observation.state,
save npz. Mirrors lerobot_eval's loading + rollout recipe 1:1, without the
(broken-for-libero) dataset recorder.

  python record_rollouts.py --ckpt <pretrained_model> --task libero_object \
      --task_id 0 --episodes 5 --out rollouts.npz
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default="libero_object")
    ap.add_argument("--task_id", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=280)
    ap.add_argument("--control_freq", type=int, default=20)
    ap.add_argument("--out", default="rollouts.npz")
    # T-hat-stagnation recovery: when the duration head's countdown stalls
    # (trailing-window stall_frac > threshold), back off (retreat upward a few
    # steps), clear the plan, and re-approach. Benign on false alarms.
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--recover_thr", type=float, default=0.45)
    ap.add_argument("--recover_window", type=int, default=80)
    ap.add_argument("--recover_cooldown", type=int, default=60)
    args = ap.parse_args()

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    env_cfg = LiberoEnvCfg(task=args.task, task_ids=[args.task_id], control_mode="relative",
                           control_freq=args.control_freq)

    envs = make_env(env_cfg, n_envs=1)
    suite = list(envs.keys())[0]
    env = envs[suite][args.task_id] if isinstance(envs[suite], dict) else envs[suite]

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preproc, postproc = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    all_actions, all_states, all_that, ep_lens, successes, n_calls = [], [], [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=1000 + ep)
        policy.reset()
        acts, states, that = [], [], []
        done, step, ep_success = False, 0, False
        n_recoveries, last_recover, cap = 0, -10_000, 24
        while not done and step < args.max_steps:
            if (args.recover and step - last_recover > args.recover_cooldown
                    and len(that) >= 68):
                th = np.array([t for t in that[-args.recover_window:] if t > 0], dtype=float)
                if len(th) >= 28:
                    trend = th[8:] - th[:-8]
                    near = th[8:] < cap
                    if near.sum() >= 20 and ((trend >= 0) & near).sum() / near.sum() > args.recover_thr:
                        # back off: open-loop upward retreat, then replan fresh
                        g_hold = acts[-1][6] if acts else -1.0  # keep gripper state (never drop a held object)
                        for _ in range(4):
                            ra = np.zeros(7, dtype=np.float32); ra[2] = 0.4; ra[6] = g_hold
                            obs, *_ = env.step(ra[None, :])
                            acts.append(ra); states.append(states[-1]); that.append(-1)
                            step += 1
                        policy.reset()
                        n_recoveries += 1
                        last_recover = step
                        continue
            observation = preprocess_observation(obs)
            try:
                observation["task"] = list(env.call("task_description"))
            except Exception:
                observation["task"] = [""]
            observation = env_pre(observation)
            states.append(observation["observation.state"][0].cpu().numpy().copy()
                          if torch.is_tensor(observation["observation.state"])
                          else np.asarray(observation["observation.state"][0]))
            observation = preproc(observation)
            with torch.inference_mode():
                action = policy.select_action(observation)
            that.append(int(getattr(policy, "last_predicted_T", -1)))  # -1 = no time head
            action = postproc(action)
            tr = env_post({ACTION: action})
            a = tr[ACTION].cpu().numpy()[0]
            acts.append(a.copy())
            obs, rew, term, trunc, info = env.step(a[None, :] if a.ndim == 1 else a)
            # success = env reward hits 1 (LIBERO gives sparse success reward),
            # robust to gymnasium final_info packing differences
            if np.asarray(rew).max() >= 1.0:
                ep_success = True
            done = bool(np.array(term).any() or np.array(trunc).any())
            step += 1
        all_actions.append(np.stack(acts))
        all_states.append(np.stack(states))
        all_that.append(np.array(that))
        ep_lens.append(step)
        successes.append(ep_success)
        calls = int(getattr(policy, "n_chunks_generated", -1))  # -1 = counter unsupported
        n_calls.append(calls)
        rec = f" recoveries={n_recoveries}" if args.recover else ""
        print(f"ep {ep}: steps={step} success={ep_success} policy_calls={calls}{rec}", flush=True)

    np.savez(
        args.out,
        actions=np.array(all_actions, dtype=object),
        states=np.array(all_states, dtype=object),
        t_hat=np.array(all_that, dtype=object),
        ep_lens=np.array(ep_lens),
        successes=np.array(successes),
        n_policy_calls=np.array(n_calls),
    )
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
