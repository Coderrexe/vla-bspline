"""Simulator half of the two-process CALVIN evaluator (runs in the *calvin* env).

Implements the official CALVIN long-horizon protocol (evaluate_policy.py):
sampled 5-task chains via the official multistep_sequences logic, official
initial-condition randomization (bit-identical seeds via a vendored FNV-1
hash), the calvin_env task oracle for success, EP_LEN=360 env steps per
subtask. The policy runs in calvin_policy_server.py (lerobot env); this
process holds each 10 fps action for --repeat sim steps (=3 at native 30 Hz),
the convention proven by the replay gate (RMSE 0.0065 m vs 0.118 for naive
subsample).

  ~/.conda/envs/calvin/bin/python calvin_eval_client.py \
      --debug_root .../calvin_debug_dataset/validation \
      --calvin_repo ~/vla_bspline/calvin --port 23456 --n_seq 100 --out r.json

--random replaces the server with N(0,.5) actions for a serverless smoke of
env + oracle + sequence machinery.
"""
from __future__ import annotations

import argparse
import collections
import collections.abc
import contextlib
import json
import os
import pickle
import socket
import struct
import sys
import time
import types

# calvin_env targets py3.8; shim the py3.10 stdlib removals its deps rely on
for _n in ("Mapping", "MutableMapping", "Sequence", "Set", "Iterable", "Callable"):
    if not hasattr(collections, _n):
        setattr(collections, _n, getattr(collections.abc, _n))
import fractions
import math
if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

import numpy as np
from numpy import pi

EP_LEN = 360  # official per-subtask budget, in native 30Hz env steps


# ---------- wire protocol (mirror of calvin_policy_server.py) ----------
def recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    hdr = recvall(conn, 8)
    if hdr is None:
        return None
    (n,) = struct.unpack("!Q", hdr)
    return pickle.loads(recvall(conn, n))


def send_msg(conn, obj):
    b = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!Q", len(b)) + b)


# ---------- vendored official-protocol pieces ----------
# calvin_agent.evaluation.utils drags in MCIL/lightning/pyhash; vendor the two
# small functions the protocol needs and stub the module so the official
# multistep_sequences.py imports cleanly.
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def fnv1_32(s: str) -> int:
    # pyhash.fnv1_32 equivalent (FNV-1: multiply then xor), for bit-identical
    # official initial-condition seeds without the pyhash C extension
    h = 0x811C9DC5
    for b in s.encode():
        h = (h * 0x01000193) & 0xFFFFFFFF
        h ^= b
    return h


def load_multistep_sequences(calvin_repo):
    import importlib.util

    pkg = types.ModuleType("calvin_agent")
    ev = types.ModuleType("calvin_agent.evaluation")
    ut = types.ModuleType("calvin_agent.evaluation.utils")
    ut.temp_seed = temp_seed
    sys.modules["calvin_agent"] = pkg
    sys.modules["calvin_agent.evaluation"] = ev
    sys.modules["calvin_agent.evaluation.utils"] = ut
    path = os.path.join(calvin_repo, "calvin_models", "calvin_agent", "evaluation", "multistep_sequences.py")
    spec = importlib.util.spec_from_file_location("multistep_sequences", path)
    mod = importlib.util.module_from_spec(spec)
    # ProcessPoolExecutor pickles its functions by module reference; forked
    # workers resolve that through sys.modules, so register before exec
    sys.modules["multistep_sequences"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_env_state_for_initial_condition(initial_condition):
    # verbatim from calvin_agent/evaluation/utils.py, hasher swapped for fnv1_32
    robot_obs = np.array(
        [0.02586889, -0.2313129, 0.5712808, 3.09045411, -0.02908596, 1.50013585,
         0.07999963, -1.21779124, 1.03987629, 2.11978254, -2.34205014, -0.87015899,
         1.64119093, 0.55344928, 1.0]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    seed = fnv1_32(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


# ---------- env / policy plumbing ----------
def make_env_and_oracle(debug_root):
    from omegaconf import OmegaConf
    import hydra

    conf = OmegaConf.load(os.path.join(debug_root, ".hydra", "merged_config.yaml"))
    OmegaConf.resolve(conf)
    env_conf = conf.env
    env_conf["use_egl"] = False
    env_conf["show_gui"] = False
    env_conf["use_vr"] = False
    env_conf["use_scene_info"] = True
    # tacto's mp renderer hangs headless; policy needs only static + gripper
    env_conf["cameras"] = {k: v for k, v in env_conf["cameras"].items() if k != "tactile"}
    env = hydra.utils.instantiate(env_conf)
    oracle = hydra.utils.instantiate(conf.tasks)
    return env, oracle


def pack_obs(obs, task):
    rgb = obs["rgb_obs"]
    top = np.ascontiguousarray(rgb["rgb_static"], dtype=np.uint8)
    wrist = np.ascontiguousarray(rgb["rgb_gripper"], dtype=np.uint8)
    return {
        "cmd": "act",
        "top": (top.tobytes(), list(top.shape)),
        "wrist": (wrist.tobytes(), list(wrist.shape)),
        "state": np.asarray(obs["robot_obs"], dtype=float).tolist(),
        "scene": np.asarray(obs["scene_obs"], dtype=float).tolist(),
        "task": task,
    }


class Policy:
    def __init__(self, port, random=False):
        self.random = random
        self.conn = None
        if not random:
            self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.conn.connect(("127.0.0.1", port))

    def reset(self):
        if self.random:
            return
        send_msg(self.conn, {"cmd": "reset"})
        recv_msg(self.conn)

    def act(self, obs, task):
        if self.random:
            a = np.clip(np.random.randn(7) * 0.5, -1, 1)
            a[6] = np.sign(a[6]) or 1.0
            return a, -1
        send_msg(self.conn, pack_obs(obs, task))
        r = recv_msg(self.conn)
        return np.asarray(r["action"], dtype=float), r["that"]

    def close(self):
        if self.conn is not None:
            try:
                send_msg(self.conn, {"cmd": "quit"})
                self.conn.close()
            except OSError:
                pass


def rollout(env, oracle, policy, subtask, lang, ep_len, repeat):
    obs = env.get_obs()
    policy.reset()
    start_info = env.get_info()
    action = None
    that = []  # predicted duration per policy call (-1 = no time head)
    for step in range(ep_len):
        if step % repeat == 0:
            action, t_hat = policy.act(obs, lang)
            that.append(int(t_hat))
            action = np.asarray(action, dtype=float).copy()
            action[6] = 1.0 if action[6] > 0 else -1.0  # oracle-legal gripper
        obs, _, _, current_info = env.step(action)
        if len(oracle.get_task_info_for_set(start_info, current_info, {subtask})) > 0:
            return True, step + 1, that
    return False, ep_len, that


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug_root", required=True)
    ap.add_argument("--calvin_repo", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--n_seq", type=int, default=100)
    ap.add_argument("--seq_offset", type=int, default=0,
                    help="evaluate sequences [offset, offset+n_seq) of the deterministic "
                         "official list — lets a big eval shard across jobs")
    ap.add_argument("--n_total", type=int, default=0,
                    help="size of the official list to generate before slicing "
                         "(default: offset+n_seq)")
    ap.add_argument("--ep_len", type=int, default=EP_LEN)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    mseq = load_multistep_sequences(args.calvin_repo)
    n_total = args.n_total or (args.seq_offset + args.n_seq)
    eval_sequences = mseq.get_sequences(n_total)[args.seq_offset: args.seq_offset + args.n_seq]
    val_annotations = OmegaConf.load(os.path.join(
        args.calvin_repo, "calvin_models", "conf", "annotations", "new_playtable_validation.yaml"))
    env, oracle = make_env_and_oracle(args.debug_root)
    policy = Policy(args.port, random=args.random)

    chain_lens, records, t0 = [], [], time.time()
    for i, (initial_state, sequence) in enumerate(eval_sequences):
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        done_count = 0
        steps, thats = [], []
        for subtask in sequence:
            lang = val_annotations[subtask][0]
            success, n_steps, that = rollout(env, oracle, policy, subtask, lang, args.ep_len, args.repeat)
            steps.append(n_steps)
            thats.append(that)
            if not success:
                break
            done_count += 1
        chain_lens.append(done_count)
        records.append({"sequence": list(sequence), "solved": done_count, "steps": steps,
                        "that": thats})
        if (i + 1) % 10 == 0:
            cl = np.array(chain_lens)
            sr = [float((cl >= k).mean()) for k in range(1, 6)]
            print(f"[{i+1}/{args.n_seq}] avg_len {cl.mean():.2f} SR {sr} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    policy.close()
    cl = np.array(chain_lens)
    result = {
        "n_seq": int(len(cl)),
        "avg_len": float(cl.mean()),
        "sr": [float((cl >= k).mean()) for k in range(1, 6)],
        "ep_len": args.ep_len,
        "repeat": args.repeat,
        "random": bool(args.random),
        "records": records,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print("SUMMARY " + json.dumps({k: result[k] for k in ("n_seq", "avg_len", "sr")}), flush=True)


if __name__ == "__main__":
    main()
