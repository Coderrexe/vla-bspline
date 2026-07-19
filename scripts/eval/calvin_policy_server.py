"""Policy half of the two-process CALVIN evaluator (runs in the *lerobot* env).

calvin_env needs py3.10/numpy-1.23 pins that conflict with lerobot's stack, so
policy inference runs here and the simulator in calvin_eval_client.py, joined
by a length-prefixed JSON-ish pickle protocol over localhost TCP. The wire
format carries images as raw bytes + shape and vectors as plain lists — never
pickled ndarrays — because numpy-2 array pickles do not load in the client's
numpy 1.23.

Loads any LeRobot policy checkpoint (smolvla baseline or smolvla_spline) with
its saved processor pipeline, so normalization (incl. the spline head's
IDENTITY action normalization) comes from the checkpoint, same as lerobot_eval.

  python calvin_policy_server.py --ckpt .../checkpoints/last/pretrained_model \
      --dataset_root ~/scratch/vla_bspline/calvin_v30 --port 23456
"""
from __future__ import annotations

import argparse
import pickle
import socket
import struct

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="pretrained_model dir")
    ap.add_argument("--dataset_root", required=True, help="local calvin_v30 root (ds_meta for make_policy)")
    ap.add_argument("--repo_id", default="fywang/calvin-task-ABCD-D-lerobot")
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()

    policy_cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    policy_cfg.pretrained_path = args.ckpt
    ds_meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    preproc, postproc = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    def act(m):
        top = np.frombuffer(m["top"][0], dtype=np.uint8).reshape(m["top"][1])
        wrist = np.frombuffer(m["wrist"][0], dtype=np.uint8).reshape(m["wrist"][1])
        batch = {
            "observation.state": torch.tensor(m["state"], dtype=torch.float32).unsqueeze(0),
            "observation.environment_state": torch.tensor(m["scene"], dtype=torch.float32).unsqueeze(0),
            "observation.images.top": torch.from_numpy(top.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
            "observation.images.wrist": torch.from_numpy(wrist.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
            "task": [m["task"]],
        }
        batch = preproc(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postproc(action)
        a = action.cpu().numpy() if torch.is_tensor(action) else action[ACTION].cpu().numpy()
        return np.asarray(a).reshape(-1).tolist(), int(getattr(policy, "last_predicted_T", -1))

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"SERVER_READY {args.port}", flush=True)
    conn, _ = srv.accept()
    n_acts = 0
    while True:
        m = recv_msg(conn)
        if m is None or m["cmd"] == "quit":
            break
        if m["cmd"] == "reset":
            policy.reset()
            send_msg(conn, {"ok": True})
        elif m["cmd"] == "act":
            a, that = act(m)
            n_acts += 1
            send_msg(conn, {"action": a, "that": that})
    print(f"SERVER_DONE acts={n_acts}", flush=True)


if __name__ == "__main__":
    main()
