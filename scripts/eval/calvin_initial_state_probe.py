"""Separate CALVIN reset-state determinism from renderer byte identity."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from calvin_eval_client import (get_env_state_for_initial_condition,
                                load_multistep_sequences, make_env_and_oracle,
                                pack_obs)


def byte_hash(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_root", required=True)
    parser.add_argument("--calvin_repo", required=True)
    parser.add_argument("--n_seq", type=int, default=50)
    parser.add_argument("--run_index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.with_suffix(".npz").exists():
        raise FileExistsError(args.out)

    module = load_multistep_sequences(args.calvin_repo)
    sequences = module.get_sequences(1000)[:args.n_seq]
    annotations = OmegaConf.load(os.path.join(
        args.calvin_repo, "calvin_models", "conf", "annotations",
        "new_playtable_validation.yaml"))
    env, _ = make_env_and_oracle(args.debug_root)
    robot, scene, top, wrist, records = [], [], [], [], []
    for index, (initial_condition, sequence) in enumerate(sequences):
        expected_robot, expected_scene = get_env_state_for_initial_condition(initial_condition)
        env.reset(robot_obs=expected_robot, scene_obs=expected_scene)
        # Match rollout(): it asks the environment for a fresh observation
        # immediately after reset, then includes the first task string in the
        # request sent to the policy process.
        observation = env.get_obs()
        first_task = str(annotations[sequence[0]][0])
        packed = pack_obs(observation, first_task)
        actual_robot = np.asarray(observation["robot_obs"], dtype="<f8")
        actual_scene = np.asarray(observation["scene_obs"], dtype="<f8")
        image_top = np.asarray(observation["rgb_obs"]["rgb_static"], dtype=np.uint8)
        image_wrist = np.asarray(observation["rgb_obs"]["rgb_gripper"], dtype=np.uint8)
        robot.append(actual_robot)
        scene.append(actual_scene)
        top.append(image_top)
        wrist.append(image_wrist)
        records.append({"index": index, "sequence": list(sequence),
                        "first_task": first_task,
                        "packed_obs_sha256": hashlib.sha256(
                            pickle.dumps(packed, protocol=4)).hexdigest(),
                        "expected_robot_sha256": byte_hash(np.asarray(expected_robot, dtype="<f8")),
                        "expected_scene_sha256": byte_hash(np.asarray(expected_scene, dtype="<f8")),
                        "actual_robot_sha256": byte_hash(actual_robot),
                        "actual_scene_sha256": byte_hash(actual_scene),
                        "top_sha256": byte_hash(image_top), "wrist_sha256": byte_hash(image_wrist)})
    arrays_path = args.out.with_suffix(".npz")
    np.savez_compressed(arrays_path, robot=np.stack(robot), scene=np.stack(scene),
                        top=np.stack(top), wrist=np.stack(wrist))
    artifact = {"run_index": args.run_index, "n_seq": args.n_seq, "records": records,
                "arrays": str(arrays_path),
                "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest()}
    with args.out.open("x") as stream:
        json.dump(artifact, stream, indent=2)


if __name__ == "__main__":
    main()
