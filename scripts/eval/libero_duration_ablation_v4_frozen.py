"""Frozen-model successor to the stochastic paired LIBERO duration evaluator.

V3 deliberately reset LIBERO before every arm.  V4 keeps its five-arm,
counterbalanced stochastic-block design, but creates one live task environment
for a task/state/repeat block, performs one explicit reset, and restores that
same compiled model plus ``mjSTATE_INTEGRATION`` for every arm.  No arm after
capture invokes ``env.reset``.
"""

from __future__ import annotations

from libero_duration_ablation_v2 import STRUCTURED_HASH_PROTOCOL, _active_duration_consumers, _single_action_sha256, _structured_sha256
from libero_locked_eval_v2 import _base_env, _determinism_metadata, _git_commit, _seed_all, _seed_policy_after_reset, _sha256

import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from deterministic_eval import ACTION_TRACE_PROTOCOL, ExecutedActionTrace, atomic_write_json_new
from duration_intervention_v2 import load_complete_fixed_manifest
from duration_intervention_v3 import PROTOCOL as INTERVENTION_PROTOCOL, DurationInterventionV3
from libero_frozen_reset import PROTOCOL as FROZEN_RESET_PROTOCOL, TERMINAL_AUTORESET_PROTOCOL, FrozenLiberoState, step_without_terminal_autoreset
from libero_hidden_state import PROTOCOL as HIDDEN_STATE_PROTOCOL, integration_state_sha256, locate_robosuite_sim, model_xml_sha256
from stochastic_duration_v3 import BLOCK_SEED_PROTOCOL, COUNTERBALANCE_PROTOCOL, EFFICACY_PERMUTATIONS, WATERMARK, aggregate_predicted_pairs, block_seeds, canonical_json_sha256, observation_component_hashes, predicted_pair_diagnostic, validate_embedded_sha256, materialize_task_order_manifest


EVALUATION_PROTOCOL = "libero_duration_stochastic_paired_block_v4_frozen_model"
BLOCK_PROTOCOL = "duration_v4_frozen_model_atomic_complete_block"
RESUME_PROTOCOL = "duration_v4_frozen_model_atomic_block_resume_identity"
ARM_TO_MODE = {
    "predicted_donor": "predicted", "predicted_eval": "predicted",
    "fixed": "fixed", "shuffled": "shuffled", "predicted_closure": "predicted",
}
ARM_LABELS = tuple(ARM_TO_MODE)


def _append_component_step(traces: dict[str, list[str]], components: dict[str, str], step: int) -> None:
    if step == 0:
        if traces:
            raise RuntimeError("component trace must be empty at step zero")
        traces.update({path: [digest] for path, digest in components.items()})
    else:
        if set(traces) != set(components):
            raise RuntimeError("observation component schema changed within episode")
        for path, digest in components.items():
            traces[path].append(digest)


def _capture_once(env: Any, state_id: int, env_seed: int) -> tuple[Any, int]:
    """The sole reset in a block; all later arms use FrozenLiberoState.restore."""
    base = _base_env(env)
    _seed_all(env_seed)
    base.init_state_id = state_id
    observation, _ = env.reset(seed=env_seed)
    realized = int(base.init_state_id) - int(base._reset_stride)
    if realized != state_id:
        raise RuntimeError(f"requested init state {state_id}, realized {realized}")
    return observation, realized


def _run_frozen_episode(*, arm_label: str, env: Any, frozen: FrozenLiberoState, task_id: int, state_id: int, repeat_index: int, env_seed: int, policy_seed: int, task_description: str, max_steps: int, policy: Any, controller: DurationInterventionV3, preprocessor: Any, postprocessor: Any, env_preprocessor: Any, env_postprocessor: Any, action_key: str, preprocess_observation: Any, shuffle_source: list[int] | None = None, shuffle_source_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Execute one arm without rebuilding or resetting its compiled XML model."""
    base_env = _base_env(env)
    observation = frozen.restore(base_env, _structured_sha256, wrapper_root=env)
    simulator, simulator_path = locate_robosuite_sim(base_env)
    # Reset all policy/controller state after restoration and reseed policy RNG
    # exactly as v3 did immediately after its reset.
    _seed_policy_after_reset(policy_seed)
    policy.reset()
    controller.begin_episode(ARM_TO_MODE[arm_label], task_id, state_id, repeat_index, shuffle_source=shuffle_source)
    initial_raw_hash = _structured_sha256(observation)
    initial_model_hash = model_xml_sha256(simulator)
    initial_state_hash = integration_state_sha256(simulator)
    if (initial_raw_hash != frozen.raw_observation_sha256 or initial_model_hash != frozen.compiled_model_xml_sha256 or initial_state_hash != frozen.integration_state_sha256):
        raise RuntimeError("frozen restore verification failed before policy inference")

    success = False; step = 0; select_action_calls = 0
    action_trace = ExecutedActionTrace()
    action_step_hashes: list[str] = []; action_prefix_hashes: list[str] = []
    raw_hashes: list[str] = []; processed_hashes: list[str] = []; state_hashes: list[str] = []
    raw_components: dict[str, list[str]] = {}; processed_components: dict[str, list[str]] = {}
    replan_inputs: list[dict[str, Any]] = []
    suppressed_terminal_autoresets = 0
    while step < max_steps:
        state_hash = integration_state_sha256(simulator)
        raw_hash = _structured_sha256(observation)
        raw_part = observation_component_hashes(observation, _structured_sha256)
        processed = preprocess_observation(observation)
        processed["task"] = [task_description]
        processed = preprocessor(env_preprocessor(processed))
        processed_hash = _structured_sha256(processed)
        processed_part = observation_component_hashes(processed, _structured_sha256)
        raw_hashes.append(raw_hash); processed_hashes.append(processed_hash); state_hashes.append(state_hash)
        _append_component_step(raw_components, raw_part, step)
        _append_component_step(processed_components, processed_part, step)
        before = int(getattr(policy, "n_chunks_generated", -1))
        with torch.inference_mode():
            action = policy.select_action(processed)
        select_action_calls += 1
        after = int(getattr(policy, "n_chunks_generated", -1))
        if before < 0 or after - before not in (0, 1):
            raise RuntimeError("policy n_chunks_generated contract failed")
        if after == before + 1:
            replan_inputs.append({"chunk_index": len(replan_inputs), "environment_step": step, "raw_observation_sha256": raw_hash, "processed_observation_sha256": processed_hash, "mujoco_integration_state_sha256": state_hash})
        action = postprocessor(action)
        transition = env_postprocessor({action_key: action})
        action_numpy = transition[action_key].detach().cpu().numpy()
        action_step_hashes.append(_single_action_sha256(action_numpy)); action_trace.update(action_numpy); action_prefix_hashes.append(action_trace.hexdigest())
        transition, reset_suppressed = step_without_terminal_autoreset(
            env, base_env, action_numpy
        )
        observation, reward, terminated, truncated, info = transition
        suppressed_terminal_autoresets += int(reset_suppressed)
        step += 1
        success = success or bool(np.asarray(reward).max() >= 1.0)
        for candidate in (info.get("is_success"), info.get("final_info", {}).get("is_success") if isinstance(info.get("final_info"), dict) else None):
            if candidate is not None:
                success = success or bool(np.asarray(candidate).reshape(-1)[0])
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break
    duration_record = controller.end_episode()
    chunks = int(getattr(policy, "n_chunks_generated", -1))
    if not (step == select_action_calls == action_trace.count == len(raw_hashes) == len(processed_hashes) == len(state_hashes) == len(action_step_hashes) == len(action_prefix_hashes)):
        raise RuntimeError("step/call/action/observation trace lengths disagree")
    if len(replan_inputs) != chunks or duration_record["n_duration_calls"] != chunks:
        raise RuntimeError("duration/replan chunk count mismatch")
    for replan, duration in zip(replan_inputs, duration_record["duration_chunks"], strict=True):
        replan["predicted_duration"] = duration["predicted_duration"]; replan["executed_duration"] = duration["executed_duration"]
        replan["canonical_replan_record_sha256"] = canonical_json_sha256(replan)
    source_length = len(shuffle_source) if shuffle_source is not None else None
    return {
        "arm_label": arm_label, "duration_mode": ARM_TO_MODE[arm_label], "task_id": task_id, "state_id": state_id, "repeat_index": repeat_index,
        "env_seed_u32": env_seed, "policy_seed_u64": policy_seed, "success": success, "steps": step, "select_action_calls": select_action_calls, "chunk_generations": chunks,
        "observation_hash_protocol": STRUCTURED_HASH_PROTOCOL, "observation_component_hash_protocol": "observation_component_merkle_sha256_v1", "hidden_state_protocol": HIDDEN_STATE_PROTOCOL, "frozen_reset_protocol": FROZEN_RESET_PROTOCOL, "terminal_autoreset_protocol": TERMINAL_AUTORESET_PROTOCOL, "suppressed_terminal_autoresets": suppressed_terminal_autoresets,
        "simulator_path": simulator_path, "compiled_model_xml_sha256": initial_model_hash, "initial_mujoco_integration_state_sha256": initial_state_hash,
        "initial_raw_observation_sha256": raw_hashes[0], "initial_processed_observation_sha256": processed_hashes[0], "raw_observation_step_sha256": raw_hashes, "processed_observation_step_sha256": processed_hashes,
        "raw_observation_component_step_sha256": raw_components, "processed_observation_component_step_sha256": processed_components, "mujoco_integration_state_step_sha256": state_hashes, "policy_replan_inputs": replan_inputs,
        "executed_action_count": action_trace.count, "executed_action_trace_protocol": ACTION_TRACE_PROTOCOL, "executed_action_trace_sha256": action_trace.hexdigest(), "executed_action_step_sha256": action_step_hashes, "executed_action_prefix_sha256": action_prefix_hashes,
        "shuffle_source_manifest_sha256": shuffle_source_manifest_sha256, "unused_counterfactual_source_values": max(source_length - chunks, 0) if source_length is not None else None,
        "equal_counterfactual_source_multiset": sorted(duration_record["executed_durations"]) == sorted(shuffle_source) if shuffle_source is not None else None,
        "frozen_restore_verified": True, **duration_record,
    }


def _initial_pairing(donor: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    names = ("env_seed_u32", "policy_seed_u64", "initial_raw_observation_sha256", "initial_processed_observation_sha256", "initial_mujoco_integration_state_sha256", "compiled_model_xml_sha256")
    return {"arm_label": episode["arm_label"], "all_frozen_initial_fields_agree": all(donor[name] == episode[name] for name in names), "seed_agreement": all(donor[name] == episode[name] for name in names[:2]), "raw_initial_observation_agreement": donor[names[2]] == episode[names[2]], "processed_initial_observation_agreement": donor[names[3]] == episode[names[3]], "mujoco_integration_initial_state_agreement": donor[names[4]] == episode[names[4]], "compiled_model_xml_agreement": donor[names[5]] == episode[names[5]], "frozen_restore_verified": bool(episode["frozen_restore_verified"])}


def _arm_summary(blocks: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    episodes = [block["episodes"][arm] for block in blocks]; calls = sum(item["n_duration_calls"] for item in episodes)
    changed = sum(item["n_intervened_duration_calls"] for item in episodes)
    absolute = sum(chunk["absolute_duration_intervention"] for item in episodes for chunk in item["duration_chunks"])
    equal = [item["equal_counterfactual_source_multiset"] for item in episodes if item["equal_counterfactual_source_multiset"] is not None]
    return {"arm_label": arm, "duration_mode": ARM_TO_MODE[arm], "role": "efficacy" if arm in ("predicted_eval", "fixed", "shuffled") else "trace_only_drift_diagnostic", "n_episodes": len(episodes), "n_successes": sum(int(item["success"]) for item in episodes), "success_rate": float(np.mean([item["success"] for item in episodes])), "n_steps": sum(item["steps"] for item in episodes), "n_duration_calls": calls, "n_intervened_duration_calls": changed, "intervention_call_fraction": changed / calls, "mean_absolute_duration_intervention": absolute / calls, "suppressed_terminal_autoresets": sum(item["suppressed_terminal_autoresets"] for item in episodes), "unused_counterfactual_source_values": sum(item["unused_counterfactual_source_values"] or 0 for item in episodes), "reused_counterfactual_values": sum(item["reused_counterfactual_values"] for item in episodes), "n_equal_counterfactual_source_multisets": sum(equal) if equal else None, "equal_counterfactual_source_multiset_fraction": sum(equal) / len(equal) if equal else None}


def _make_block_env(make_env: Any, env_cfg: Any, suite: str, task_id: int) -> tuple[Any, dict[int, Any]]:
    environments = make_env(env_cfg, n_envs=1)[suite]
    if task_id not in environments:
        raise ValueError(f"task {task_id} unavailable in {suite}")
    return environments[task_id], environments


def main() -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True); parser.add_argument("--fixed_manifest", required=True); parser.add_argument("--suite", default="libero_10"); parser.add_argument("--task_ids"); parser.add_argument("--state_start", type=int, default=0); parser.add_argument("--states_per_task", type=int, default=12); parser.add_argument("--repeats", type=int, default=1); parser.add_argument("--seed_base", type=int, default=100_000); parser.add_argument("--order_seed", type=int, default=20_260_821); parser.add_argument("--control_freq", type=int, default=20); parser.add_argument("--shuffle_seed", type=int, default=73_921); parser.add_argument("--shuffle_strength_threshold", type=float, default=0.5); parser.add_argument("--repeatability_equivalence_margin", type=float, default=.10); parser.add_argument("--require_exact_predicted_replay", action="store_true"); parser.add_argument("--require_frozen_initial_pairing", action="store_true"); parser.add_argument("--block_dir", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.state_start < 0 or args.states_per_task <= 0 or args.repeats <= 0: raise ValueError("invalid state/repeat grid")
    if not 0 <= args.shuffle_strength_threshold <= 1 or not 0 < args.repeatability_equivalence_margin < 1: raise ValueError("invalid gate threshold")
    checkpoint = Path(args.ckpt).resolve(); config_path = checkpoint / "config.json"; model_path = checkpoint / "model.safetensors"; output = Path(args.out); block_dir = Path(args.block_dir)
    if not config_path.is_file() or not model_path.is_file(): raise FileNotFoundError("checkpoint is incomplete")
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    block_dir.mkdir(parents=True, exist_ok=True); output.parent.mkdir(parents=True, exist_ok=True)
    fixed_duration, fixed_payload = load_complete_fixed_manifest(Path(args.fixed_manifest).resolve(), config_path)
    _seed_all(args.seed_base); policy_cfg = PreTrainedConfig.from_pretrained(str(checkpoint)); policy_cfg.pretrained_path = str(checkpoint)
    if not bool(getattr(policy_cfg, "predict_duration", False)): raise ValueError("checkpoint is not duration-enabled")
    active_consumers = _active_duration_consumers(policy_cfg)
    if active_consumers: raise ValueError(f"duration ablation requires neutral optional decode controllers: {active_consumers}")
    env_cfg = LiberoEnvCfg(task=args.suite, control_mode="relative", control_freq=args.control_freq, max_parallel_tasks=1)
    # Avoid a bootstrap environment that would create an extra compiled model.
    # LIBERO's standard evaluator suites are ten-task indexed grids.  A new
    # suite must name task IDs explicitly until its cardinality is reviewed.
    if args.task_ids is None:
        if args.suite != "libero_10":
            raise ValueError("--task_ids is required for a nonstandard LIBERO suite")
        task_ids = list(range(10))
    else:
        task_ids = sorted(int(item) for item in args.task_ids.split(",") if item.strip())
        if not task_ids or len(task_ids) != len(set(task_ids)) or min(task_ids) < 0:
            raise ValueError("--task_ids must be a nonempty duplicate-free nonnegative CSV")
    state_ids = list(range(args.state_start, args.state_start + args.states_per_task))
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg); policy.eval(); controller = DurationInterventionV3(fixed_duration=fixed_duration, shuffle_seed=args.shuffle_seed); controller.attach(policy)
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=policy_cfg, pretrained_path=str(checkpoint), preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}})
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    config = json.loads(config_path.read_text()); policy_source = Path(inspect.getfile(type(policy))).resolve(); evaluator_path = Path(__file__).resolve(); frozen_path = Path(inspect.getfile(FrozenLiberoState)).resolve()
    source_hashes = {"evaluator_sha256": _sha256(evaluator_path), "frozen_reset_helper_sha256": _sha256(frozen_path), "controller_sha256": _sha256(Path(inspect.getfile(DurationInterventionV3)).resolve()), "policy_source_sha256": _sha256(policy_source), "lerobot_commit": _git_commit(policy_source.parents[4])}
    resume_identity = {"protocol": RESUME_PROTOCOL, "checkpoint": str(checkpoint), "model_sha256": _sha256(model_path), "checkpoint_config_sha256": _sha256(config_path), "fixed_manifest_sha256": _sha256(Path(args.fixed_manifest).resolve()), "suite": args.suite, "task_ids": task_ids, "state_ids": state_ids, "repeats": args.repeats, "seed_base": args.seed_base, "order_seed": args.order_seed, "control_frequency_hz": args.control_freq, "shuffle_seed": args.shuffle_seed, "shuffle_strength_threshold": args.shuffle_strength_threshold, "repeatability_equivalence_margin": args.repeatability_equivalence_margin, "mujoco_gl": os.environ.get("MUJOCO_GL"), "require_exact_predicted_replay": args.require_exact_predicted_replay, "require_frozen_initial_pairing": args.require_frozen_initial_pairing, **source_hashes}
    resume_sha = canonical_json_sha256(resume_identity); runtime = {"protocol": "duration_v4_frozen_model_process_runtime_manifest", "determinism": _determinism_metadata(), "python_executable": sys.executable, "python_version": sys.version, "argv": sys.argv, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"), "mujoco_gl": os.environ.get("MUJOCO_GL")}; runtime_sha = canonical_json_sha256(runtime)
    manifests = {task_id: materialize_task_order_manifest(suite=args.suite, task_id=task_id, state_ids=state_ids, repeats=args.repeats, seed_base=args.seed_base, order_seed=args.order_seed) for task_id in task_ids}
    seen_env_seeds: dict[int, tuple[int, int, int]] = {}; seen_policy_seeds: dict[int, tuple[int, int, int]] = {}
    permutation_counts = {"/".join(order): 0 for order in EFFICACY_PERMUTATIONS}
    for task_id, manifest in manifests.items():
        for order, count in manifest["permutation_counts"].items(): permutation_counts[order] += count
        if max(manifest["permutation_counts"].values()) - min(manifest["permutation_counts"].values()) > 1: raise RuntimeError("per-task efficacy-order quotas are unbalanced")
        for spec in manifest["blocks"]:
            coordinate = (task_id, int(spec["state_id"]), int(spec["repeat_index"])); env_seed, policy_seed = int(spec["env_seed_u32"]), int(spec["policy_seed_u64"])
            if env_seed in seen_env_seeds or policy_seed in seen_policy_seeds: raise RuntimeError(f"block seed collision at {coordinate}")
            seen_env_seeds[env_seed] = coordinate; seen_policy_seeds[policy_seed] = coordinate
    all_blocks: list[dict[str, Any]] = []; tasks: list[dict[str, Any]] = []; runtime_manifests = {runtime_sha: runtime}; executed = resumed = 0; started = time.time()
    try:
        for ordinal, task_id in enumerate(task_ids):
            task_blocks: list[dict[str, Any]] = []; manifest = manifests[task_id]
            for spec in manifest["blocks"]:
                state_id, repeat = int(spec["state_id"]), int(spec["repeat_index"]); env_seed, policy_seed = int(spec["env_seed_u32"]), int(spec["policy_seed_u64"]); block_path = block_dir / f"task{task_id:02d}_state{state_id:04d}_repeat{repeat:03d}.json"
                if block_path.exists():
                    artifact = json.loads(block_path.read_text()); expected = {"schema_version": 1, "protocol": BLOCK_PROTOCOL, "resume_identity_sha256": resume_sha, "task_order_manifest_sha256": manifest["manifest_sha256"], "task_id": task_id, "state_id": state_id, "repeat_index": repeat}
                    if any(artifact.get(k) != v for k, v in expected.items()) or artifact.get("block_sha256") != canonical_json_sha256(artifact.get("block")): raise ValueError(f"incompatible frozen block resume: {block_path}")
                    prior_runtime, prior_runtime_sha = artifact.get("execution_runtime_manifest"), artifact.get("execution_runtime_sha256")
                    if not isinstance(prior_runtime, dict) or prior_runtime_sha != canonical_json_sha256(prior_runtime): raise ValueError(f"resumed runtime provenance mismatch: {block_path}")
                    block = artifact["block"]
                    if block.get("execution_runtime_sha256") != prior_runtime_sha: raise ValueError(f"resumed runtime binding mismatch: {block_path}")
                    if block.get("arm_order") != list(spec["arm_order"]) or set(block.get("episodes", {})) != set(ARM_LABELS): raise ValueError(f"resumed block is incomplete: {block_path}")
                    source = block.get("shuffle_source_manifest")
                    if not isinstance(source, dict): raise ValueError(f"resumed block lacks shuffle source: {block_path}")
                    validate_embedded_sha256(source)
                    if block["episodes"]["shuffled"].get("shuffle_source_manifest_sha256") != source["manifest_sha256"]: raise ValueError(f"resumed shuffle binding mismatch: {block_path}")
                    runtime_manifests[prior_runtime_sha] = prior_runtime; task_blocks.append(block); all_blocks.append(block); resumed += 1; continue
                env, built_envs = _make_block_env(make_env, env_cfg, args.suite, task_id)
                try:
                    max_steps = int(env.call("_max_episode_steps")[0]); task_description = str(next(iter(env.call("task_description"))))
                    captured_observation, realized = _capture_once(env, state_id, env_seed); frozen = FrozenLiberoState.capture(_base_env(env), captured_observation, _structured_sha256, wrapper_root=env)
                    if realized != state_id: raise RuntimeError("captured wrong requested state")
                    kwargs = {"env": env, "frozen": frozen, "task_id": task_id, "state_id": state_id, "repeat_index": repeat, "env_seed": env_seed, "policy_seed": policy_seed, "task_description": task_description, "max_steps": max_steps, "policy": policy, "controller": controller, "preprocessor": preprocessor, "postprocessor": postprocessor, "env_preprocessor": env_preprocessor, "env_postprocessor": env_postprocessor, "action_key": ACTION, "preprocess_observation": preprocess_observation}
                    donor = _run_frozen_episode(arm_label="predicted_donor", **kwargs)
                    source = {"protocol": "in_process_predicted_donor_frozen_model_shuffle_source_v4", "resume_identity_sha256": resume_sha, "checkpoint": str(checkpoint), "model_sha256": _sha256(model_path), "checkpoint_config_sha256": _sha256(config_path), "suite": args.suite, "task_order_manifest_sha256": manifest["manifest_sha256"], "task_id": task_id, "state_id": state_id, "repeat_index": repeat, "env_seed_u32": env_seed, "policy_seed_u64": policy_seed, "control_frequency_hz": args.control_freq, "shuffle_seed": args.shuffle_seed, "intervention_protocol": INTERVENTION_PROTOCOL, "frozen_reset_protocol": FROZEN_RESET_PROTOCOL, "captured_compiled_model_xml_sha256": frozen.compiled_model_xml_sha256, "captured_initial_mujoco_integration_state_sha256": frozen.integration_state_sha256, "captured_initial_raw_observation_sha256": frozen.raw_observation_sha256, "predicted_durations": donor["predicted_durations"], "donor_action_trace_sha256": donor["executed_action_trace_sha256"]}; source["manifest_sha256"] = canonical_json_sha256(source); validate_embedded_sha256(source)
                    episodes = {"predicted_donor": donor}
                    for arm in spec["efficacy_permutation"]: episodes[arm] = _run_frozen_episode(arm_label=arm, shuffle_source=donor["predicted_durations"] if arm == "shuffled" else None, shuffle_source_manifest_sha256=source["manifest_sha256"] if arm == "shuffled" else None, **kwargs)
                    episodes["predicted_closure"] = _run_frozen_episode(arm_label="predicted_closure", **kwargs)
                    if episodes["shuffled"]["shuffle_source_manifest_sha256"] != source["manifest_sha256"] or episodes["shuffled"]["counterfactual_source_length"] != len(donor["predicted_durations"]): raise RuntimeError("shuffle-source binding failed")
                    closure = episodes["predicted_closure"]; arm_order = list(spec["arm_order"])
                    block = {"execution_runtime_sha256": runtime_sha, "task_block_ordinal": spec["task_block_ordinal"], "task_order_manifest_sha256": manifest["manifest_sha256"], "task_id": task_id, "state_id": state_id, "repeat_index": repeat, "env_seed_u32": env_seed, "policy_seed_u64": policy_seed, "arm_order": arm_order, "efficacy_permutation": list(spec["efficacy_permutation"]), "frozen_capture": {"simulator_path": frozen.simulator_path, "compiled_model_xml_sha256": frozen.compiled_model_xml_sha256, "initial_mujoco_integration_state_sha256": frozen.integration_state_sha256, "initial_raw_observation_sha256": frozen.raw_observation_sha256}, "shuffle_source_manifest": source, "initial_pairing_by_arm": {arm: _initial_pairing(donor, episodes[arm]) for arm in arm_order[1:]}, "predicted_repeatability": {"donor_vs_eval": predicted_pair_diagnostic(donor, episodes["predicted_eval"]), "donor_vs_closure": predicted_pair_diagnostic(donor, closure), "eval_vs_closure": predicted_pair_diagnostic(episodes["predicted_eval"], closure)}, "episodes": episodes}
                    artifact = {"schema_version": 1, "protocol": BLOCK_PROTOCOL, "watermark": WATERMARK, "resume_identity_sha256": resume_sha, "task_order_manifest_sha256": manifest["manifest_sha256"], "task_id": task_id, "state_id": state_id, "repeat_index": repeat, "execution_runtime_sha256": runtime_sha, "execution_runtime_manifest": runtime, "block_sha256": canonical_json_sha256(block), "block": block}; atomic_write_json_new(block_path, artifact); executed += 1; task_blocks.append(block); all_blocks.append(block)
                finally:
                    for candidate in built_envs.values():
                        try: candidate.close()
                        except (RuntimeError, OSError): pass
            tasks.append({"task_id": task_id, "task_ordinal": ordinal, "task_order_manifest": manifest, "n_blocks": len(task_blocks), "blocks": task_blocks})
    finally:
        pass
    expected = len(task_ids) * len(state_ids) * args.repeats
    if len(all_blocks) != expected: raise RuntimeError("incomplete block grid")
    summaries = {arm: _arm_summary(all_blocks, arm) for arm in ARM_LABELS}; pairs = ("donor_vs_eval", "donor_vs_closure", "eval_vs_closure")
    repeatability = {pair: aggregate_predicted_pairs([block["predicted_repeatability"][pair] for block in all_blocks]) for pair in pairs}
    exact_replay = all(summary["n_exact_action_trace_agreements"] == expected and summary["n_exact_predicted_duration_trace_agreements"] == expected and summary["first_raw_observation_divergence_step"]["n"] == 0 and summary["first_processed_observation_divergence_step"]["n"] == 0 and summary["first_mujoco_integration_state_divergence_step"]["n"] == 0 for summary in repeatability.values())
    pairings = [pairing for block in all_blocks for pairing in block["initial_pairing_by_arm"].values()]
    pairing_summary = {"n_arm_episode_pairs": len(pairings), "n_all_frozen_initial_field_agreements": sum(item["all_frozen_initial_fields_agree"] for item in pairings), "n_seed_agreements": sum(item["seed_agreement"] for item in pairings), "n_raw_initial_observation_agreements": sum(item["raw_initial_observation_agreement"] for item in pairings), "n_processed_initial_observation_agreements": sum(item["processed_initial_observation_agreement"] for item in pairings), "n_mujoco_integration_initial_state_agreements": sum(item["mujoco_integration_initial_state_agreement"] for item in pairings), "n_compiled_model_xml_agreements": sum(item["compiled_model_xml_agreement"] for item in pairings), "n_frozen_restore_verifications": sum(item["frozen_restore_verified"] for item in pairings)}
    initial_complete = all(value == len(pairings) for key, value in pairing_summary.items() if key != "n_arm_episode_pairs")
    order_gate = {"protocol": "per_task_six_permutation_quota_balance_v3_preserved_v4", "passed": all(max(task["task_order_manifest"]["permutation_counts"].values()) - min(task["task_order_manifest"]["permutation_counts"].values()) <= 1 for task in tasks), "per_task_counts": {str(task["task_id"]): task["task_order_manifest"]["permutation_counts"] for task in tasks}}
    shuffled_tasks = {str(task["task_id"]): _arm_summary(task["blocks"], "shuffled") for task in tasks}; shuffle_overall = summaries["shuffled"]["intervention_call_fraction"] >= args.shuffle_strength_threshold; shuffle_taskwise = all(item["intervention_call_fraction"] >= args.shuffle_strength_threshold for item in shuffled_tasks.values()); shuffle_ok = shuffle_overall and shuffle_taskwise
    shuffle_gate = {"protocol": "shuffle_effective_changed_call_fraction_v3_preserved_v4", "target_minimum": args.shuffle_strength_threshold, "observed_changed_call_fraction": summaries["shuffled"]["intervention_call_fraction"], "passed": shuffle_ok, "passed_overall": shuffle_overall, "all_tasks_passed": shuffle_taskwise, "unused_counterfactual_source_values": summaries["shuffled"]["unused_counterfactual_source_values"], "reused_counterfactual_values": summaries["shuffled"]["reused_counterfactual_values"], "equal_source_multiset_fraction": summaries["shuffled"]["equal_counterfactual_source_multiset_fraction"], "taskwise": shuffled_tasks}
    predicted_rates = {arm: summaries[arm]["success_rate"] for arm in ("predicted_donor", "predicted_eval", "predicted_closure")}; task_rates = {str(task["task_id"]): {arm: _arm_summary(task["blocks"], arm)["success_rate"] for arm in predicted_rates} for task in tasks}; spreads = {task: max(rates.values()) - min(rates.values()) for task, rates in task_rates.items()}
    repeat_gate = {"protocol": "predicted_position_marginal_equivalence_target_v3_preserved_v4", "equivalence_margin": args.repeatability_equivalence_margin, "predicted_success_rates": predicted_rates, "observed_max_minus_min": max(predicted_rates.values()) - min(predicted_rates.values()), "pilot_observed_within_margin": max(predicted_rates.values()) - min(predicted_rates.values()) <= args.repeatability_equivalence_margin, "taskwise_predicted_success_rates": task_rates, "taskwise_observed_max_minus_min": spreads, "pilot_all_tasks_within_margin": all(value <= args.repeatability_equivalence_margin for value in spreads.values()), "confirmatory_95pct_ci_margin": .05}
    by_task_repeatability = {str(task["task_id"]): {pair: aggregate_predicted_pairs([block["predicted_repeatability"][pair] for block in task["blocks"]]) for pair in pairs} for task in tasks}
    result = {"schema_version": 4, "protocol": EVALUATION_PROTOCOL, "watermark": WATERMARK, "stochastic_result": True, "bitwise_replay_assumed": False, "frozen_model_paired_reset": True, "frozen_reset_protocol": FROZEN_RESET_PROTOCOL, "terminal_autoreset_protocol": TERMINAL_AUTORESET_PROTOCOL, "no_reset_after_block_capture": True, "exact_predicted_replay_required": args.require_exact_predicted_replay, "exact_predicted_replay_all_pairs": exact_replay, "frozen_initial_pairing_required": args.require_frozen_initial_pairing, "frozen_initial_pairing_complete": initial_complete, "scientific_arms": ["predicted_eval", "fixed", "shuffled"], "trace_only_arms": ["predicted_donor", "predicted_closure"], "primary_predicted_arm": "predicted_eval", "execution_scope": "one selected live LIBERO model per task/state/repeat block; terminal autoresets suppressed; no post-capture reset or XML reload", "block_execution": "predicted donor; one counterbalanced efficacy permutation; predicted closure", "counterbalance_protocol": COUNTERBALANCE_PROTOCOL, "order_seed": args.order_seed, "efficacy_permutations": [list(order) for order in EFFICACY_PERMUTATIONS], "efficacy_permutation_counts": permutation_counts, "block_seed_protocol": BLOCK_SEED_PROTOCOL, "duration_prediction_source": "bound original production policy._predicted_T(tokens) called once inside the common wrapper in every arm", "intervention_protocol": INTERVENTION_PROTOCOL, "fixed_duration": fixed_duration, "fixed_manifest": str(Path(args.fixed_manifest).resolve()), "fixed_manifest_sha256": _sha256(Path(args.fixed_manifest).resolve()), "fixed_manifest_protocol": fixed_payload.get("protocol"), "fixed_manifest_complete_training_scan": fixed_payload["complete_training_scan"], "checkpoint": str(checkpoint), "checkpoint_config_sha256": _sha256(config_path), "model_sha256": _sha256(model_path), "policy_type": config["type"], "suite": args.suite, "task_ids": task_ids, "state_ids": state_ids, "repeats": args.repeats, "seed_base": args.seed_base, "control_frequency_hz": args.control_freq, "n_blocks": expected, "n_episodes": expected * 5, "n_blocks_executed_this_process": executed, "n_blocks_resumed_atomically": resumed, "block_directory": str(block_dir.resolve()), "resume_identity": resume_identity, "resume_identity_sha256": resume_sha, "arm_summaries": summaries, "predicted_repeatability_all_pairs": repeatability, "predicted_repeatability_by_task": by_task_repeatability, "repeatability_equivalence_gate": repeat_gate, "order_balance_gate": order_gate, "shuffle_strength_gate": shuffle_gate, "initial_pairing_complete": initial_complete, "initial_pairing_summary": pairing_summary, "scientific_pilot_eligible": initial_complete and order_gate["passed"] and shuffle_gate["passed"] and repeat_gate["pilot_observed_within_margin"] and repeat_gate["pilot_all_tasks_within_margin"], "current_execution_runtime_sha256": runtime_sha, "execution_runtime_manifests": runtime_manifests, "determinism": runtime["determinism"], "elapsed_seconds": time.time() - started, "source_hashes": source_hashes, "tasks": tasks}
    atomic_write_json_new(output, result)
    print(f"saved frozen-model paired blocks {output}: blocks={expected} initial_pairing={initial_complete}", flush=True)
    if args.require_frozen_initial_pairing and not initial_complete: raise RuntimeError("frozen initial pairing gate failed; diagnostic result was published")
    if args.require_exact_predicted_replay and not exact_replay: raise RuntimeError("exact predicted replay was required but diverged; diagnostic result was published")


if __name__ == "__main__":
    main()
