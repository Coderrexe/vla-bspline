# RoboCasa port plan (Xiatao T1: 5 tasks, benchmark #3)

Recon done July 18. Everything needed exists; est. 2-3 days elapsed.

## What's already in place

- Our LeRobot tree ships a **RoboCasa365 env wrapper** (`lerobot/envs/robocasa.py`,
  PandaOmron: state 16 = base_pos+base_quat+ee_rel+gripper, action 12 =
  base_motion(4)+control_mode(1)+ee_pos(3)+ee_rot(3)+gripper(1), cameras
  agentview_left / eye_in_hand / agentview_right) → **stock lerobot_eval works**,
  no custom harness like CALVIN needed.
- **Dataset chosen: `ember-lab-berkeley/robocasa365-target-atomic`** — LeRobot
  v3.0 (no conversion!), fps 20, 9,126 episodes, 231 atomic tasks, schema
  exactly matching the wrapper (verified July 18). Backup: the pretrain-atomic
  sibling (7,356 eps) for more data.
- robosuite 1.4 + mujoco already in the lerobot conda env. Missing only the
  `robocasa` package + kitchen assets (lightwheel pack; wrapper is already
  configured for `obj_registries=["lightwheel"]`).

## Task selection (pre-registered rule)

5 atomic tasks: prefer overlap with BSP's published RoboCasa cells
(sink faucet / coffee button / microwave / close door — POSITIONING_BSP.md)
where present in the 231, filled to 5 by episode-count ranking. Decide from
`annotation.human.task_name` value counts at download time.

## The one modeling extension: action-layout mapping

Our head assumes action[:6] = EE pose deltas, action[6] = gripper. RoboCasa:
pose at dims 5-10, gripper at 11, base+mode at 0-4. Add config fields
(`pose_slice_start=5`, `grip_index=11`, `passthrough_dims=[0..4]`) —
passthrough channels are fitted with the same gripper-style low-order curve
(they're near-constant for fixed-base atomic tasks). Self-test: layout
round-trip on a synthetic 12-dim chunk. Keep LIBERO/CALVIN defaults untouched.

## Steps

1. `pip install robocasa` (+ `download_kitchen_assets --type objs_lw`) in the
   lerobot env; smoke the env wrapper on 1 episode (CPU render).
2. Download target-atomic; task histogram → pick 5; offline gate
   (`calvin_offline_gate.py` generalized): event stats, fit error vs jitter
   floor, choose n_ctrl/cap; generate v2 stats JSON.
3. Action-layout extension + self-test (½ day).
4. Train calvA-style pair (A waypoint + C spline) on the 5 tasks, 20k smoke →
   100k. Spline arm ETA ~6h/100k by CALVIN scaling.
5. Eval: lerobot_eval on the wrapper, n=50/task × 5 tasks × 3 seeds eventually;
   paired seeds/init-conditions per EVAL_DESIGN.md; report success + completion
   time (BSP metrics) → direct comparison table vs their published cells.
6. Retiming ladder spot-check (uniform + interval) — RoboCasa demos are
   MimicGen-synthesized + human; dead-time census tells us which regime to
   expect BEFORE evaluating (prediction registered in advance = stronger
   paper claim).
