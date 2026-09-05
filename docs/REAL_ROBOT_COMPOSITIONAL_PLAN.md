# Real-robot compositional-language plan

## Confirmed platform and ICRA scope (3 September 2026)

The photographed Yale setup contains two 7-DoF xArm7 manipulators on independent
1-DoF linear actuators, an existing handheld/kinesthetic teleoperation interface, a
camera-equipped arm, a gripper-equipped arm, and the lab's 3D-printed non-cuboid
stacking and drawer fixtures. The active manipulator therefore has up to eight DoF
when its rail is enabled. Keep the rail fixed unless the required workspace cannot be
reached without it; avoiding unnecessary base motion lowers collection and control
risk.

This platform and fixture family appear in the accepted ICRA 2026 hPGA-DP paper.
That paper used two real tasks, 200 demonstrations per task, and approximately one
page for hardware. ACG (ICRA 2026) used two tasks with 50/40 demonstrations and
10 trials repeated three times; ITPS (ICRA 2025) used two kitchen skills with 60
demonstrations each. For this paper, **two focused tasks and 0.65--0.9 page are in
line with accepted work** if the physical experiment directly tests composition,
alignment, and timing rather than merely showing deployment.

Before collection, confirm which robot is the active gripper, which camera streams
are recordable by LeRobot, whether the handle provides Cartesian pose and gripper
buttons, the commanded action space, and whether the hPGA task recorder/controller
can be reused.

## Paper objective

Use two visually obvious tabletop tasks, plus an optional timing condition on an
existing checkpoint, to test the same claim as the simulation holdout: executable
clauses let a policy recombine known physical substructures and align a language
change with the part of the trajectory that should change. Hardware is not only a
spline deployment demo.

The primary setup is deliberately arm-agnostic: one fixed external camera, one
wrist camera if already available, a parallel gripper, a matte tabletop, soft
objects, and high-contrast target zones. Do not purchase or integrate new
hardware until the lab confirms the available arm, teleoperator, cameras,
control mode, and safety interface.

## Two-task minimum and one stretch task

| priority | task | training programs | held-out test | main metric |
|---|---|---|---|---|
| P1 | printed drawer program | open drawer; insert a red object; close drawer, with the component clauses represented separately | matched later clause: close versus leave open, and a held-out compatible object/order if available | object inserted, final drawer state, prefix similarity and late divergence |
| P2 | printed non-cuboid stack/program | two physically valid primitive placements and demonstrated two-step programs | one held-out valid composition or order | both placements correct and requested first primitive |
| P3 | same learned task under timing/rate change | native-rate demonstrations only | language pace or second deploy rate on the same checkpoint | success, completion time, tracking/jerk |

P1 is the must-have result because it can reuse the drawer infrastructure from the
prior lab paper while asking a new question. The matched pair begins from the same
scene and executes the same `open drawer; insert object` prefix, then differs only in
`close the drawer` versus `leave the drawer open`. This gives a clean trajectory-level
alignment test even if a more ambitious held-out composition is not ready.

P2 is used only if the printed pieces admit more than one physically stable order or
composition. If their geometry dictates one unique order, do not manufacture an
invalid generalization split: use stacking as the second task for pace/control-rate
transfer instead. P3 is then an extra condition on an existing checkpoint, not a third
collection campaign.

## Factorial and data

For P1, collect the physical demonstrations once and derive paired language
views exactly as in simulation:

| head | whole instruction | executable clauses |
|---|---|---|
| waypoint | A-whole | A-clause |
| spline | C-whole | C-clause |

Start with 40--60 successful demonstrations per seen P1 program, balanced
across object-pose bins and execution order. Reserve fixed validation episodes
from the seen programs; never use held-out-program robot demonstrations for
checkpoint selection. If seen-program success is below 70%, add demonstrations
in balanced blocks of 10 per program rather than changing only the weakest
treatment. Because whole and clause datasets are two language views of the same
physical episodes, this collection supports all four A/C × whole/clause cells. For
P2, begin only after P1's recorder, training, and rollout loop is stable.

Record language/clause ID at every frame, clause transition time, raw command,
executed state, timestamps, both camera streams, success annotations, and reset
configuration. Keep the same physical episode IDs for the whole/clause dataset
pair. LeRobot's current recorder supports calibrated teleoperation, cameras,
resumable collection, and local LeRobotDataset storage; use the lab's existing
robot adapter rather than starting a new control stack. [LeRobot real-robot
guide](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot)

## Evaluation

Freeze at least 10 and preferably 20 initial configurations per task before evaluating
any checkpoint; 20 gives a much more useful paired estimate while remaining realistic
under the deadline.
Interleave model/treatment order within each session, reset from photographed
templates, and log every attempt. Primary P1 outcomes are full held-out-program
success, first-subgoal correctness, and retention of the first placement.
Report paired discordances because every treatment sees the same reset grid.

For the matched later-clause intervention, run paired prompts from each initial
configuration. Measure end-effector/action similarity before the clause switch,
divergence afterward, and whether only the requested terminal state changes.
The strongest visual is a synchronized pair of videos with a shared early
trajectory and different late outcomes.

Secondary timing metrics remain useful: completion time, command/executed jerk,
replanning latency, and success under a second control rate. They connect the
hardware section to the established retiming results without displacing the
composition/alignment claim.

## Backbone plan

SmolVLA is the fastest route for the controlled A/C head comparison. In
parallel, stage one pretrained-backbone transfer after the robot/data schema is
working. The official OpenPI repository provides a small custom-DROID
fine-tuning path using LeRobot data and a π0.5-DROID checkpoint; it should be
used only if the lab platform/action representation is compatible, since action
conversion and normalization are robot-specific. [OpenPI custom-DROID
fine-tuning](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md)

## First 48 hours after hardware access

1. Confirm arm, gripper, teleop device, camera IDs, control rate, action space,
   workspace limits, E-stop, and collision bounds.
2. Record and replay five calibration episodes; verify timestamps, action/state
   signs, cameras, gripper convention, and safe policy stop.
3. Collect a 10-episode P1 pilot balanced across the three seen programs.
4. Train one waypoint and one spline smoke model and evaluate only seen programs.
5. If both can exceed 50% seen success, begin the paired 90-demo collection;
   otherwise fix the data/control interface before scaling collection.

No liquids, sharp tools, human contact, high-speed motions, or unattended robot
execution are needed for this study.
