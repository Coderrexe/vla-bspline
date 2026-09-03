# Real-robot compositional-language plan

## Paper objective

Use three visually obvious tabletop tasks to test the same claim as the
simulation holdout: executable clauses let a policy recombine known physical
substructures and align a language change with the part of the trajectory that
should change. Hardware is not only a spline deployment demo.

The primary setup is deliberately arm-agnostic: one fixed external camera, one
wrist camera if already available, a parallel gripper, a matte tabletop, soft
objects, and high-contrast target zones. Do not purchase or integrate new
hardware until the lab confirms the available arm, teleoperator, cameras,
control mode, and safety interface.

## Three tasks

| priority | task | training programs | held-out test | main metric |
|---|---|---|---|---|
| P1 | two-object kitting | A→left then B→right; B→left then C→right; C→left then A→right | A→left then C→right, plus its reversed order | both placements correct and requested first object |
| P2 | place + button | object A/B to one marked pad paired with green/blue button presses | a held-out object-button pairing and reversed action order | placement, correct button, requested order |
| P3 | drawer sequence | open drawer; insert A or B; close drawer, with clauses also demonstrated separately | held-out object plus final intervention “close” versus “leave open” | object inside and final drawer state |

P1 is the must-have result. Every primitive clause in its held-out program is
seen during training, but that pair/order is not. The three objects remain
visible in every episode; initial poses are sampled from small taped regions so
the result is not a fixed-position shortcut.

P2 provides a second interaction type without adding difficult contact-rich
insertion. P3 is the long-horizon stretch task and the cleanest later-clause
intervention: two programs share “open the drawer; put object A inside” and
differ only in “close the drawer” versus “leave the drawer open.” Their image,
state, and action prefixes should agree until the final clause becomes active,
then diverge.

## Factorial and data

For P1, collect the physical demonstrations once and derive paired language
views exactly as in simulation:

| head | whole instruction | executable clauses |
|---|---|---|
| waypoint | A-whole | A-clause |
| spline | C-whole | C-clause |

Start with 30 successful demonstrations per seen P1 program (90 total), balanced
across object-pose bins and execution order. Reserve fixed validation episodes
from the seen programs; never use held-out-program robot demonstrations for
checkpoint selection. If seen-program success is below 70%, add demonstrations
in balanced blocks of 10 per program rather than changing only the weakest
treatment. For P2/P3, begin with 20 per seen program and expand only after P1's
pipeline is stable.

Record language/clause ID at every frame, clause transition time, raw command,
executed state, timestamps, both camera streams, success annotations, and reset
configuration. Keep the same physical episode IDs for the whole/clause dataset
pair. LeRobot's current recorder supports calibrated teleoperation, cameras,
resumable collection, and local LeRobotDataset storage; use the lab's existing
robot adapter rather than starting a new control stack. [LeRobot real-robot
guide](https://huggingface.co/docs/lerobot/main/getting_started_real_world_robot)

## Evaluation

Freeze 30 initial configurations per task before evaluating any checkpoint.
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
