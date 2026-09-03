"""Pure instruction-program helpers for RoboCasa diagnostic evaluation."""

from __future__ import annotations

from typing import Any


RINSE_REGION_PROGRAM = [
    "Turn on the sink faucet.",
    "Move the faucet spout to wash the left side of the sink basin.",
    "Move the faucet spout to wash the center of the sink basin.",
    "Move the faucet spout to wash the right side of the sink basin.",
]

RINSE_OFFICIAL_PROGRAM = [
    "turn on the sink",
    "move the spout to wash all locations of the sink basin",
]
RINSE_OFFICIAL_FIXED_SWITCH_STEP = 162

# These strings are the canonical phase vocabulary in RoboCasa's official
# July-2026 KettleBoiling annotations. The archive uses a fixture-specific
# burner name; at test time KettleBoiling permits any burner, so the final
# clause refers to the visually selected burner where the policy placed it.
KETTLE_OFFICIAL_PROGRAM = [
    "pick up the kettle from the counter",
    "place the kettle on the stove burner",
    "turn on the burner where the kettle is placed",
]

# Medians from all 501 official demonstrations: 128 pick frames followed by
# 114 place frames. These constants were frozen before policy evaluation.
KETTLE_OFFICIAL_FIXED_SWITCH_STEPS = (128, 242)
KETTLE_OFFICIAL_SLOW_PLACE_SWITCH_STEPS = (128, 356)

KETTLE_TRAINED_BURNER_LOCATIONS = (
    "front-left",
    "front-right",
    "left",
    "rear-left",
    "rear-right",
    "center",
    "front-center",
)


def kettle_burner_instruction(location: str) -> str:
    """Return the exact official phase template for a selected burner."""

    if not isinstance(location, str) or not location.strip():
        raise ValueError("burner location must be a nonempty string")
    canonical = location.strip().lower().replace("_", "-").replace(" ", "-")
    if canonical not in KETTLE_TRAINED_BURNER_LOCATIONS:
        raise ValueError(f"unsupported official Kettle burner location: {location!r}")
    return f"turn on the {canonical} burner where the kettle is placed"


def kettle_goal_program(location: str) -> list[str]:
    """Return a goal-consistent Kettle program using trained vocabulary."""

    burner_instruction = kettle_burner_instruction(location)
    canonical = burner_instruction.removeprefix("turn on the ").removesuffix(
        " burner where the kettle is placed"
    )
    return [
        "pick up the kettle from the counter",
        f"place the kettle on the {canonical} stove burner",
        burner_instruction,
    ]


def kettle_goal_program_advance(pointer: int, sample: dict[str, Any]) -> int:
    """Oracle upper-bound transition using the selected, not any, burner."""

    if not 0 <= pointer < 3:
        raise ValueError(f"invalid goal-consistent Kettle pointer: {pointer}")
    if pointer == 0 and bool(sample["object_grasped"]):
        pointer = 1
    if pointer == 1 and bool(
        sample["object_stove_contact"]
        and sample["selected_burner_xy_distance"] < 0.15
        and sample["gripper_object_far"]
    ):
        pointer = 2
    return pointer


def rinse_program_advance(pointer: int, sample: dict[str, Any]) -> int:
    """Return the next Rinse-region clause under predicate-completion gating.

    Simulator predicates make this an oracle mechanism probe, not a deployable
    scheduler. Multiple already-complete phases are skipped deterministically.
    """

    if not 0 <= pointer < len(RINSE_REGION_PROGRAM):
        raise ValueError(f"invalid Rinse program pointer: {pointer}")
    predicates = ("water_on", "washed_left", "washed_center")
    while pointer < len(predicates) and bool(sample[predicates[pointer]]):
        pointer += 1
    return pointer


def rinse_official_program_advance(pointer: int, sample: dict[str, Any]) -> int:
    """Advance official Rinse prerequisite phases using simulator water state."""

    if not 0 <= pointer < len(RINSE_OFFICIAL_PROGRAM):
        raise ValueError(f"invalid official Rinse pointer: {pointer}")
    return 1 if pointer == 0 and bool(sample["water_on"]) else pointer


def rinse_official_fixed_pointer(step: int) -> int:
    """Official Rinse program under the frozen median demonstration clock."""

    if step < 0:
        raise ValueError("step must be non-negative")
    return int(step >= RINSE_OFFICIAL_FIXED_SWITCH_STEP)


def kettle_program_advance(pointer: int, sample: dict[str, Any]) -> int:
    """Advance the official Kettle program under simulator predicates.

    This is a diagnostic mechanism upper bound, not a deployable scheduler:
    grasp completion advances pick->place, and the benchmark's placement
    predicates advance place->burner.
    """

    if not 0 <= pointer < len(KETTLE_OFFICIAL_PROGRAM):
        raise ValueError(f"invalid Kettle program pointer: {pointer}")
    if pointer == 0 and bool(sample["object_grasped"]):
        pointer = 1
    if pointer == 1 and bool(
        sample["object_stove_contact"]
        and sample["kettle_near_burner"]
        and sample["gripper_object_far"]
    ):
        pointer = 2
    return pointer


def kettle_fixed_program_pointer(step: int, *, slow_place: bool = False) -> int:
    """Return the non-privileged Kettle clause at a frozen demo-median clock."""

    if step < 0:
        raise ValueError("step must be non-negative")
    first, second = (
        KETTLE_OFFICIAL_SLOW_PLACE_SWITCH_STEPS
        if slow_place
        else KETTLE_OFFICIAL_FIXED_SWITCH_STEPS
    )
    if step < first:
        return 0
    if step < second:
        return 1
    return 2
