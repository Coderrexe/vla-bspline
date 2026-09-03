import unittest

from scripts.eval.robocasa_language_programs import (
    KETTLE_OFFICIAL_FIXED_SWITCH_STEPS,
    KETTLE_OFFICIAL_PROGRAM,
    KETTLE_OFFICIAL_SLOW_PLACE_SWITCH_STEPS,
    RINSE_REGION_PROGRAM,
    RINSE_OFFICIAL_FIXED_SWITCH_STEP,
    RINSE_OFFICIAL_PROGRAM,
    kettle_fixed_program_pointer,
    kettle_burner_instruction,
    kettle_goal_program,
    kettle_goal_program_advance,
    kettle_program_advance,
    rinse_official_fixed_pointer,
    rinse_official_program_advance,
    rinse_program_advance,
)


class RinseRegionProgramTest(unittest.TestCase):
    def test_advances_and_skips_already_complete_regions(self) -> None:
        sample = {
            "water_on": True,
            "washed_left": False,
            "washed_center": True,
        }
        self.assertEqual(rinse_program_advance(0, sample), 1)
        sample["washed_left"] = True
        self.assertEqual(rinse_program_advance(1, sample), 3)
        self.assertEqual(len(RINSE_REGION_PROGRAM), 4)

    def test_holds_until_current_predicate_is_complete(self) -> None:
        sample = {
            "water_on": False,
            "washed_left": True,
            "washed_center": True,
        }
        self.assertEqual(rinse_program_advance(0, sample), 0)
        self.assertEqual(rinse_program_advance(3, sample), 3)
        with self.assertRaisesRegex(ValueError, "invalid"):
            rinse_program_advance(4, sample)

    def test_official_two_phase_rinse_program(self) -> None:
        self.assertEqual(len(RINSE_OFFICIAL_PROGRAM), 2)
        self.assertEqual(RINSE_OFFICIAL_FIXED_SWITCH_STEP, 162)
        self.assertEqual(rinse_official_program_advance(0, {"water_on": False}), 0)
        self.assertEqual(rinse_official_program_advance(0, {"water_on": True}), 1)
        self.assertEqual(rinse_official_program_advance(1, {"water_on": False}), 1)
        self.assertEqual(rinse_official_fixed_pointer(161), 0)
        self.assertEqual(rinse_official_fixed_pointer(162), 1)


class KettleOfficialProgramTest(unittest.TestCase):
    def test_burner_instruction_matches_official_templates(self) -> None:
        self.assertEqual(
            kettle_burner_instruction("front_left"),
            "turn on the front-left burner where the kettle is placed",
        )
        self.assertEqual(
            kettle_burner_instruction("rear-right"),
            "turn on the rear-right burner where the kettle is placed",
        )
        with self.assertRaises(ValueError):
            kettle_burner_instruction("rear_center")

    def test_goal_consistent_program(self) -> None:
        self.assertEqual(
            kettle_goal_program("front_left"),
            [
                "pick up the kettle from the counter",
                "place the kettle on the front-left stove burner",
                "turn on the front-left burner where the kettle is placed",
            ],
        )
        sample = {
            "object_grasped": True,
            "object_stove_contact": True,
            "selected_burner_xy_distance": 0.1,
            "gripper_object_far": True,
        }
        self.assertEqual(kettle_goal_program_advance(0, sample), 2)

    def test_oracle_advances_on_grasp_then_placement(self) -> None:
        sample = {
            "object_grasped": False,
            "object_stove_contact": False,
            "kettle_near_burner": False,
            "gripper_object_far": False,
        }
        self.assertEqual(kettle_program_advance(0, sample), 0)
        sample["object_grasped"] = True
        self.assertEqual(kettle_program_advance(0, sample), 1)
        sample.update(
            object_grasped=False,
            object_stove_contact=True,
            kettle_near_burner=True,
            gripper_object_far=True,
        )
        self.assertEqual(kettle_program_advance(1, sample), 2)
        self.assertEqual(kettle_program_advance(2, sample), 2)
        self.assertEqual(len(KETTLE_OFFICIAL_PROGRAM), 3)

    def test_fixed_clock_uses_frozen_official_demo_medians(self) -> None:
        first, second = KETTLE_OFFICIAL_FIXED_SWITCH_STEPS
        self.assertEqual((first, second), (128, 242))
        self.assertEqual(kettle_fixed_program_pointer(0), 0)
        self.assertEqual(kettle_fixed_program_pointer(first - 1), 0)
        self.assertEqual(kettle_fixed_program_pointer(first), 1)
        self.assertEqual(kettle_fixed_program_pointer(second - 1), 1)
        self.assertEqual(kettle_fixed_program_pointer(second), 2)
        slow_first, slow_second = KETTLE_OFFICIAL_SLOW_PLACE_SWITCH_STEPS
        self.assertEqual((slow_first, slow_second), (128, 356))
        self.assertEqual(kettle_fixed_program_pointer(slow_first, slow_place=True), 1)
        self.assertEqual(kettle_fixed_program_pointer(slow_second - 1, slow_place=True), 1)
        self.assertEqual(kettle_fixed_program_pointer(slow_second, slow_place=True), 2)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            kettle_fixed_program_pointer(-1)


if __name__ == "__main__":
    unittest.main()
