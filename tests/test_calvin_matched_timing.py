"""Synthetic fixtures for fail-closed matrix validation and paired contrasts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from check_calvin_matched_timing import ARMS, check
from summarize_calvin_matched_timing import summarize


class MatchedTimingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, nchains=2, seeds=(1000,)):
        logs = []
        for seed in seeds:
            for arm in ARMS:
                run = self.root / f"{arm}_{seed}"
                run.mkdir()
                config = {"n_action_steps": 5,
                          "type": "smolvla" if arm == "A_native" else
                          "smolvla_interp" if arm == "A_linear" else "smolvla_spline",
                          "exec_horizon": 30, "chunk_size": 50, "min_seg": 8,
                          "horizon_max": 16, "exec_rate_ratio": 1,
                          "speedup_alpha": .6 if arm == "C_retime" else 1.,
                          "feasibility_stretch": False, "actuator_bound": 1.}
                source = self.root / f"calv{arm[0]}_100k_s{seed}" / "checkpoints/last/pretrained_model"
                (run / "view_manifest.json").write_text(json.dumps({
                    "arm": arm, "source": str(source), "config": config,
                    "files": {"synthetic_model": f"{arm[0]}_{seed}"}}))
                (run / "interpolation_gate.json").write_text(json.dumps({
                    "passed": True, "native_identity_exact": True}))
                records = [{"chain_index": index, "sequence": ["test"] * 5,
                            "solved": 0, "steps": [360], "that": [[-1] * 120],
                            "traces": [{"initial_obs_sha256": f"initial_{index}",
                                        "action_sha256": "0" * 64, "policy_calls": 24}]}
                           for index in range(nchains)]
                (run / "result.json").write_text(json.dumps({
                    "n_seq": nchains, "n_total": 1000, "seq_offset": 0,
                    "ep_len": 360, "repeat": 3, "policy_seed": 700000,
                    "random": False, "records": records, "avg_len": 0., "sr": [0.] * 5}))
                log = run / "slurm.out"
                log.write_text(f"RESULT {run / 'result.json'}\n")
                logs.append(log)
        return logs

    def test_accept_complete_paired_smoke(self):
        self.assertTrue(check(self.fixture(), 2, [1000])["passed"])

    def test_reject_incomplete(self):
        with self.assertRaises(AssertionError):
            check(self.fixture()[:-1], 2, [1000])

    def test_reject_mismatched_initialization(self):
        logs = self.fixture()
        path = logs[-1].parent / "result.json"
        data = json.loads(path.read_text())
        data["records"][1]["traces"][0]["initial_obs_sha256"] = "different"
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AssertionError, "initial observation mismatch"):
            check(logs, 2, [1000])

    def test_reject_mismatched_cadence(self):
        logs = self.fixture()
        path = logs[-1].parent / "result.json"
        data = json.loads(path.read_text())
        data["records"][0]["traces"][0]["policy_calls"] = 25
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(AssertionError, "neural-query cadence"):
            check(logs, 2, [1000])

    def test_summary_zero_contrasts_on_identical_outcomes(self):
        audit = check(self.fixture(50, (1000, 1001, 1002)), 50, [1000, 1001, 1002])
        summary = summarize(audit, bootstraps=100)
        for values in summary["contrasts"].values():
            self.assertEqual(values["mean"], 0.)
            self.assertEqual(values["seed_t95_df2"], [0., 0.])
            self.assertEqual(values["hierarchical_seed_and_paired_chain95"], [0., 0.])

    def test_summary_known_nonzero_contrasts(self):
        logs = self.fixture(50, (1000, 1001, 1002))
        for log in logs:
            path = log.parent / "result.json"
            data = json.loads(path.read_text())
            arm = json.loads((log.parent / "view_manifest.json").read_text())["arm"]
            solved = {"A_native": 1, "A_linear": 2, "C_native": 2, "C_retime": 4}[arm]
            for record in data["records"]:
                record["solved"] = solved
                record["steps"] = [15] * solved + [360]
                record["that"] = [[-1] * 5 for _ in range(solved)] + [[-1] * 120]
                record["traces"] = [dict(record["traces"][0], policy_calls=1)
                                    for _ in range(solved)] + record["traces"]
            data["avg_len"] = float(solved)
            data["sr"] = [float(solved >= depth) for depth in range(1, 6)]
            path.write_text(json.dumps(data))
        summary = summarize(check(logs, 50, [1000, 1001, 1002]), bootstraps=100)
        for key, expected in {"waypoint_retiming": 1., "spline_retiming": 2.,
                              "retimed_spline_minus_retimed_waypoint": 2.,
                              "retiming_interaction": 1.}.items():
            result = summary["contrasts"][key]
            self.assertEqual(result["per_seed"], [expected] * 3)
            self.assertEqual(result["seed_t95_df2"], [expected] * 2)
            self.assertEqual(result["hierarchical_seed_and_paired_chain95"], [expected] * 2)


if __name__ == "__main__":
    unittest.main()
