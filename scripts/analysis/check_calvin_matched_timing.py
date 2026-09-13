"""Validate a complete matched-clock matrix before interpreting its outcomes.

Read-only; accepts Slurm logs whose final RESULT line names the saved artifact.
Writes its compact audit to stdout, so the caller can preserve it separately.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


ARMS = ("A_native", "A_linear", "C_native", "C_retime")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_paths_from_logs(logs):
    paths = []
    for log in logs:
        result_lines = [line[7:] for line in log.read_text().splitlines()
                        if line.startswith("RESULT ")]
        assert len(result_lines) == 1, (str(log), "missing or duplicate RESULT")
        paths.append(Path(result_lines[0]))
    return paths


def validate_reset_audit(path):
    audit = json.loads(path.read_text())
    assert audit["passed"] and audit["n_seq"] == 50 and audit["runs"] >= 2
    for comparison in audit["comparisons_to_run0"]:
        for key in ("robot", "scene", "top", "wrist"):
            assert comparison[key]["exact"], (str(path), comparison["run_index"], key)
        packed = comparison.get("packed_observation")
        assert packed and packed["available"] and packed["exact"], (
            str(path), comparison["run_index"], "packed observation")
    return {"path": str(path), "sha256": digest(path), "runs": audit["runs"],
            "n_seq": audit["n_seq"], "all_components_exact": True,
            "packed_policy_requests_exact": True}


def check_results(paths, nchains, seeds, reset_audit=None):
    cells = {}
    initializations = {}
    sequences = {}
    legacy_hash_matches = {}
    reset_certificate = validate_reset_audit(reset_audit) if reset_audit else None
    for path in paths:
        path = Path(path)
        result = json.loads(path.read_text())
        manifest = json.loads((path.parent / "view_manifest.json").read_text())
        gate = json.loads((path.parent / "interpolation_gate.json").read_text())
        assert gate["passed"] and gate["native_identity_exact"]
        config = manifest["config"]
        arm = manifest["arm"]
        source_name = Path(manifest["source"]).parents[2].name
        seed = int(source_name.rsplit("_s", 1)[1]) if "_s" in source_name else 1000
        assert arm in ARMS and seed in seeds, (arm, seed, source_name)
        key = (seed, arm)
        assert key not in cells, key
        for field, expected in {"n_seq": nchains, "n_total": 1000, "seq_offset": 0,
                                "ep_len": 360, "repeat": 3, "policy_seed": 700000,
                                "random": False}.items():
            assert result[field] == expected, (key, field, result[field])
        assert config["n_action_steps"] == 5
        assert config["type"] == {"A_native": "smolvla", "A_linear": "smolvla_interp",
                                  "C_native": "smolvla_spline", "C_retime": "smolvla_spline"}[arm]
        if arm == "A_linear":
            assert config["exec_horizon"] == 30 and config["chunk_size"] == 50
        if arm.startswith("C"):
            assert config["min_seg"] == 8 and config["horizon_max"] == 16
            assert config["exec_rate_ratio"] == 1
            assert config["speedup_alpha"] == (.6 if arm == "C_retime" else 1.)
        if arm != "A_native":
            assert not config["feasibility_stretch"] and config["actuator_bound"] == 1.
        assert len(result["records"]) == nchains
        for index, record in enumerate(result["records"]):
            assert record["chain_index"] == index
            assert len(record["traces"]) == len(record["steps"]) == len(record["that"])
            assert len(record["steps"]) == min(record["solved"] + 1, 5)
            if record["solved"] < 5:
                assert record["steps"][-1] == 360, (key, index, "truncated failed subtask")
            first_hash = record["traces"][0]["initial_obs_sha256"]
            initializations.setdefault(index, first_hash)
            matches = first_hash == initializations[index]
            legacy_hash_matches.setdefault(key, []).append(matches)
            if reset_certificate is None:
                assert matches, (key, index, "initial observation mismatch")
            sequences.setdefault(index, record["sequence"])
            assert record["sequence"] == sequences[index], (key, index, "sequence mismatch")
            for steps, that, trace in zip(record["steps"], record["that"], record["traces"]):
                assert 1 <= steps <= 360
                assert len(that) == math.ceil(steps / 3), (key, index, "issued-action count")
                assert trace["policy_calls"] == math.ceil(len(that) / 5), (
                    key, index, "neural-query cadence", trace["policy_calls"], len(that))
                assert len(trace["action_sha256"]) == 64
        mean = sum(record["solved"] for record in result["records"]) / nchains
        assert abs(result["avg_len"] - mean) < 1e-12
        expected_sr = [sum(r["solved"] >= depth for r in result["records"]) / nchains
                       for depth in range(1, 6)]
        assert vectors_close(result["sr"], expected_sr)
        cells[key] = {"result": str(path), "result_sha256": digest(path),
                      "manifest_sha256": digest(path.parent / "view_manifest.json"),
                      "checkpoint_files": manifest["files"], "avg_len": mean}
    assert set(cells) == {(seed, arm) for seed in seeds for arm in ARMS}, "incomplete matrix"
    for seed in seeds:
        for native, retimed in [("A_native", "A_linear"), ("C_native", "C_retime")]:
            assert cells[seed, native]["checkpoint_files"] == cells[seed, retimed]["checkpoint_files"]
    return {"passed": True, "nchains_per_cell": nchains, "training_seeds": seeds,
            "reset_certificate": reset_certificate,
            "legacy_packed_hash_matches": {
                f"{arm}_s{seed}": {"matching": sum(values), "total": len(values)}
                for (seed, arm), values in sorted(legacy_hash_matches.items())},
            "legacy_packed_hashes_all_match": all(
                all(values) for values in legacy_hash_matches.values()),
            "five_actions_per_query_verified": True,
            "cells": {f"{arm}_s{seed}": cell for (seed, arm), cell in sorted(cells.items())}}


def check(logs, nchains, seeds):
    """Backward-compatible strict gate for RESULT-bearing Slurm logs."""
    return check_results(result_paths_from_logs(logs), nchains, seeds)


def vectors_close(left, right):
    """Scalar-list comparison without a NumPy dependency for the audit gate."""
    return len(left) == len(right) and all(abs(a - b) < 1e-12 for a, b in zip(left, right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--nchains", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.logs, args.nchains, args.seeds), indent=2))


if __name__ == "__main__":
    main()
