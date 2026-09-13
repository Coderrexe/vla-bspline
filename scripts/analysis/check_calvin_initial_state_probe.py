"""Compare independent CALVIN reset probes and quantify pixel variation."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    assert len(args.artifacts) >= 2
    metadata = [json.loads(path.read_text()) for path in args.artifacts]
    arrays = [np.load(item["arrays"]) for item in metadata]
    reference = arrays[0]
    result = {"passed": True, "runs": len(arrays), "n_seq": metadata[0]["n_seq"],
              "comparisons_to_run0": []}
    assert all(item["n_seq"] == result["n_seq"] for item in metadata)
    for item, value in zip(metadata[1:], arrays[1:]):
        comparison = {"run_index": item["run_index"]}
        for key in ("robot", "scene", "top", "wrist"):
            delta = np.abs(reference[key].astype(float) - value[key].astype(float))
            comparison[key] = {"exact": bool(np.array_equal(reference[key], value[key])),
                               "mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
                               "fraction_nonzero": float(np.count_nonzero(delta) / delta.size)}
        # Pairing requires simulator state identity. Images may exhibit renderer noise,
        # which is measured rather than silently treated as identical.
        assert comparison["robot"]["exact"] and comparison["scene"]["exact"]
        reference_hashes = [record.get("packed_obs_sha256") for record in metadata[0]["records"]]
        value_hashes = [record.get("packed_obs_sha256") for record in item["records"]]
        comparison["packed_observation"] = {
            "available": all(value is not None for value in reference_hashes + value_hashes),
            "exact": reference_hashes == value_hashes,
            "matching": sum(left == right for left, right in zip(reference_hashes, value_hashes)),
            "total": len(reference_hashes),
        }
        if comparison["packed_observation"]["available"]:
            assert comparison["packed_observation"]["exact"]
        result["comparisons_to_run0"].append(comparison)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
