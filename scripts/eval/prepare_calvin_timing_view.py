"""Create an isolated, no-clobber config view; never modify a trained checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--arm", required=True, choices=["A_native", "A_linear", "C_native", "C_retime"])
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    assert config["type"] == ("smolvla" if args.arm.startswith("A") else "smolvla_spline")
    config["n_action_steps"] = 5
    config["pretrained_path"] = str(args.destination.resolve())
    if args.arm == "A_linear":
        config.update(type="smolvla_interp", exec_horizon=30,
                      feasibility_stretch=False, actuator_bound=1.0)
        assert config["chunk_size"] == 50
    elif args.arm.startswith("C"):
        assert config["min_seg"] == 8 and config["horizon_max"] == 16
        config.update(speedup_alpha=.6 if args.arm == "C_retime" else 1.,
                      speedup_T_threshold=0, exec_rate_ratio=1.,
                      feasibility_stretch=False, actuator_bound=1.0,
                      replan_frac=None, replan_margin=None)
    args.destination.mkdir(parents=True, exist_ok=False)
    manifest = {"arm": args.arm, "source": str(source), "files": {}}
    for path in sorted(source.iterdir()):
        if path.name == "config.json" or not path.is_file():
            continue
        (args.destination / path.name).symlink_to(path)
        with path.open("rb") as stream:
            manifest["files"][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    with (args.destination / "config.json").open("x") as stream:
        json.dump(config, stream, indent=2)
    manifest["config"] = config
    with (args.destination.parent / "view_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)


if __name__ == "__main__":
    main()
