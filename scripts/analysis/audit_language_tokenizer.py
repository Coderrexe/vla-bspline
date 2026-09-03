"""Audit whether a semantic-caption control survives the exact policy tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids(tokenizer: Any, text: str, max_length: int | None) -> list[int]:
    kwargs = {"padding": False, "add_special_tokens": True}
    if max_length is None:
        kwargs["truncation"] = False
    else:
        kwargs.update(truncation=True, max_length=max_length)
    encoded = tokenizer(text + "\n", **kwargs)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one token sequence")
        encoded = encoded[0]
    return [int(value) for value in encoded]


def audit_pairs(pairs: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    collisions = []
    truncated_pairs = 0
    token_jaccards = []
    rows = []
    for pair in pairs:
        original_full = _ids(tokenizer, str(pair["original_label"]), None)
        shuffled_full = _ids(tokenizer, str(pair["shuffled_label"]), None)
        original = _ids(tokenizer, str(pair["original_label"]), max_length)
        shuffled = _ids(tokenizer, str(pair["shuffled_label"]), max_length)
        original_truncated = len(original_full) > max_length
        shuffled_truncated = len(shuffled_full) > max_length
        truncated_pairs += int(original_truncated or shuffled_truncated)
        collision = original == shuffled
        if collision:
            collisions.append(int(pair["task_index"]))
        union = set(original) | set(shuffled)
        jaccard = len(set(original) & set(shuffled)) / len(union) if union else 1.0
        token_jaccards.append(jaccard)
        rows.append(
            {
                "task_index": int(pair["task_index"]),
                "original_token_count": len(original_full),
                "shuffled_token_count": len(shuffled_full),
                "original_truncated": original_truncated,
                "shuffled_truncated": shuffled_truncated,
                "token_id_collision": collision,
                "token_id_jaccard": jaccard,
            }
        )
    return {
        "pair_count": len(pairs),
        "token_id_collision_count": len(collisions),
        "token_id_collision_task_indices": collisions,
        "pairs_with_any_truncation": truncated_pairs,
        "mean_token_id_jaccard": sum(token_jaccards) / len(token_jaccards) if token_jaccards else None,
        "max_token_id_jaccard": max(token_jaccards) if token_jaccards else None,
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping_report", type=Path, required=True)
    parser.add_argument(
        "--tokenizer_name",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.max_length <= 0:
        raise ValueError("max_length must be positive")

    from transformers import AutoTokenizer

    mapping = json.loads(args.mapping_report.read_text())
    if mapping.get("control_kind") != "same-support-caption-derangement":
        raise ValueError("mapping report is not the expected semantic control")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    result = audit_pairs(mapping["mapping"], tokenizer, args.max_length)
    vocab_digest = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()
    ).hexdigest()
    result.update(
        schema_version=1,
        protocol="semantic_shuffle_policy_tokenizer_audit_v1",
        mapping_report=str(args.mapping_report.resolve()),
        mapping_report_sha256=sha256(args.mapping_report),
        tokenizer_name=args.tokenizer_name,
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_commit=getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        tokenizer_vocab_sha256=vocab_digest,
        max_length=args.max_length,
        prompt_suffix="newline",
        auditor_sha256=sha256(Path(__file__).resolve()),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(
        f"tokenizer collisions={result['token_id_collision_count']}/"
        f"{result['pair_count']}; truncated_pairs={result['pairs_with_any_truncation']}"
    )


if __name__ == "__main__":
    main()
