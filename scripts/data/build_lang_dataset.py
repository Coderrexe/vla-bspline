"""Build the C-lang derived dataset (Xiatao T2): LIBERO with Molmo segment
labels as per-frame task strings on the object suite.

Single-variable design: full-40-task copy of HuggingFaceVLA/libero where ONLY
object-suite frames have task_index remapped to their segment's Molmo label
(appended to the tasks table); all other frames keep the original suite-level
string. Training against the existing Cn8 recipe then isolates language
granularity as the sole delta. Videos + untouched metadata are hardlinked.

  python build_lang_dataset.py \
      --labels ~/scratch/vla_bspline/molmo_libero_object/seg_labels.jsonl \
      --out ~/scratch/vla_bspline/libero_lang
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="HuggingFaceVLA/libero")
    ap.add_argument("--mix", type=float, default=0.0,
                    help="probability a segment KEEPS the original task string (mixed conditioning)")
    ap.add_argument("--adverbs", action="store_true",
                    help="prefix labels with truthful speed adverbs (language-commanded time)")
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo)          # resolves hub cache root
    src = str(ds.root)
    out = os.path.expanduser(args.out)
    labels = [json.loads(l) for l in open(os.path.expanduser(args.labels))]
    print(f"src {src}\n{len(labels)} segment labels")

    if args.adverbs:
        # language-commanded TIME: prefix each label with a truthful speed
        # adverb from the segment's mean pose-delta speed, terciled WITHIN its
        # original task (adverb = relative pace on comparable motions). The
        # policy then learns adverb -> duration/speed coupling — steerable at
        # test time through language alone, an axis only the duration head has.
        epf0 = sorted(glob.glob(os.path.join(src, "meta", "episodes", "**", "*.parquet"), recursive=True))
        em = pd.concat([pd.read_parquet(f) for f in epf0])
        lo_of = dict(zip(em["episode_index"], em["dataset_from_index"]))
        need_eps = {r["episode_index"] for r in labels}
        ep_speed: dict[int, np.ndarray] = {}
        for pqf in sorted(glob.glob(os.path.join(src, "data", "**", "*.parquet"), recursive=True)):
            df = pd.read_parquet(pqf, columns=["action", "episode_index"])
            hit = set(df["episode_index"].unique()) & need_eps
            for ep in hit:
                a = np.stack(df[df["episode_index"] == ep]["action"].to_numpy())
                ep_speed[ep] = np.linalg.norm(a[:, :6], axis=1)
        for r in labels:
            sp = ep_speed[r["episode_index"]][r["seg_start"]: r["seg_end"]]
            r["_speed"] = float(sp.mean()) if len(sp) else 0.0
        import collections
        by_task_sp = collections.defaultdict(list)
        for r in labels:
            by_task_sp[r["task"]].append(r["_speed"])
        for r in labels:
            v = np.array(by_task_sp[r["task"]])
            lo_t, hi_t = np.quantile(v, [1 / 3, 2 / 3])
            if r["_speed"] >= hi_t:
                r["label"] = "quickly " + r["label"]
            elif r["_speed"] <= lo_t:
                r["label"] = "slowly and carefully " + r["label"]
        n_q = sum(1 for r in labels if r["label"].startswith("quickly"))
        n_s = sum(1 for r in labels if r["label"].startswith("slowly"))
        print(f"adverbs: {n_q} quickly / {n_s} slowly / {len(labels)-n_q-n_s} plain")

    # label id table: original tasks first (ids preserved), unique labels appended
    tasks_pq = pd.read_parquet(os.path.join(src, "meta", "tasks.parquet"))
    orig_tasks = list(tasks_pq.index) if tasks_pq.index.dtype == object else list(tasks_pq.iloc[:, 0])
    uniq = sorted({r["label"] for r in labels})
    label_id = {s: len(orig_tasks) + i for i, s in enumerate(uniq)}
    print(f"{len(orig_tasks)} original tasks + {len(uniq)} unique labels")

    # per-episode segment lookup
    rng = __import__("numpy").random.default_rng(0)
    by_ep: dict[int, list] = {}
    kept = 0
    for r in labels:
        if args.mix > 0 and rng.random() < args.mix:
            kept += 1
            continue                      # segment keeps the original task string
        by_ep.setdefault(r["episode_index"], []).append((r["seg_start"], r["seg_end"], label_id[r["label"]]))
    if args.mix > 0:
        print(f"mixed conditioning: {kept}/{len(labels)} segments keep original strings")
    for v in by_ep.values():
        v.sort()

    # mirror tree: hardlink everything, then rewrite data parquets + tasks table
    if os.path.exists(out):
        raise SystemExit(f"{out} exists — remove first")
    for dirpath, _, files in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        os.makedirs(os.path.join(out, rel), exist_ok=True)
        for f in files:
            # hub snapshots store files as symlinks into blobs/ — link the
            # resolved target so the mirror is self-contained regular files
            s = os.path.realpath(os.path.join(dirpath, f))
            d = os.path.join(out, rel, f)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    # episode frame ranges
    epf = sorted(glob.glob(os.path.join(src, "meta", "episodes", "**", "*.parquet"), recursive=True))
    eps_meta = pd.concat([pd.read_parquet(f) for f in epf])
    ep_lo = dict(zip(eps_meta["episode_index"], eps_meta["dataset_from_index"]))

    n_rewritten = 0
    for pqf in sorted(glob.glob(os.path.join(src, "data", "**", "*.parquet"), recursive=True)):
        rel = os.path.relpath(pqf, src)
        df = pd.read_parquet(pqf)
        eps_here = set(df["episode_index"].unique()) & set(by_ep)
        if not eps_here:
            continue                        # hardlinked copy already correct
        ti = df["task_index"].to_numpy().copy()
        gidx = df["index"].to_numpy()
        epi = df["episode_index"].to_numpy()
        for ep in eps_here:
            lo = ep_lo[ep]
            m = epi == ep
            local = gidx[m] - lo
            new = ti[m]
            for (s, e, lid) in by_ep[ep]:
                new[(local >= s) & (local < e)] = lid
            ti[m] = new
            n_rewritten += int(m.sum())
        df["task_index"] = ti
        dst = os.path.join(out, rel)
        os.unlink(dst)                      # break hardlink before rewrite
        df.to_parquet(dst, index=False)

    # extended tasks table (schema-matched to source)
    all_tasks = orig_tasks + uniq
    if tasks_pq.index.dtype == object:      # tasks as index
        new_tasks = pd.DataFrame({tasks_pq.columns[0] if len(tasks_pq.columns) else "task_index":
                                  range(len(all_tasks))}, index=all_tasks)
        new_tasks.index.name = tasks_pq.index.name
    else:
        new_tasks = pd.DataFrame({tasks_pq.columns[0]: all_tasks})
    dst = os.path.join(out, "meta", "tasks.parquet")
    os.unlink(dst)
    new_tasks.to_parquet(dst)

    # info.json total_tasks update
    info_p = os.path.join(out, "meta", "info.json")
    info = json.load(open(info_p))
    if "total_tasks" in info:
        os.unlink(info_p)
        info["total_tasks"] = len(all_tasks)
        json.dump(info, open(info_p, "w"), indent=2)

    print(f"rewrote task_index on {n_rewritten} frames; dataset at {out}")


if __name__ == "__main__":
    main()
