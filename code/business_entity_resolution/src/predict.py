"""Inference on the test set: blocking -> features -> LightGBM -> match selection.

Writes output/candidate_pairs.tsv (exact candidate set scored by the model) and
output/matching_results.tsv (final matches), one row per test Source 1 entity.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import generate_candidates
from config import DEFAULT_OUT_DIR, DEFAULT_WORK_DIR
from features import build_features, pair_fields
from rowio import fetch_rows, n_rows

CHUNK_S1 = 200_000


def write_grouped(path, ids, qrow, cand, col):
    """One row per S1 id (ids[i] is S1 row i); cand[j] belongs to S1 row qrow[j]."""
    order = np.argsort(qrow, kind="stable")
    qs, cs = qrow[order], cand[order]
    bounds = np.searchsorted(qs, np.arange(len(ids) + 1))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for i, s1 in enumerate(ids):
            f.write(s1 + "\t" + ",".join(sorted(set(cs[bounds[i]:bounds[i + 1]]))) + "\n")


def select_mask(pairs, threshold, rel, one_to_one):
    """Boolean mask of pairs kept as matches (see select_matches.select)."""
    p = pairs.p.values
    keep = p >= threshold
    if rel > 0:
        best = pairs.groupby("qrow").p.transform("max").values
        keep &= p >= rel * best
    if one_to_one:
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(-p[idx], kind="stable")]
        key = pairs.src.values[idx].astype(np.int64) * (1 << 32) + pairs.crow.values[idx]
        _, first = np.unique(key, return_index=True)
        keep = np.zeros(len(p), dtype=bool)
        keep[idx[first]] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--no-one-to-one", action="store_true")
    ap.add_argument("--model-dir", default=None, help="dir with model.txt/model_meta.json (default: work dir)")
    ap.add_argument("--k", type=int, default=None, help="override blocking K from model_meta")
    ap.add_argument("--reuse-scores", action="store_true",
                    help="reuse cached test_scored_*.parquet (e.g. to try another --threshold)")
    ap.add_argument("--limit-s1", type=int, default=None, help="smoke test: block only first N S1 rows")
    args = ap.parse_args()
    work, t0 = args.work_dir, time.time()
    mdir = args.model_dir or work
    meta = json.load(open(os.path.join(mdir, "model_meta.json")))
    model = lgb.Booster(model_file=os.path.join(mdir, "model.txt"))

    q_path = os.path.join(work, "test_s1.parquet")
    pools = {2: os.path.join(work, "test_s2.parquet"), 3: os.path.join(work, "test_s3.parquet")}

    k = args.k or meta["k"]
    q_rows = None if args.limit_s1 is None else np.arange(args.limit_s1)
    tag = f"{k}_{meta['cap']}" + ("" if q_rows is None else f"_lim{args.limit_s1}")
    pairs_path = os.path.join(work, f"test_pairs_{tag}.parquet")
    if os.path.exists(pairs_path):
        pairs = pd.read_parquet(pairs_path)
    else:
        pairs = generate_candidates(q_path, q_rows, pools, k=k, cap=meta["cap"])
        pairs.to_parquet(pairs_path, index=False)
    print(f"test pairs {len(pairs):,} ({time.time() - t0:.0f}s)", flush=True)

    scored_path = os.path.join(work, f"test_scored_{tag}.parquet")
    if args.reuse_scores and os.path.exists(scored_path):
        pairs = pd.read_parquet(scored_path)
    else:
        n_q = n_rows(q_path)
        p = np.zeros(len(pairs), dtype=np.float32)
        for s in range(0, n_q, CHUNK_S1):
            m = ((pairs.qrow.values >= s) & (pairs.qrow.values < s + CHUNK_S1))
            if not m.any():
                continue
            part = pairs[m].reset_index(drop=True)
            Q, C = pair_fields(part, q_path, pools)
            X = build_features(part, Q, C)[meta["features"]]
            del Q, C
            p[m] = model.predict(X).astype(np.float32)
            del X, part
            print(f"  scored {min(s + CHUNK_S1, n_q):,}/{n_q:,} ({time.time() - t0:.0f}s)", flush=True)
        pairs["p"] = p
        pairs.to_parquet(scored_path, index=False)

    # integer rows -> entity ids
    ids = fetch_rows(q_path, np.arange(n_rows(q_path)), ["entity_id"])["entity_id"]
    cand = np.empty(len(pairs), dtype=object)
    for src, path in pools.items():
        m = pairs.src.values == src
        cand[m] = fetch_rows(path, pairs.crow.values[m], ["entity_id"])["entity_id"]

    thr = args.threshold if args.threshold is not None else meta["threshold"]
    keep = select_mask(pairs, thr, meta["rel"],
                       one_to_one=meta.get("one_to_one", True) and not args.no_one_to_one)
    qrow = pairs.qrow.values
    os.makedirs(args.out_dir, exist_ok=True)
    write_grouped(os.path.join(args.out_dir, "candidate_pairs.tsv"), ids, qrow, cand, "candidate_entity_ids")
    write_grouped(os.path.join(args.out_dir, "matching_results.tsv"), ids, qrow[keep], cand[keep], "matched_entity_ids")
    n_empty = len(ids) - len(np.unique(qrow[keep]))
    print(f"wrote outputs: {len(ids):,} S1, {int(keep.sum()):,} matches, {n_empty:,} empty "
          f"(threshold {thr}, rel {meta['rel']}) ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
