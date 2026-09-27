"""Train the pairwise matcher on a sample of training Source 1 entities.

Steps: block the sample against the full training S2/S3 pools, label pairs with
the ground truth, build features, fit LightGBM on 80% of S1 entities (grouped), and
tune the decision threshold on the held-out 20% for macro F0.5.
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from blocking import generate_candidates
from config import DEFAULT_DATA_DIR, DEFAULT_WORK_DIR
from evaluate import blocking_recall, macro_f05
from features import build_features, pair_fields
from preprocess import read_tsv
from rowio import fetch_rows, n_rows
from select_matches import select

LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127,
                  min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--n-s1", type=int, default=100_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    work = args.work_dir
    t0 = time.time()

    q_path = os.path.join(work, "train_s1.parquet")
    pools = {2: os.path.join(work, "train_s2.parquet"), 3: os.path.join(work, "train_s3.parquet")}
    rng = np.random.RandomState(args.seed)
    q_rows = np.sort(rng.choice(n_rows(q_path), size=min(args.n_s1, n_rows(q_path)), replace=False))
    q_ids = fetch_rows(q_path, q_rows, ["entity_id", "country"])

    gt = read_tsv(os.path.join(args.data_dir, "dataset", "train", "train_ground_truth.tsv"))
    gt = gt[gt.source1_entity_id.isin(set(q_ids["entity_id"]))]
    sample_true = {a: set(x for x in b.split(",") if x)
                   for a, b in zip(gt.source1_entity_id, gt.matched_entity_ids)}
    del gt
    row2id = dict(zip(q_rows, q_ids["entity_id"]))

    pairs_path = os.path.join(work, f"train_pairs_{args.n_s1}_{args.k}_{args.cap}.parquet")
    if os.path.exists(pairs_path):
        pairs = pd.read_parquet(pairs_path)
    else:
        pairs = generate_candidates(q_path, q_rows, pools, k=args.k, cap=args.cap)
        pairs.to_parquet(pairs_path, index=False)
    print(f"pairs {len(pairs):,}  ({time.time() - t0:.0f}s)", flush=True)

    Q, C = pair_fields(pairs, q_path, pools)
    pairs["s1_id"] = pd.Series(pairs.qrow.values).map(row2id).values
    cand_ids = np.empty(len(pairs), dtype=object)
    for src, path in pools.items():
        m = pairs.src.values == src
        cand_ids[m] = fetch_rows(path, pairs.crow.values[m], ["entity_id"])["entity_id"]
    pairs["cand_id"] = cand_ids

    cand_map = pairs.groupby("s1_id").cand_id.agg(set).to_dict()
    print(f"blocking recall {blocking_recall(cand_map, sample_true):.4f}", flush=True)
    country_of = dict(zip(q_ids["entity_id"], q_ids["country"]))
    for c in sorted(set(q_ids["country"])):
        sub = {k: v for k, v in sample_true.items() if country_of[k] == c}
        print(f"   {c}: {blocking_recall(cand_map, sub):.4f}")
    del cand_map

    X = build_features(pairs, Q, C)
    del Q, C
    y = np.fromiter((c in sample_true[q] for q, c in zip(pairs.s1_id, pairs.cand_id)), np.int8, len(pairs))
    print(f"features {X.shape}  pos rate {y.mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)

    ids = q_ids["entity_id"]
    val_ids = set(rng.choice(ids, size=len(ids) // 5, replace=False))
    is_val = pairs.s1_id.isin(val_ids).values
    dtr = lgb.Dataset(X[~is_val], y[~is_val])
    dva = lgb.Dataset(X[is_val], y[is_val], reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=1500, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    pv = model.predict(X[is_val], num_iteration=model.best_iteration)
    vp = pairs.loc[is_val, ["s1_id", "cand_id"]].copy()
    vp["p"] = pv
    val_true = {k: sample_true[k] for k in val_ids}
    best = (0, None)
    for t in np.arange(0.2, 0.9, 0.025):
        for rel in (0.0, 0.3, 0.5):
            for o2o in (False, True):
                sc = macro_f05(select(vp, t, rel, one_to_one=o2o), val_true)
                if sc > best[0]:
                    best = (sc, (float(t), rel, o2o))
    print(f"val macro F0.5 {best[0]:.4f} at threshold/rel/one_to_one {best[1]}", flush=True)
    imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    print(imp.head(20).to_string())

    # Refit on all sampled pairs with the tuned number of rounds.
    final = lgb.train(LGB_PARAMS, lgb.Dataset(X, y), num_boost_round=model.best_iteration)
    final.save_model(os.path.join(work, "model.txt"))
    with open(os.path.join(work, "model_meta.json"), "w") as f:
        json.dump({"threshold": best[1][0], "rel": best[1][1], "one_to_one": best[1][2],
                   "val_f05": best[0],
                   "k": args.k, "cap": args.cap, "features": list(X.columns)}, f, indent=2)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
