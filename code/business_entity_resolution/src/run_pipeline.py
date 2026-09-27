"""End-to-end: raw TSVs -> normalized parquet -> trained matcher -> test outputs.

Usage:
    python run_pipeline.py --data-dir <student_resource> [--work-dir W] [--out-dir O]
                           [--n-s1 100000] [--skip-train]

Steps that already have cached outputs in the work dir are skipped
(preprocess parquet files, blocking pairs); delete them to recompute.
"""
import argparse
import os
import subprocess
import sys

from config import DEFAULT_DATA_DIR, DEFAULT_OUT_DIR, DEFAULT_WORK_DIR

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    cmd = [sys.executable, os.path.join(HERE, script), *map(str, args)]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-s1", type=int, default=100_000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cap", type=int, default=2000)
    ap.add_argument("--limit-s1", type=int, default=None,
                    help="sanity run: only block/score the first N test S1 rows")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse work-dir/model.txt + model_meta.json")
    args = ap.parse_args()

    run("preprocess.py", "--data-dir", args.data_dir, "--work-dir", args.work_dir)
    if not args.skip_train:
        run("train.py", "--data-dir", args.data_dir, "--work-dir", args.work_dir,
            "--n-s1", args.n_s1, "--k", args.k, "--cap", args.cap)
    extra = [] if args.limit_s1 is None else ["--limit-s1", args.limit_s1]
    run("predict.py", "--work-dir", args.work_dir, "--out-dir", args.out_dir, *extra)
    validator = os.path.join(args.data_dir, "utils", "validate_submission.py")
    if os.path.exists(validator):
        subprocess.run([sys.executable, validator,
                        "--matching", os.path.join(args.out_dir, "matching_results.tsv"),
                        "--candidate", os.path.join(args.out_dir, "candidate_pairs.tsv"),
                        "--test-dir", os.path.join(args.data_dir, "dataset", "test")], check=False)


if __name__ == "__main__":
    main()
