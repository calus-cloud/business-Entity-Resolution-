# Business Entity Resolution — Amazon ML Challenge 2026

For each Source 1 business, find all matching Source 2 / Source 3 records.
Pipeline: normalization → IDF-weighted token blocking (top-K per source) →
pairwise string-similarity features → LightGBM matcher → thresholded,
one-to-one match selection.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ (pandas 3 needs it; developed on 3.13). No external data, APIs or pretrained models are used.

## Reproduce both outputs end-to-end

```bash
cd src
python run_pipeline.py --data-dir /path/to/student_resource \
                       --work-dir /path/to/work --out-dir /path/to/output
```

`--data-dir` must contain `dataset/train/*.tsv` and `dataset/test/*.tsv`.
Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs
`utils/validate_submission.py` when it is present in the data dir.

Individual steps (cached artifacts in the work dir are reused):

| Step | Command | Output (work dir) |
|------|---------|-------------------|
| 1. Normalize all 6 source files | `python preprocess.py --data-dir D --work-dir W` | `{train,test}_s{1,2,3}.parquet` |
| 2. Train matcher (sample of train S1) | `python train.py --data-dir D --work-dir W --n-s1 100000` | `train_pairs_*.parquet`, `model.txt`, `model_meta.json` |
| 3. Block + score test, write outputs | `python predict.py --work-dir W --out-dir O` | `test_pairs_*.parquet`, `test_scored.parquet` |

`predict.py --reuse-scores --threshold T` re-selects matches from cached scores without re-scoring.

## Files

| File | Purpose |
|------|---------|
| `src/normalize.py` | Name/address cleaning, abbreviation canonicalization, legal-suffix removal, blocking tokens |
| `src/preprocess.py` | Streams each TSV through `normalize.record_fields` (multiprocess) into parquet |
| `src/rowio.py` | Reads selected rows/columns of the parquet files without loading them whole |
| `src/blocking.py` | Per-country hashed-token IDF cosine, top-K Source 2 and Source 3 candidates per Source 1 record |
| `src/features.py` | Pairwise features (rapidfuzz similarities on names/addresses, blocking score/rank context) |
| `src/train.py` | Labels pairs from ground truth, fits LightGBM, tunes the threshold for macro F0.5 on held-out S1 entities |
| `src/predict.py` | Test blocking, chunked scoring, match selection, output writing |
| `src/evaluate.py` | Local macro-F0.5 metric (challenge definition) and blocking recall |
| `src/run_pipeline.py` | Runs all steps |

## Resources

Designed for a machine with ~8 GB RAM: every step streams or works one country
at a time and stores candidate pairs as integer row indices. More RAM/CPU just
makes it faster (`ER_WORKERS` sets preprocessing worker count, default 3).

## Running on a bigger Linux box (AWS)

```bash
bash aws_run.sh /path/to/student_resource 300000 20    # N_S1 sample size, blocking K
```

Creates a venv, installs requirements, runs all steps and logs to `pipeline.log`.
Quick sanity run first (small train sample, first 20k test S1 only):
`bash aws_run.sh /path/to/student_resource 2000 10 20000`.
Recommended: >= 32 GB RAM, >= 8 vCPU.
