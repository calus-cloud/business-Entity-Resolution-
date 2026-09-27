# Business Entity Resolution — Plan

Amazon ML Challenge 2026. Deadline: **2026-09-27 23:59 IST**. Max 5 leaderboard submissions/day.

## Data location

Student resource (read-only source of truth):

```
E:\6ab10eb3b23ba_student_resource\student_resource\
├── dataset/
│   ├── train/  train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/   test_source1.tsv, test_source2.tsv, test_source3.tsv
├── utils/validate_submission.py
├── Documentation_template.md
└── README.md
```

(Ignore the `__MACOSX/` folder next to it.) Code reads this path from a `--data-dir` argument / `DATA_DIR` env var, defaulting to the path above.

## Goal

For each Source 1 entity, output all matching Source 2 / Source 3 IDs (zero, one or many).
Metric: macro-averaged F0.5 per Source 1 entity. Singletons score 1.0 only if predicted empty. **Precision first.**

## Rules to respect

- No external data, APIs, geocoding or registries.
- Any model used: MIT/Apache-2.0, ≤ 8B params.
- Country is an open set (test adds France). No hard-coding to US/India.
- Every Source 1 test entity gets exactly one row, even with no matches.

## Pipeline

1. **Load** all TSVs with `sep="\t"`, `dtype=str`, `keep_default_na=False`.
2. **Normalize** (`src/normalize.py`)
   - Lowercase, strip accents (unidecode), `&` → `and`, remove punctuation.
   - Expand abbreviations: corp/corporation, pvt/private, ltd/limited, inc, co, rd/road, st/street, ave, blvd, nagar, marg, etc.
   - Name core = name with legal suffixes removed (inc, llc, ltd, pvt, private, limited, corp, sarl, sas, sa, gmbh...).
   - Address: extract digit tokens (house number, PIN/ZIP/postcode), drop landmark phrases ("near ...", "opp ...").
3. **Blocking** (`src/blocking.py`)
   - TF-IDF on char 3–4-grams over name core, and separately over name + address.
   - For each S1 record, get the top-K nearest S2 and S3 records (cosine, sparse NN), within the same country when the country is present.
   - Union with exact matches on postcode + first name token.
   - Tune K on train to reach recall ≥ 98% while keeping the candidate count small.
   - Output → `candidate_pairs.tsv`.
4. **Pair features** (`src/features.py`)
   - Name: Jaro-Winkler, Levenshtein ratio, token-set/sort ratio, TF-IDF cosine, Jaccard of tokens, core-name exact-match flag, acronym match.
   - Address: token-set ratio, TF-IDF cosine, number-token overlap, postcode equal/conflict, city-token overlap.
   - Context: candidate's rank for this S1, score gap to the best candidate, source (S2/S3), address missing/short flags.
   - No country one-hot. Features stay country-agnostic so they transfer to France.
5. **Model** (`src/train.py`)
   - LightGBM binary classifier on candidate pairs, labels from ground truth.
   - GroupKFold by S1 entity (5 folds), so the validation split mimics the test set.
6. **Decision** (`src/predict.py`)
   - Keep candidates with prob ≥ threshold `t`, tuned to maximize the local macro-F0.5 metric.
   - Optionally also require prob ≥ α × best prob for that S1 entity. Tune α too.
7. **Write outputs** → `output/matching_results.tsv`, `output/candidate_pairs.tsv`. Matches must be a subset of candidates.
8. **Validate** with `python utils/validate_submission.py --matching ... --candidate ... --test-dir .../dataset/test`.

## Local evaluation

`src/evaluate.py`: exact macro-F0.5 from the problem statement, including singleton handling. It also reports blocking recall and the reduction ratio.

## Repo layout (matches final zip)

```
code/business_entity_resolution/
├── src/  normalize.py, blocking.py, features.py, train.py, predict.py, evaluate.py, run_pipeline.py
├── README.md
└── requirements.txt
output/  matching_results.tsv, candidate_pairs.tsv
Documentation_template.md   (filled in)
```

## Timeline (today)

1. EDA: sizes, match-count distribution, singleton rate, example noise per country.
2. Baseline: blocking + simple thresholded similarity, then **submit #1**.
3. LightGBM matcher + threshold tuning, then submit #2.
4. Feature iteration (abbreviation lists, number/postcode logic), then submits #3–#5.
5. Pack the zip and fill in the documentation.

## Risks

- France is unseen: rely on generic char-n-gram and accent-stripping features, not language-specific rules.
- Overfitting the public LB: trust the local CV F0.5 over public LB swings.
- Blocking recall caps the score: measure it first.
