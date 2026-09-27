# Handoff — Business Entity Resolution (Amazon ML Challenge 2026)

**Deadline: 2026-09-27 23:59 IST.** Max 5 leaderboard submissions per day.
**Status: the code is complete and tested up to training. Training plus full test inference still need to run on AWS.**

---

## 1. The problem

There are 3 sources of business records (`entity_id, business_name, business_address, country`).

- **Source 1 (S1)** is the clean reference: one row per real business.
- **Source 2 / Source 3** are noisy copies of those businesses.
- For every S1 record, output **all** S2/S3 records that are the same business. That can be zero, one or many.

Scoring is **macro F0.5 per S1 entity**, so **precision counts 2× recall**.
- A wrong merge hurts more than a missed one.
- An S1 with no true matches scores 1.0 only if we predict nothing.

The rules: no external data or APIs, any model must be MIT/Apache and ≤ 8B params, and country is an open set. Train has US and India only; test also has **France**.

## 2. The data (what we learned)

| | S1 | S2 | S3 |
|---|---|---|---|
| train | 2.21M | 5.03M | 5.29M |
| test | 1.73M | 4.89M | 5.08M |

- Most S1 entities have **2–6 matches**. 5.6% have none and 5.4% have exactly one.
- About 74% of S2/S3 records belong to some S1, and each belongs to **at most one** S1.
- Noise seen in the data:
  - transliteration: `sunrise infrastructure private limited` → `snraaij inphraasttrkcr praaivett limittedd`
  - typos and leetspeak: `hloland`, `techn0logies`
  - merged words: `sammymcgowanmetro`
  - legal-suffix variants: `pvt ltd`, `private limited`, `llc`
  - missing or truncated addresses (about 21%)
  - house numbers split (`121` → `1 21`)
- **Business names repeat a lot across different businesses.** Name alone is weak evidence; address numbers and streets are what separate them.

## 3. The approach (intuition)

Classic entity-resolution pipeline: **cheap search for candidates → careful pairwise model → precision-tuned decision.**

1. **Normalize** (`normalize.py`)
   - Transliterate to ASCII and lowercase.
   - Canonicalize abbreviations (`street`→`st`, ...).
   - Strip legal suffixes to get a *core name*.
   - Build a **consonant skeleton** per name word. It is robust to vowel noise: `praaivett` → `prvt`.
   - Build a **concatenated name**, which is robust to split or merged words.
   - Extract address numbers and `number_street` keys.

   Every record becomes a bag of typed tokens: `n:` name word, `k:` skeleton, `c:` concat name, `a:` address word, `d:` number, `p:` number+street.

2. **Blocking** (`blocking.py`). Comparing 1.7M × 10M pairs is impossible, so for each S1 we pull only the **top-K most similar S2 and top-K S3** records.
   - Tokens are hashed and IDF-weighted, so rare tokens count more. Tokens seen more than 2000 times are dropped.
   - Similarity is cosine, computed within the same country.
   - The search is a multi-threaded sparse top-n product (`sparse_dot_topn`).
   - A country missing from the pool falls back to searching everything, which keeps country open-set.

3. **Features** (`features.py`): 34 per candidate pair.
   - Fuzzy name similarities (ratio, token-set, Jaro-Winkler, skeleton, concat).
   - Address and number similarities.
   - Blocking context: cosine score, rank, gap to the best candidate.
   - **No country feature**, so the model transfers to France.

4. **Model** (`train.py`): LightGBM binary classifier.
   - Trained on candidate pairs of a random sample of train S1 entities, blocked against the **full** train pools. Negatives are therefore as hard as at test time.
   - 20% of the S1 entities are held out.

5. **Decision.** Keep a candidate if its probability ≥ threshold. Two optional constraints:
   - probability ≥ rel × the best probability for that S1
   - one-to-one: each S2/S3 record goes only to the S1 that scores it highest

   The threshold and both options are grid-searched to maximize the exact challenge metric on held-out entities.

6. **Outputs** (`predict.py`): `output/matching_results.tsv` (the final matches) and `output/candidate_pairs.tsv` (all blocking candidates; the matches are a subset of them).

**Why it's built this way:** the first version crashed with `MemoryError` on an 8 GB laptop. Everything now streams:
- pairs are stored as integer row indices, not string IDs
- data is processed one country at a time, and test S1 is scored in chunks of 200k
- only the parquet columns and rows actually needed are read (`rowio.py`)

## 4. What's done

- [x] Preprocessing of all 6 files, streaming, with 3 workers and 200k-row chunks.
- [x] Blocking rewritten to be memory-lean. Measured on held-out train S1 against the full pools:

  | K per source | recall of true pairs | candidates / S1 |
  |---|---|---|
  | 5 | 89.4% | 10 |
  | 10 | 92.4% | 20 |
  | 20 | **94.1%** | 40 |
  | 30 | 94.6% | 60 |
  | 50 | 95.3% | 100 |

  Two ideas tried and dropped:
  - adding name char-trigrams to the tokens: worse, 91.5% at K=10
  - a separate name-only search channel: only +0.3–0.5 points
- [x] Features verified: 34 columns, no NaNs.
- [x] Full `predict.py` path run on 20k test S1 with a dummy scorer. The **official validator PASSES**.
- [x] `run_pipeline.py` (one entry point), `aws_run.sh`, `README.md`, `requirements.txt`.
- [x] `Documentation_template.md` drafted. The TODOs are waiting for training numbers.
- [ ] **`train.py`'s LightGBM part has never run.** The sanity run below exists to catch bugs in it.

## 5. What to do next (in order)

### Step 1: set up AWS (≥ 32 GB RAM, ≥ 8 vCPU, Python ≥ 3.11)
```bash
git clone https://github.com/calus-cloud/business-Entity-Resolution-
cd business-Entity-Resolution-
# copy the student_resource folder (dataset/ + utils/, ~2.3 GB) onto the box
```

### Step 2: sanity run (a few minutes; catches bugs in training)
```bash
bash code/business_entity_resolution/aws_run.sh /path/to/student_resource 2000 10 20000
```
This uses a 2k train sample, K=10, and scores only the first 20k test S1. It should end with the validator saying `PASS`.

### Step 3: full run
```bash
bash code/business_entity_resolution/aws_run.sh /path/to/student_resource 300000 20
```
- The log goes to `code/business_entity_resolution/pipeline.log`.
- The outputs go to `output/`.
- Look for `blocking recall`, `val macro F0.5 ... at threshold/rel/one_to_one`, and the feature importances.
- Full test inference time is **unknown**, so start early.
- Preprocessing and blocking pairs are cached in `work/`. A rerun skips them.

### Step 4: submit #1
Upload `output/matching_results.tsv` to the leaderboard. **Keep a copy of every submitted file with its score** (the rules require version history).

### Step 5: improve (only if time allows; trust local val F0.5 over public LB)
- **Change only the threshold** (no re-scoring):
  ```bash
  cd code/business_entity_resolution/src
  python predict.py --work-dir ../../../work --reuse-scores --threshold 0.55
  ```
  Higher threshold = more precision, which usually helps F0.5.
- Try a larger training sample (`--n-s1`), or a larger K for more recall (but more candidates and more time).

### Step 6: final package
- Fill the TODOs in `Documentation_template.md`: val F0.5, number of candidate pairs, error examples.
- Zip `output/` (both TSVs), `code/business_entity_resolution/` and `Documentation_template.md`.

## 6. Open issues and risks

- **The repo is public.** Make it private until the challenge ends: GitHub → Settings → Danger Zone → Change visibility. A private clone needs a personal access token.
- `Unidecode` is GPL-2.0. The licence rule covers *models*, so this is probably fine. If in doubt, swap it for stdlib `unicodedata` and rerun preprocessing.
- Blocking recall (~94%) is the ceiling on recall. The remaining misses are mostly empty addresses combined with heavily corrupted names.
- France is unseen in training. We rely on country-agnostic features. Check the France share of the matches after the full run.

## 7. File map

```
code/business_entity_resolution/
  aws_run.sh            one-command run on Linux
  README.md             setup + reproduce instructions
  requirements.txt
  src/
    run_pipeline.py     ENTRY POINT: preprocess -> train -> predict -> validate
    normalize.py        name/address cleaning, blocking tokens
    preprocess.py       raw TSV -> normalized parquet (work/)
    rowio.py            read selected rows/columns of parquet
    blocking.py         top-K candidate search per country
    features.py         34 pair features
    train.py            LightGBM + threshold tuning -> work/model.txt, model_meta.json
    predict.py          test blocking + scoring -> output/*.tsv
    evaluate.py         macro F0.5 metric, blocking recall
    select_matches.py   threshold / rel / one-to-one selection
Documentation_template.md   methodology write-up (needs final numbers)
PLAN.md                     original plan
```
