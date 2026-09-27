# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We solve the task as a classic two-stage entity-resolution pipeline: an aggressive,
country-agnostic normalizer feeds an IDF-weighted typed-token blocking step that retrieves
the top-K Source 2 and top-K Source 3 candidates per Source 1 record, and a LightGBM
pairwise classifier over ~34 string-similarity and blocking-context features scores each
candidate pair. Final matches are chosen with a probability threshold (plus relative-to-best
and one-record-one-owner constraints) tuned directly for macro F0.5 on held-out Source 1
entities. Everything is engineered to run on an 8 GB laptop over ~12M training and ~8.5M test records.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale:** train = 2.21M S1 / 5.03M S2 / 5.29M S3 records; test = 1.73M S1 / 4.89M S2 / 5.08M S3.
  Train countries are US and India only; test adds France (e.g. 731k French S3 records).
- **Match multiplicity:** most S1 entities have 2–6 matches (S2 + S3 combined);
  5.6% have none (singletons, scored 1.0 only when predicted empty) and 5.4% have one.
  About 74% of all S2/S3 records belong to some S1 entity, and each belongs to at most one.
- **Noise patterns observed:**
  - Phonetic/transliteration corruption of Indian names (`sunrise infrastructure private limited`
    → `snraaij inphraasttrkcr praaivett limittedd`, `star trading` → `sttaar ttredding`).
  - Character typos, swaps and leetspeak (`holland` → `hloland`, `technologies` → `techn0logies`).
  - Word splitting / merging (`sammy mcgowan metro` → `sammymcgowanmetro`).
  - Legal suffix variation / reordering (`inc prairie ...`, `pvt ltd`, `private limited`, `llc`),
    extra tokens (`id 82095`, `services`), and names given as web domains.
  - Addresses: missing entirely (≈21% of candidates in blocking), truncated to city/state,
    reordered components, house numbers split (`121` → `1 21`), landmark phrases (`opp ...`, `near ...`).
- Business names are highly repetitive across different entities (a name-only retrieval channel
  reached just 34% recall), so address evidence (numbers, street words) is essential.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier  
**Core Innovation:** A single typed-token representation (name words, consonant skeletons,
concatenated name, address words, numbers, number+street keys) shared by blocking and by
features, IDF-weighted per country so the method is open-set over countries (France needs no
special handling); plus an out-of-core implementation (hashed features, integer row-index
pairs, per-country and per-chunk processing) that makes the full data tractable on 8 GB RAM.

---

## 3. Candidate Generation (Blocking)

- **Normalization (`normalize.py`):** unidecode transliteration, lowercase, `&`→`and`,
  punctuation removal, null-token removal, abbreviation canonicalization (street/st, road/rd,
  avenue/ave, ...), legal-form and filler removal to get a *core name*, consonant skeleton per
  name word (robust to vowel noise like `praaivett`), concatenated core name (robust to split /
  merged words), address digit tokens and `number_street` keys.
- **Blocking keys used:** each record becomes a bag of typed tokens
  `n:` name word, `k:` consonant skeleton, `c:` concatenated name, `a:` address word, `d:` number,
  `p:` number+street key. Tokens are hashed (2^23 buckets), IDF-weighted, tokens with document
  frequency > 2000 are dropped, rows are L2-normalized, and each S1 record retrieves its top-K
  S2 and top-K S3 records by cosine similarity within the same country
  (multi-threaded sparse top-n product, `sparse_dot_topn`). Unseen country labels fall back
  to searching the whole pool.
- **Candidate pairs generated:** K × 2 per S1 record — [TODO: fill in total from final run,
  e.g. 1.73M × 2 × K].
- **How you ensured true matches were not lost:** recall measured on held-out train S1
  samples against the full train pools:

  | K per source | Blocking recall (pairs) | Candidates / S1 |
  |---|---|---|
  | 5  | 89.4% | 10 |
  | 10 | 92.4% | 20 |
  | 15 | 93.5% | 30 |
  | 20 | 94.1% | 40 |
  | 30 | 94.6% | 60 |
  | 50 | 95.3% | 100 |

  (India 88.5% / US 93.4% at K=10.) We also tried appending name character trigrams to the
  token bag (recall fell to 91.5% at K=10: dilutes address evidence) and a separate name-trigram
  channel unioned with the main one (+0.3–0.5 points for +30–70% candidates); neither was worth it.
  The remaining misses are mostly records with empty addresses *and* heavily corrupted names,
  which the precision-weighted metric would largely reject anyway.

---

## 4. Matching Model

**Features used (`features.py`, all computed with rapidfuzz in C):**
- Name features: ratio, token-set, token-sort, partial ratio and Jaro-Winkler on the core name;
  ratio / token-set on the consonant skeleton; token-set on the full cleaned name; ratio and
  partial ratio on the concatenated name; 4-char prefix equality; exact core-name equality;
  name lengths; "name is a web domain" flags.
- Address features: ratio, token-set, token-sort, partial ratio on the cleaned address;
  token-set / token-sort on digit tokens; token-set on number+street keys; address lengths;
  empty-address and no-number flags.
- Other (blocking context): cosine score, rank within source, score relative to and gap from
  the best candidate of that source, score relative to the best candidate overall, number of
  candidates, source id (S2/S3). No country feature is used, so the model transfers to France.

**Model type:** LightGBM binary classifier (learning rate 0.08, 127 leaves, early stopping on
held-out entities, then refit on all sampled pairs). Trained on candidate pairs of a random
sample of [TODO: N] training S1 entities blocked against the *full* train S2/S3 pools, so the
negatives have the same difficulty as at test time. 20% of the sampled S1 entities are held out
(grouped by S1 entity).

**Threshold selection method:** grid search over probability threshold (0.2–0.9), a
relative-to-best constraint (p ≥ rel × best p of that S1 ∈ {0, 0.3, 0.5}) and an optional
one-to-one constraint (each S2/S3 record assigned only to the S1 entity that scores it highest),
choosing the combination that maximizes the exact challenge macro F0.5 (including singleton
handling) on the held-out S1 entities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [TODO: validation score from train.py / public LB score]
- **Blocking recall ceiling:** [TODO: final K] → see table in §3.
- **Common false positives (wrong merges):** [TODO after training — expected: different branches
  of the same chain / same generic name in the same city with missing addresses.]
- **Common false negatives (missed matches):** [TODO after training — expected: candidates with
  empty address and heavily transliterated names (`eses inphraa praa li` vs `ss infra pvt ltd`).]

---

## 6. Conclusion

A carefully normalized, typed-token IDF blocking step plus a gradient-boosted pairwise matcher
tuned directly for macro F0.5 gives a strong, fully country-agnostic solution that runs
out-of-core on commodity hardware. The biggest lessons: address numbers and number+street keys
carry most of the discriminative signal because business names repeat so much, and blocking
recall (not the classifier) is the main ceiling.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `src/` (all source), `README.md`, `requirements.txt`, `aws_run.sh`.

| Module | Role |
|---|---|
| `normalize.py` | name/address normalization and typed blocking tokens |
| `preprocess.py` | streams each raw TSV through the normalizer into parquet (multiprocess) |
| `rowio.py` | row/column-selective parquet access (keeps memory low) |
| `blocking.py` | per-country hashed IDF cosine top-K candidate generation |
| `features.py` | pairwise features |
| `train.py` | labels, LightGBM training, macro-F0.5 threshold tuning |
| `predict.py` | test blocking, chunked scoring, match selection, output writing |
| `evaluate.py` | challenge metric and blocking recall |
| `run_pipeline.py` | **entry point**: reproduces both output files from the raw data |

Reproduce: `cd src && python run_pipeline.py --data-dir <student_resource> --work-dir <work> --out-dir <output>`
(writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` and runs the official validator).

### B. Additional Results

[TODO: feature importance table printed by train.py; submission history.]
