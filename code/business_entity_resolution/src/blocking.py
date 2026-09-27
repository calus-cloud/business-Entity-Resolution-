"""Candidate generation (blocking).

Each record is a bag of typed tokens (name words, consonant skeletons, concatenated
name, address words, numbers, house-number+street keys; see normalize.py). Within
each country (an open set of labels), tokens are hashed, IDF-weighted, very frequent
tokens (df > cap) are dropped, and every Source 1 record retrieves the top-K Source 2
and top-K Source 3 records by cosine similarity (multi-threaded sparse top-n product).

Memory-lean: only entity_id/country/btoks are read, one country at a time.
"""
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sparse_dot_topn import sp_matmul_topn

from rowio import country_rows, fetch_rows, n_rows

N_FEATURES = 2 ** 23
THREADS = os.cpu_count()


def _hash(docs, chunk=500_000):
    vec = HashingVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None,
                            n_features=N_FEATURES, binary=True, norm=None,
                            alternate_sign=False, dtype=np.float32)
    return sp.vstack([vec.transform(docs[i:i + chunk]) for i in range(0, len(docs), chunk)],
                     format="csr")


def _weighted(X, idf):
    """Scale binary token matrix by idf and L2-normalize rows (float32 CSR)."""
    X = X @ sp.diags(idf)
    X = sp.csr_matrix(X, dtype=np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.csr_matrix(sp.diags((1.0 / norms).astype(np.float32)) @ X, dtype=np.float32)


def block_one(q_tok, p_tok, k, cap, chunk=100_000):
    """Top-k pool indices for each query; returns arrays (qi, pi, score, rank)."""
    P = _hash(p_tok)
    df = np.bincount(P.indices, minlength=N_FEATURES)
    n = P.shape[0]
    idf = (np.log((n + 1) / (df + 1)) + 1.0).astype(np.float32)
    idf[(df > cap) | (df == 0)] = 0.0
    P = _weighted(P, idf)
    PT = sp.csr_matrix(P.T)
    del P
    out = []
    for s in range(0, len(q_tok), chunk):
        Q = _weighted(_hash(q_tok[s:s + chunk]), idf)
        S = sp_matmul_topn(Q, PT, top_n=k, sort=True, n_threads=THREADS)
        S = sp.csr_matrix(S)
        cnt = np.diff(S.indptr)
        qi = np.repeat(np.arange(S.shape[0], dtype=np.int32), cnt) + s
        rk = (np.arange(S.nnz) - np.repeat(S.indptr[:-1], cnt)).astype(np.int16)
        out.append((qi, S.indices.astype(np.int32), S.data.astype(np.float32), rk))
    return tuple(np.concatenate([o[i] for o in out]) for i in range(4))


def generate_candidates(q_path, q_rows, pool_paths, k=10, cap=2000, log=print):
    """Block Source 1 rows `q_rows` (global row indices into q_path, or None = all)
    against each pool parquet in pool_paths ({src: path}).

    Returns frame of integer row indices: qrow, src, crow, bscore (cosine), brank.
    """
    q_by_c = country_rows(q_path)
    if q_rows is not None:
        keep = np.zeros(n_rows(q_path), dtype=bool)
        keep[q_rows] = True
        q_by_c = {c: r[keep[r]] for c, r in q_by_c.items()}
    frames = []
    for src, path in pool_paths.items():
        p_by_c = country_rows(path)
        for country in sorted(q_by_c):
            qr = q_by_c[country]
            if len(qr) == 0:
                continue
            # an unseen country label in this pool: search everything
            pr = p_by_c.get(country, np.arange(n_rows(path), dtype=np.int32))
            q_tok = fetch_rows(q_path, qr, ["btoks"])["btoks"]
            p_tok = fetch_rows(path, pr, ["btoks"])["btoks"]
            qi, pi, sc, rk = block_one(q_tok, p_tok, k, cap)
            del q_tok, p_tok
            frames.append(pd.DataFrame({"qrow": qr[qi], "src": np.int8(src), "crow": pr[pi],
                                        "bscore": sc, "brank": rk}))
            log(f"  block {country} S{src}: {len(qr):,} x {len(pr):,} -> {len(qi):,} pairs",
                flush=True)
    return pd.concat(frames, ignore_index=True)
