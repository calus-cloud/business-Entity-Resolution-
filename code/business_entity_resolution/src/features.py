"""Pairwise features for (Source 1, candidate) pairs.

All string similarities are computed in C with rapidfuzz.process.cpdist (multi-
threaded, element-wise). Features are country-agnostic (no country one-hot) so the
model transfers to France, which is unseen in training.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from rowio import fetch_rows

FIELDS = ["name_clean", "name_core", "name_skel", "name_concat", "is_domain",
          "addr_clean", "addr_nums", "addr_keys"]


def _sim(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _prefix_equal(a, b, n=4):
    return np.array([x[:n] == y[:n] and len(x) >= n for x, y in zip(a, b)], dtype=np.int8)


def pair_fields(pairs, q_path, pool_paths):
    """Fetch FIELDS for both sides of each pair: returns (Q, C) dicts of arrays."""
    Q = fetch_rows(q_path, pairs.qrow.values, FIELDS)
    C = {f: np.empty(len(pairs), dtype=object) for f in FIELDS}
    for src, path in pool_paths.items():
        m = (pairs.src.values == src)
        got = fetch_rows(path, pairs.crow.values[m], FIELDS)
        for f in FIELDS:
            C[f][m] = got[f]
    C["is_domain"] = C["is_domain"].astype(np.int64)
    return Q, C


def build_features(pairs, Q, C):
    """pairs: qrow, src, crow, bscore, brank. Q/C: field arrays aligned with pairs.
    Returns a float32 feature frame aligned with pairs."""
    F = {}
    F["src"] = pairs.src.values.astype(np.float32)
    F["bscore"] = pairs.bscore.values
    F["brank"] = pairs.brank.values.astype(np.float32)
    g = pairs.groupby(["qrow", "src"], sort=False).bscore
    F["bscore_rel"] = (pairs.bscore / g.transform("max")).values
    F["bscore_gap"] = (g.transform("max") - pairs.bscore).values
    F["n_cand_src"] = g.transform("size").values.astype(np.float32)
    F["bscore_rel_all"] = (pairs.bscore / pairs.groupby("qrow", sort=False).bscore.transform("max")).values

    qn, cn = Q["name_core"], C["name_core"]
    F["n_ratio"] = _sim(qn, cn, fuzz.ratio)
    F["n_tset"] = _sim(qn, cn, fuzz.token_set_ratio)
    F["n_tsort"] = _sim(qn, cn, fuzz.token_sort_ratio)
    F["n_partial"] = _sim(qn, cn, fuzz.partial_ratio)
    F["n_jw"] = _sim(qn, cn, JaroWinkler.normalized_similarity)
    F["n_skel_ratio"] = _sim(Q["name_skel"], C["name_skel"], fuzz.ratio)
    F["n_skel_tset"] = _sim(Q["name_skel"], C["name_skel"], fuzz.token_set_ratio)
    F["n_clean_tset"] = _sim(Q["name_clean"], C["name_clean"], fuzz.token_set_ratio)
    F["n_concat_ratio"] = _sim(Q["name_concat"], C["name_concat"], fuzz.ratio)
    F["n_concat_partial"] = _sim(Q["name_concat"], C["name_concat"], fuzz.partial_ratio)
    F["n_prefix4"] = _prefix_equal(Q["name_concat"], C["name_concat"]).astype(np.float32)
    F["n_exact"] = (qn == cn).astype(np.float32)
    F["q_name_len"] = np.fromiter((len(x) for x in qn), np.float32, len(qn))
    F["c_name_len"] = np.fromiter((len(x) for x in cn), np.float32, len(cn))
    F["q_is_domain"] = Q["is_domain"].astype(np.float32)
    F["c_is_domain"] = C["is_domain"].astype(np.float32)

    qa, ca = Q["addr_clean"], C["addr_clean"]
    F["a_ratio"] = _sim(qa, ca, fuzz.ratio)
    F["a_tset"] = _sim(qa, ca, fuzz.token_set_ratio)
    F["a_tsort"] = _sim(qa, ca, fuzz.token_sort_ratio)
    F["a_partial"] = _sim(qa, ca, fuzz.partial_ratio)
    F["a_num_tset"] = _sim(Q["addr_nums"], C["addr_nums"], fuzz.token_set_ratio)
    F["a_num_tsort"] = _sim(Q["addr_nums"], C["addr_nums"], fuzz.token_sort_ratio)
    F["a_key_tset"] = _sim(Q["addr_keys"], C["addr_keys"], fuzz.token_set_ratio)
    F["q_addr_len"] = np.fromiter((len(x) for x in qa), np.float32, len(qa))
    F["c_addr_len"] = np.fromiter((len(x) for x in ca), np.float32, len(ca))
    F["c_addr_empty"] = (F["c_addr_len"] == 0).astype(np.float32)
    F["c_nums_empty"] = np.fromiter((len(x) == 0 for x in C["addr_nums"]), np.float32, len(ca))
    return pd.DataFrame(F)
