"""Local evaluation: macro-averaged F0.5 per Source 1 entity (challenge metric)."""
import numpy as np


def f05(pred, true):
    """F0.5 for one entity; pred/true are sets. Empty/empty scores 1.0."""
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    """Average F0.5 over every S1 id in true_map (missing predictions = empty)."""
    return float(np.mean([f05(pred_map.get(k, set()), v) for k, v in true_map.items()]))


def blocking_recall(cand_map, true_map):
    """Fraction of true matched pairs that survive blocking (the recall ceiling)."""
    tot = hit = 0
    for k, v in true_map.items():
        c = cand_map.get(k, set())
        tot += len(v)
        hit += len(v & c)
    return hit / max(tot, 1)
