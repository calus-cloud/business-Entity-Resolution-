"""Turn pair probabilities into final match sets."""


def select(scored, threshold, rel=0.0, one_to_one=False):
    """scored: frame with s1_id, cand_id, p. Returns {s1_id: set(cand_id)}.

    Keep a pair when p >= threshold and p >= rel * (best p for that S1 entity).
    With one_to_one, each candidate record is kept only for the S1 entity that
    gives it the highest probability (Source 1 is deduplicated, so a record
    belongs to at most one S1 entity).
    """
    s = scored[scored.p >= threshold]
    if rel > 0:
        s = s[s.p >= rel * s.groupby("s1_id").p.transform("max")]
    if one_to_one:
        s = s.sort_values("p", ascending=False).drop_duplicates("cand_id")
    return s.groupby("s1_id").cand_id.agg(set).to_dict()
