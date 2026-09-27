"""Row-level access to the normalized parquet files without loading them whole."""
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


def n_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def country_rows(path):
    """{country: sorted global row indices} for one parquet file."""
    col = pq.read_table(path, columns=["country"])["country"]
    out = {}
    for c in pc.unique(col).to_pylist():
        out[c] = np.flatnonzero(pc.equal(col, c).to_numpy(zero_copy_only=False)).astype(np.int32)
    return out


def fetch_rows(path, idx, cols, batch=500_000):
    """Values of `cols` for global row indices `idx` (any order, duplicates ok).

    Returns {col: numpy array aligned with idx}.
    """
    idx = np.asarray(idx, dtype=np.int64)
    uniq, inv = np.unique(idx, return_inverse=True)
    parts = {c: [] for c in cols}
    off = 0
    for rb in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=list(cols)):
        a, b = np.searchsorted(uniq, [off, off + rb.num_rows])
        if b > a:
            take = rb.take(uniq[a:b] - off)
            for c in cols:
                parts[c].append(take.column(c).to_numpy(zero_copy_only=False))
        off += rb.num_rows
    return {c: np.concatenate(parts[c])[inv] for c in cols}
