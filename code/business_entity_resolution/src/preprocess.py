"""Normalize every source file once and cache the result as parquet.

Usage: python preprocess.py --data-dir <student_resource> --work-dir <work>
"""
import argparse
import os
from multiprocessing import Pool

import pandas as pd

from config import DEFAULT_DATA_DIR, DEFAULT_WORK_DIR
from normalize import record_fields

CHUNK = 200_000
BLOCK = 5_000
WORKERS = int(os.environ.get("ER_WORKERS", 3))


def _process(rows):
    """Worker: rows is a list of (name, address) tuples."""
    return [record_fields(n, a) for n, a in rows]


def read_tsv(path, **kw):
    """Read a challenge TSV with every field as a plain string."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3, **kw)


def preprocess_file(src, dst, pool):
    """Normalize one source TSV into a parquet file, streaming chunk by chunk."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer, n, tmp = None, 0, dst + ".tmp"
    try:
        for ch in read_tsv(src, chunksize=CHUNK):
            rows = list(zip(ch.business_name, ch.business_address))
            blocks = [rows[i:i + BLOCK] for i in range(0, len(rows), BLOCK)]
            recs = [r for block in pool.imap(_process, blocks) for r in block]
            out = pd.DataFrame(recs)
            out.insert(0, "entity_id", ch.entity_id.values)
            out.insert(1, "country", ch.country.str.strip().str.lower().values)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table.cast(writer.schema))
            n += len(out)
            del rows, blocks, recs, out, table
            print(f"  {os.path.basename(src)}: {n:,}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    with Pool(WORKERS, maxtasksperchild=200) as pool:
        for split in args.splits.split(","):
            for i in (1, 2, 3):
                src = os.path.join(args.data_dir, "dataset", split, f"{split}_source{i}.tsv")
                dst = os.path.join(args.work_dir, f"{split}_s{i}.parquet")
                if os.path.exists(dst):
                    print("skip", dst)
                    continue
                preprocess_file(src, dst, pool)


if __name__ == "__main__":
    main()
