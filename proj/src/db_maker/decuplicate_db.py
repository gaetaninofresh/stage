'''
    Should have done this in the disasm -> make_db pipeline but I don't feel like poking into that
    (I hated writing that)
'''

from argparse import ArgumentParser
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to parquet db")
    parser.add_argument("-o", "--out", default="./out",
                        type=str, help="")
    args = parser.parse_args()

    lf = pl.scan_parquet(args.db)

    lf = lf.with_columns(
        pl.col("ids").hash().alias("hash")
    )

    conflict_mask = (
        lf.group_by("hash")
        .agg(pl.col("labels").n_unique().alias("label_count"))
        .filter(pl.col("label_count") == 1)
        .select("hash")
    )

    lf_clean = lf.join(conflict_mask, on="hash", how="inner")

    lf_final = lf_clean.unique(subset=["hash"], keep="first")
    lf_final.sink_parquet(args.out)
