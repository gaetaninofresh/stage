import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any


class IncrementalDBWriter:
    """
    Append small batches (row-groups) to a Parquet file.
    """

    def __init__(self, path: str):
        self.path = path
        self.schema = pa.schema({
            "ids": pa.list_(pa.uint32()),
            "attention_mask": pa.list_(pa.uint32()),
            "labels": pa.uint8()
        })
        self.writer = pq.ParquetWriter(path, self.schema, use_dictionary=False)

    def write_batch(self, encodings: Dict[str, List[List[int]]], labels: List[int]):
        if len(labels) == 0:
            return

        table = pa.table({
            "ids": pa.array(encodings["ids"], type=pa.list_(pa.uint32())),
            "attention_mask": pa.array(encodings["attention_mask"], type=pa.list_(pa.uint32())),
            "labels": pa.array(labels, type=pa.uint8())
        }, schema=self.schema)

        self.writer.write_table(table)

    def close(self):
        self.writer.close()
