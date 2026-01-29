import os
import uuid
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict


class IncrementalDBWriter:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        self.schema = pa.schema({
            "ids": pa.list_(pa.uint32()),
            "attention_mask": pa.list_(pa.uint32()),
            "hash": pa.uint64(),
            "labels": pa.uint8()
        })

    def write_batch(self, encodings: Dict[str, List[List[int]]], hashes: List[int], labels: List[int]):
        if len(labels) == 0:
            return

        table = pa.table({
            "ids": pa.array(encodings["ids"], type=pa.list_(pa.uint32())),
            "attention_mask": pa.array(encodings["attention_mask"], type=pa.list_(pa.uint32())),
            "hash": pa.array(hashes, type=pa.uint64()),
            "labels": pa.array(labels, type=pa.uint8())
        }, schema=self.schema)

        # Write a unique file for every batch or chunk
        filename = f"part-{uuid.uuid4().hex}.parquet"
        out_path = os.path.join(self.base_dir, filename)

        pq.write_table(table, out_path)

    def close(self):
        pass
