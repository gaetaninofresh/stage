import os
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Dict, Any
from worker import init_worker, process_bin_chunk
from parqet_writer import IncrementalDBWriter


CHECKPOINT_FN = "processed_checkpoint.txt"


def load_checkpoint(cp_path: str) -> set:
    if not os.path.exists(cp_path):
        return set()
    with open(cp_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def append_checkpoint(cp_path: str, processed_bins: List[str]):
    with open(cp_path, "a") as f:
        for b in processed_bins:
            f.write(b + "\n")


def chunkify(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def process_chunk_wrapper(args):
    bin_chunk, filters = args
    return process_bin_chunk(bin_chunk, filters)


def make_db(bins: List[str],
            out_path: str,
            filters: Dict[str, Any] | None = None,
            workers: int | None = None,
            chunk_size: int = 16,
            write_batch_min: int = 128,
            tokenizer_args: Dict[str, Any] | None = None,
            checkpoint_path: str | None = None):

    workers = workers or max(1, cpu_count() - 4)  # just to be safe
    checkpoint_path = checkpoint_path or CHECKPOINT_FN
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    writer = IncrementalDBWriter(out_path)

    already = load_checkpoint(checkpoint_path)

    # filter bins
    todo_bins = [b for b in bins if b not in already]
    if not todo_bins:
        print("Nothing to do; all binaries were processed according to checkpoint.")
        writer.close()
        return out_path

    buffer_enc = {"ids": [], "attention_mask": []}
    buffer_labels = []

    bin_chunks = list(chunkify(todo_bins, chunk_size))

    print(f"Workers={workers}, chunks={
          len(bin_chunks)}, chunk_size={chunk_size}")

    # Use a Pool with initializer to load tokenizer in each worker once
    with Pool(processes=workers, initializer=init_worker, initargs=(tokenizer_args or {},), maxtasksperchild=10) as pool:

        tasks = [(chunk, filters) for chunk in bin_chunks]

        for encodings, labels, processed_bins in tqdm(
                pool.imap_unordered(process_chunk_wrapper, tasks),
                total=len(tasks)):

            # extend buffer
            buffer_enc["ids"].extend(encodings["ids"])
            buffer_enc["attention_mask"].extend(encodings["attention_mask"])
            buffer_labels.extend(labels)

            # write if we have at least write_batch_min entries
            if len(buffer_labels) >= write_batch_min:
                writer.write_batch(buffer_enc, buffer_labels)
                buffer_enc = {"ids": [], "attention_mask": []}
                buffer_labels = []

            # checkpoint processed bins for resume
            if processed_bins:
                append_checkpoint(checkpoint_path, processed_bins)

    # flush remaining buffer
    if buffer_labels:
        writer.write_batch(buffer_enc, buffer_labels)

    writer.close()


def is_executable_binary(path):
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header == b"\x7fELF":
            return True
        else:
            return False
    except Exception:
        return False


def discover_bins(bins_dir):
    bins = []
    for root, _, files in os.walk(bins_dir):
        for f in files:
            p = os.path.join(root, f)
            if is_executable_binary(p):
                bins.append(p)
    return bins


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bins_dir", type=str)
    p.add_argument("-o", "--out", default="./out/db.parquet")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--write_batch_min", type=int, default=256)
    p.add_argument("--checkpoint", type=str,
                   default="processed_checkpoint.txt")
    args = p.parse_args()

    bins = discover_bins(args.bins_dir)
    print(f"Discovered {len(bins)} binaries")

    make_db(
        bins,
        out_path=args.out,
        workers=args.workers,
        chunk_size=args.chunk_size,
        write_batch_min=args.write_batch_min,
        checkpoint_path=args.checkpoint
    )
    print("Done.")
