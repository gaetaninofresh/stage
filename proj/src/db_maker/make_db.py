import re
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import queue
from tqdm import tqdm
from typing import List, Dict, Any, Literal
from worker import init_worker, decompile_bin, consumer_tokenize
from parquet_writer import IncrementalDBWriter
from concurrent.futures import TimeoutError
from pebble import ProcessPool


def make_db(bins_path: list[str],
            out_path: str,
            label_mode: Literal = 'safe',
            filters: Dict = None,
            workers: int = None,
            consumers: int = None,
            write_batch_min: int = 128,
            tokenizer_args: Dict = None,
            timeout_seconds: int = 180):

    # Setup
    workers = workers or max(1, os.cpu_count() - 2)
    consmers = max(1, workers / 4)

    out_path = os.path.dirname(out_path or '.')
    os.makedirs(out_path, exist_ok=True)
    writer_args = {'base_dir': out_path}

    # Manage a queue for feeding data to tokenizers from decompilers
    # and a queue to know what binaries have been processed

    with mp.Manager() as manager:
        # very big to avoid starvation and deadlocks
        tokenize_queue = manager.Queue(maxsize=2048)
        done_queue = manager.Queue(maxsize=256)  # should be big enough
        tokenize_processes = []

        # Create Tokenizer processes
        for _ in range(consumers):
            p = mp.Process(
                target=consumer_tokenize,
                args=(tokenize_queue, done_queue, label_mode,
                      tokenizer_args, writer_args,)
            )
            p.daemon = True  # avoids some nasty zombie processes on crash
            p.start()
            tokenize_processes.append(p)

        with ProcessPool(max_workers=workers) as pool:

            # Set up decompiler's workers
            dec_worker_args = ((bin, tokenize_queue, filters)
                               for bin in bins_path)
            dec_worker_iterator = iter(dec_worker_args)

            # Create a buffer for workers so we load jobs progressively and save RAM
            futures = set()
            for _ in range(workers * 4):
                try:
                    args = next(dec_worker_iterator)
                    f = pool.schedule(
                        decompile_bin,
                        args=args,
                        timeout=timeout_seconds
                    )
                    futures.add(f)
                except StopIteration:
                    break

            # Prograss tracking and exception handling loop
            completed_count = 0
            total = len(bins_path)
            progress_log = open(os.path.join(
                out_path, 'progress_log.txt'), mode='a+')
            processed = []
            with tqdm(total=total) as pbar:
                # Check what bins have been processed
                while completed_count < total:
                    try:
                        while True:
                            done = done_queue.get_nowait()
                            pbar.write(f'{done} processed succesfully')
                            progress_log.write(f'{done}_{label_mode}\n')
                            processed.append(done)
                            pbar.update(1)
                            completed_count += 1
                    except queue.Empty:
                        pass

                    if futures:
                        # Check for exceptions/failures
                        done, futures = concurrent.futures.wait(
                            futures, timeout=.1, return_when=concurrent.futures.FIRST_COMPLETED)

                        for f in done:
                            try:
                                f.result()
                            except Exception as e:
                                pbar.write(f'Decompiler failed: {e}')
                                pbar.update(1)
                                completed_count += 1

                            # Refill binaries queue for pool
                            try:
                                new_args = next(dec_worker_iterator)
                                new_future = pool.schedule(
                                    decompile_bin,
                                    args=new_args,
                                    timeout=timeout_seconds
                                )
                                futures.add(new_future)
                            except StopIteration:
                                pass
                    else:
                        break
                print('Done processing binaries, shutting down')
                progress_log.close()
            # Shutdown tokenizers processes and pool
            for _ in range(len(tokenize_processes)):
                tokenize_queue.put(None)

            for p in (tokenize_processes):
                p.join()

            print('Done')
            return processed


def is_executable_binary(path):
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header == b"\x7fELF"
    except Exception:
        return False


def discover_bins(bins_dir, regex=r'.*'):
    safe = []
    vuln = []
    print(f"Scanning {bins_dir}...")
    for root, _, files in os.walk(bins_dir):
        for f in files:
            bin_path = os.path.abspath(os.path.join(root, f))
            if re.match(regex, bin_path) and is_executable_binary(bin_path):
                if re.match('.*bad.*', bin_path) or re.match('.*vulnerable.*', bin_path):
                    vuln.append(bin_path)
                else:
                    safe.append(bin_path)
    return safe, vuln


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bins_dir", type=str)
    p.add_argument("tok_files_path", type=str,
                   default='./vulberta_tokenizer_config/')
    p.add_argument("-o", "--out", default="./out/")
    p.add_argument('-w', "--workers", type=int, default=None)
    p.add_argument('-c', '--consumers', type=int, default=None)
    args = p.parse_args()

    safe, vuln = discover_bins(args.bins_dir, regex=r'.*opt2.*')
    print(f"Discovered {len(safe)} safe binaries and {
          len(vuln)} vulnerable ones.")

    filters = {'exclude_plt': True, 'exclude_no_xref': True}

    out_safe = os.path.join(args.out, 'safe/')
    out_vuln = os.path.join(args.out, 'vuln/')
    os.makedirs(out_safe, exist_ok=True)
    os.makedirs(out_vuln, exist_ok=True)

    print("\n--- Processing SAFE Binaries ---")
    processed_safe = make_db(
        safe,
        label_mode='safe',
        out_path=out_safe,
        workers=args.workers,
        consumers=args.consumers,
        tokenizer_args={'path': args.tok_files_path},
        filters=filters
    )

    print("\n--- Processing VULN Binaries ---")

    # so that we can keep the hash diff consistent later
    vuln_bins = [re.sub('/patch/', '/vulnerable/', safe_bin_path)
                 for safe_bin_path in processed_safe]

    make_db(
        vuln_bins,
        label_mode='vuln',
        out_path=out_vuln,
        workers=args.workers,
        consumers=args.consumers,
        tokenizer_args={'path': args.tok_files_path},
        filters=filters,
    )

    print("Done.")
