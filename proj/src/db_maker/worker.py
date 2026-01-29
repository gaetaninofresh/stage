import re
import queue
import hashlib
from typing import Dict, Literal
from decompiler.decompile import Decompiler
from tokenizer import load_tokenizer
import traceback
from parquet_writer import IncrementalDBWriter
_tokenizer = None


def init_worker(tokenizer_args: dict = None):
    global _tokenizer
    _tokenizer = load_tokenizer(**(tokenizer_args or {}))


def _clean_code(src: str) -> str:
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', src)
    code = code.replace('\n', '').replace('\t', '')
    return code.strip()


def _batch_list(lst, n):
    if n >= len(lst):
        yield lst
    else:
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


def decompile_bin(bin_path, out_queue, filters):
    CHUNK_SIZE = 256
    func_buffer = {'code': [], 'hash': []}
    try:
        with Decompiler(bin_path) as d:
            fs = d.enum_f()
            d.clean_f_name(fs)
            filter_funcs = d.filter_funcs(**(filters or {}))
            fs = [f for f in fs if f not in filter_funcs]

            # Decompile everything into local memory first.
            for f in fs:
                try:
                    decomp = d.decompile_func(f['addr'], format='raw')
                    code = _clean_code(str(decomp))
                    if not code:
                        continue
                    else:
                        code_hash = int.from_bytes(
                            hashlib.blake2b(
                                code.encode(encoding='utf-8',
                                            errors='replace'),
                                digest_size=8
                            ).digest(),
                            byteorder='little'
                        )
                        func_buffer['code'].append(code)
                        func_buffer['hash'].append(code_hash)
                    # Push data to tokenizer's queue
                    if len(func_buffer['code']) > CHUNK_SIZE:
                        out_queue.put(
                            {'code': func_buffer['code'], 'hash': func_buffer['hash'], 'bin': bin_path, 'done': False})
                        func_buffer = {'code': [], 'hash': []}
                except Exception as e:
                    print(f'--- Exception during function decompilation: {e}')
                continue
            # clear buffer
            out_queue.put(
                {'code': func_buffer['code'], 'hash': func_buffer['hash'], 'bin': bin_path, 'done': True})

    except Exception as e:
        print(f'--- Exception during binary decompilation: {e}')
        raise


def consumer_tokenize(
        tokenize_queue,
        done_queue,
        label_mode: Literal,
        tokenizer_args: Dict,
        writer_args: Dict
):
    tokenizer = load_tokenizer(**(tokenizer_args or {}))
    writer = IncrementalDBWriter(**(writer_args or {}))

    BATCH_SIZE = 2048
    code_buffer = {'code': [], 'hash': []}
    metadata_buffer = []
    shutdown = False

    while True:
        flush_buffer = False

        # Fill buffer
        try:
            item = tokenize_queue.get(timeout=0.5)
            if item is None:
                shutdown = True
                # Flush remaining data before exiting
                if code_buffer['code']:
                    flush_buffer = True
            else:
                code = item['code']
                if not code:
                    if item['done']:
                        done_queue.put(item['bin'])
                    continue
                code_hash = item['hash']
                bin_path = item['bin']

                # Create metadata - mark last item with binary path
                meta = [None] * len(code)
                if item['done']:
                    meta[-1] = bin_path

                code_buffer['code'].extend(code)
                code_buffer['hash'].extend(code_hash)
                metadata_buffer.extend(meta)

        except queue.Empty:
            pass

        # Check if buffer is full
        if len(code_buffer['code']) >= BATCH_SIZE:
            flush_buffer = True

        # Batched tokenization
        if flush_buffer and code_buffer['code']:
            try:
                batch_end_i = min(len(code_buffer['code']), BATCH_SIZE)

                batch_code = code_buffer['code'][:batch_end_i]
                code_buffer['code'] = code_buffer['code'][batch_end_i:]

                batch_hash = code_buffer['hash'][:batch_end_i]
                code_buffer['hash'] = code_buffer['hash'][batch_end_i:]

                meta_processed = metadata_buffer[:batch_end_i]
                metadata_buffer = metadata_buffer[batch_end_i:]

                # Tokenize the batch
                batch_encs = tokenizer.encode_batch(batch_code)

                if batch_encs:
                    ids = [e.ids for e in batch_encs]
                    masks = [e.attention_mask for e in batch_encs]

                    data = {
                        'ids': ids,
                        'attention_mask': masks
                    }
                    hashes = [hash for hash in batch_hash]

                    # Write to parquet
                    writer.write_batch(
                        data,
                        hashes,
                        labels=[0 if label_mode == 'safe' else 1] * len(ids)
                    )

                    # Notify about completed binaries
                    meta_processed_set = set(meta_processed)
                    meta_processed_set.discard(None)
                    for bin_path in meta_processed_set:
                        done_queue.put(bin_path)

            except Exception as e:
                print(f'--- Exception during tokenization: {e}')
                traceback.print_exc()

        # Exit if shutdown requested and buffer is empty
        if shutdown and not code_buffer['code']:
            break

    writer.close()
    print(f"Tokenizer process exiting (mode: {label_mode})")
