import re
from typing import List, Tuple, Dict, Any
from decompile import Decompiler
from tokenizer import load_tokenizer
import traceback

_tokenizer = None


def init_worker(tokenizer_args: dict = None):
    """
    initializer for worker processes; load tokenizer once per worker.
    tokenizer_args is passed through from main (can be None).
    """
    global _tokenizer
    _tokenizer = load_tokenizer(**(tokenizer_args or {}))


def _clean_code(src: str) -> str:
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', src)
    code = code.replace('\n', '').replace('\t', '')
    return code.strip()


def process_bin_chunk(bin_paths: List[str], filters: Dict[str, Any] | None = None
                      ) -> Tuple[Dict[str, List[List[int]]], List[int], List[str]]:
    """
    Process a chunk of binaries.
    """
    global _tokenizer
    all_ids = []
    all_masks = []
    all_labels = []
    processed_bins = []

    for bin_path in bin_paths:
        try:
            with Decompiler(bin_path) as d:
                fs = d.enum_f()
                d.clean_f_name(fs)
                filter_funcs = d.filter_funcs(**(filters or {}))
                fs = [f for f in fs if f not in filter_funcs]

                rel_fs = d.relevant_fs(check_sec_calls=True if re.match(
                    '.*good.*', bin_path) else False)

                code_list = []
                for f in rel_fs:
                    decomp = d.decompile_func(f['addr'], format='raw')
                    code = _clean_code(str(decomp))
                    code_list.append(code)

            if not code_list:
                # nothing decompiled in this binary
                processed_bins.append(bin_path)
                continue

            # Use batch encode if available; fallback to per-item encode
            try:
                encodings = _tokenizer.encode_batch(code_list)
                ids_list = [enc.ids for enc in encodings]
                masks_list = [enc.attention_mask for enc in encodings]
            except Exception:
                # fallback
                ids_list = []
                masks_list = []
                for code in code_list:
                    enc = _tokenizer.encode(code)
                    ids_list.append(enc.ids)
                    masks_list.append(enc.attention_mask)

            label_val = 1 if re.match('.*bad.*', bin_path) else 0
            labels_for_bin = [label_val] * len(ids_list)
            all_ids.extend(ids_list)
            all_masks.extend(masks_list)
            all_labels.extend(labels_for_bin)
            processed_bins.append(bin_path)

        except Exception as e:
            # return partial results for already processed binaries in the chunk
            traceback.print_exc()
            continue

    encodings_out = {"ids": all_ids, "attention_mask": all_masks}
    return encodings_out, all_labels, processed_bins
