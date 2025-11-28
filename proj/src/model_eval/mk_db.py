import os
import re
import torch
from typing import Dict, List, Any, Tuple
from argparse import ArgumentParser
from pathlib import Path
from decompiler.decompile import Decompiler
from db import DecompDB
from tokenizer import load_tokenizer
from tqdm import tqdm


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


def clean(code):
    '''
    Remove code comments
    '''
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    code = re.sub(pat, '', code)
    code = re.sub('\n', '', code)
    code = re.sub('\t', '', code)
    return code


def process_encodings(encodings):
    input_ids = []
    attention_mask = []
    for enc in encodings:
        input_ids.append(enc.ids)
        attention_mask.append(enc.attention_mask)
    return {'ids': input_ids, 'attention_mask': attention_mask}


def process_bin(
        bin: str,
        filters: Dict[str, Any] | None):
    d = Decompiler(bin)

    fs = d.enum_f()
    d.clean_f_name(fs)

    fs = [f for f in d.enum_f() if f not in (d.filter_funcs(
        **filters) if filters is not None else d.filter_funcs())]

    tk = load_tokenizer()
    rel_fs = d.relevant_fs(
        check_sec_calls=True if re.match('.*good.*', bin) else False
    )
    labels = []
    encodings = []
    for f in rel_fs:
        tqdm.write(f'\tAnalyzing {f['name']}')
        decomp = d.decompile_func(f['addr'], format='raw')
        decomp = str(decomp).strip()
        code = clean(decomp)
        labels.append(1 if re.match('.*bad.*', bin) else 0)  # Horrible
        enc = tk.encode(code)
        encodings.append(enc)
    encodings = process_encodings(encodings)
    return encodings, labels


def make_db(bins: List[str], filters: Dict[str, Any] | None = None) -> DecompDB:
    decomp_db = DecompDB({'ids': [], 'attention_mask': []}, [])
    for bin in tqdm(bins):
        tqdm.write(f'Processing {bin}')
        enc, labels = process_bin(bin, filters)
        decomp_db.append(enc, labels)
    return decomp_db


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('bins_dir', type=str)
    parser.add_argument('-o', "--out", default='./out/db',
                        action='store', type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.bins_dir):
        print(f'{args.bins_dir} is not a directory')
    if not Path(args.out).exists():
        os.mkdir(args.out)

    bins = []

    for root, dirs, files in os.walk(args.bins_dir):
        for bin in files:
            bin_path = os.path.join(root, bin)
            if is_executable_binary(bin_path):
                bins.append(bin_path)

    db = make_db(bins)
    db.save_arrow(f'{args.out}/db.arrow')
