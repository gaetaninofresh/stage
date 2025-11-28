import os
import re
import torch
from typing import Dict, List, Any, Tuple
from argparse import ArgumentParser
from pathlib import Path
from decompile import Decompiler
from tokenizer import load_tokenizer
from tqdm import tqdm


class DecompDB():
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['ids']) == len(
            self.encodings['attention_mask']) == len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def is_executable_binary(path):
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        if header.startswith(b"MZ"):
            return True
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
        filters: Dict[str, Any] | None) -> Tuple[List, List]:
    d = Decompiler(bin)

    fs = d.enum_f()
    d.clean_f_name(fs)

    fs = [f for f in d.enum_f() if f not in (d.filter_funcs(
        **filters) if filters is not None else d.filter_funcs())]

    tk = load_tokenizer()
    rel_fs = d.relevant_fs()

    encodings = labels = []

    for f in rel_fs:
        decomp = d.decompile_func(f['addr'], format='raw')
        print(decomp)
        decomp = str(decomp).strip()
        code = clean(decomp)
        label = 1 if re.match('bad', bin) else 0

        encoding = tk.encode(code)
        encoding = {'ids': encoding.ids,
                    'attention_mask': encoding.attention_mask}
        encodings.append(encoding)
        labels.append(label)

    return encodings, labels


def make_db(bins: List[str], filters: Dict[str, Any] | None) -> DecompDB:
    bin_db = {}
    encodings = labels = []
    for bin in tqdm(bins):
        tqdm.write(f'Processing {bin}')
        encodings, labels = process_bin(bin, filters)
        encodings.extend(encodings)
        labels.extend(labels)
    decomp_db = DecompDB(encodings, labels)
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

    db = make_db(bins, {})
    for k in db:
        print()
