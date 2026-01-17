import pandas as pd
import numpy as np
import pyarrow as pa
import json
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Tuple

special_tokens = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 4}


def wrap_tokens(tokens: List[int], mask_len: int = 1024):
    toks = [special_tokens['<s>']]
    toks.extend(tokens)

    toks.append(special_tokens['</s>'])

    toks.extend([special_tokens['<pad>']] * (mask_len - len(tokens)))
    return toks


def probe_same_token(token: int, tok_n: int, mask_len: int = 1024) -> Tuple[List[int], List[int]]:

    tokens = wrap_tokens([token]*tok_n, mask_len)
    mask = ([1] * (tok_n+2))
    mask.extend([0] * (mask_len - tok_n))

    return tokens, mask


def probe_rand_tokens(token_set: List[int], tok_n: int, mask_len: int = 1024) -> Tuple[List[int], List[int]]:
    token_set = set(token_set).difference(special_tokens.values())
    token_set = [*token_set]

    rng = np.random.default_rng()

    tokens = rng.choice(token_set, size=tok_n).tolist()

    mask = [1] * (len(tokens) + 2)

    mask.extend([0] * (mask_len - len(tokens)))

    tokens = wrap_tokens(tokens, mask_len+2)

    return tokens, mask


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str,
                        help="Path to json token dictionary")
    parser.add_argument('-o', '--out', type=str, help='Path to output dir')
    args = parser.parse_args()

    data_rand_homo = {
        'ids': [], 'attention_mask': [], 'labels': []}

    with open(args.source) as f:
        tok_dict = json.load(f)
        tok_set = [*tok_dict.values()]

        for k in range(0, 16):
            for i in tqdm(range(0, 1024, 8)):
                tok_rand_homo, mask_rand_homo = probe_rand_tokens(
                    tok_set, i)

                data_rand_homo['ids'].append(tok_rand_homo)
                data_rand_homo['attention_mask'].append(mask_rand_homo)
                data_rand_homo['labels'].append(0)  # useless

        df_rand_homo = pd.DataFrame(data_rand_homo)
        df_rand_homo.to_parquet(args.out+'/random_token_homo.parquet')

        schema = pa.schema([
            ('ids', pa.list_(pa.int64())),
            ('attention_mask', pa.list_(pa.int64())),
            ('labels', pa.int64())
        ])

        writer = pa.parquet.ParquetWriter(
            f"{args.out}/same_token_probe.parquet", schema)
        batch_buffer = {'ids': [], 'attention_mask': [], 'labels': []}

        for tok in tqdm(tok_set):
            for i in range(0, 1024, 32):
                tok_same, mask_same = probe_same_token(tok, i)

                batch_buffer['ids'].append(tok_same)
                batch_buffer['attention_mask'].append(mask_same)
                batch_buffer['labels'].append(0)  # useless

                if len(batch_buffer['ids']) >= 500:

                    table = pa.Table.from_pydict(
                        batch_buffer, schema=schema)
                    writer.write_table(table)
                    batch_buffer = {'ids': [],
                                    'attention_mask': [], 'labels': []}
    if batch_buffer['ids']:
        table = pa.Table.from_pydict(batch_buffer, schema=schema)
        writer.write_table(table)
    writer.close()
