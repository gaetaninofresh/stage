import os
import datasets as ds
from pathlib import Path
from transformers import AutoTokenizer
from argparse import ArgumentParser
import random
from collections import defaultdict
import json


def strat_split_db(json_db, ratio, seed):
    random.seed(seed)

    classes = defaultdict(list)
    for entry in json_db:
        label = entry['label']
        classes[label].append(entry)

    train_set = []
    test_set = []

    for label, cases in classes.items():
        random.shuffle(cases)
        split_index = round(len(cases) * ratio)

        train_set.extend(cases[:split_index])
        test_set.extend(cases[split_index:])
    return train_set, test_set


def tokenize_dataset(train_set, test_set):
    data = {
        'train': train_set,
        'test': test_set
    }
    dataset = ds.load_dataset('json', data_files=data)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

    def tokenize(example):
        return tokenizer(
            example["code"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tok_ds = dataset.map(tokenize, batched=True)
    tok_ds = tok_ds.remove_columns(['code', 'file'])
    return tok_ds


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src', type=str)
    parser.add_argument('-o', '--out', action='store',
                        type=str, default='.')
    args = parser.parse_args()
    src = str(Path(args.src).resolve())
    dir = Path(args.out)

    set_path = str(dir.resolve()) + '/json_ds'
    tok_path = str(dir.resolve()) + '/tokenized_ds'

    if not Path.exists(dir):
        os.mkdir(dir)
    if not Path.exists(Path(set_path)):
        os.mkdir(set_path)
    if not Path.exists(Path(tok_path)):
        os.mkdir(tok_path)

    data = [json.loads(line) for line in open(src)]
    train, test = strat_split_db(data, 0.7, 23)

    with open(f'{set_path}/train.json', 'w+') as f:
        json.dump(train, f, indent=2)
    with open(f'{set_path}/test.json', 'w+') as f:
        json.dump(test, f, indent=2)

    tok_ds = tokenize_dataset(
        f'{set_path}/train.json', f'{set_path}/test.json')

    train = tok_ds['train']
    test = tok_ds['test']
    train.save_to_disk(f'{tok_path}/tokenized_train')
    test.save_to_disk(f'{tok_path}/tokenized_test')
