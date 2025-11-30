from argparse import ArgumentParser
from toch.utils.data import DataLoader
from datasets import Dataset
from db import DecompDB
import pyarrow.feather as feather

table = feather.read_table("decomp.arrow")
dataset = Dataset(table)

dataset.rename_column('ids', 'input_ids')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('db', type=str)
    parser.add_argument('-o', "--out", default='./out/db',
                        action='store', type=str)
    args = parser.parse_args()
