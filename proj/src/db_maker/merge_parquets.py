import os
import glob
import argparse
import pandas as pd


def merge_and_decouple(input_dirs, output_dir, hash_col, label_col):
    files = []
    for d in input_dirs:
        files.extend(glob.glob(os.path.join(
            d, "**", "*.parquet"), recursive=True))

    if not files:
        return

    # Load all files
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    df_zeros = df[df[label_col] == 0]
    df_others = df[df[label_col] != 0]

    # Concatenate with 0 first
    full_df = pd.concat([df_zeros, df_others], ignore_index=True)

    # Drop duplicates on hash, keeping the top row (Label 0)
    full_df.drop_duplicates(subset=[hash_col], keep='first', inplace=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    full_df.to_parquet(os.path.join(output_dir, "merged.parquet"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', nargs='+', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--hash_col', default='hash')
    parser.add_argument('--label_col', default='labels')

    args = parser.parse_args()
    merge_and_decouple(args.inputs, args.output, args.hash_col, args.label_col)
