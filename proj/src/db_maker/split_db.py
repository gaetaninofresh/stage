import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to parquet db")
    parser.add_argument("-o", "--out", default="./out",
                        type=str, help="")
    args = parser.parse_args()

    file_path = args.db
    out_path = args.out

    df = pd.read_parquet(file_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['labels'],
        random_state=23
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['labels'],
        random_state=23
    )

    train_df.to_parquet(out_path + 'train.parquet', index=False)
    val_df.to_parquet(out_path + 'validate.parquet', index=False)
    test_df.to_parquet(out_path + 'test.parquet', index=False)

    print("Original Distribution:\n",
          df['labels'].value_counts(normalize=True))
    print("Train Distribution:\n",
          train_df['labels'].value_counts(normalize=True))
    print("Test Distribution:\n",
          test_df['labels'].value_counts(normalize=True))
