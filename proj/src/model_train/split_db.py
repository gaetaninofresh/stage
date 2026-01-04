import pandas as pd
from sklearn.model_selection import train_test_split

file_path = './dbs/full_padded_db.parquet'

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

train_df.to_parquet('./dbs/train.parquet')
val_df.to_parquet('./dbs/validate.parquet')
test_df.to_parquet('./dbs/test.parquet')

print("Original Distribution:\n", df['labels'].value_counts(normalize=True))
print("Train Distribution:\n", train_df['labels'].value_counts(normalize=True))
print("Test Distribution:\n", test_df['labels'].value_counts(normalize=True))
