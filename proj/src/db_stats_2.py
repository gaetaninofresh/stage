import pandas as pd
import argparse
import os
import glob
import json
import numpy as np
from pathlib import Path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def calculate_stats(file_path, output_dir):
    print(f"Processing {file_path}...")

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    stats = {
        "db_name": Path(file_path).name,
        "total_rows": len(df),
    }

    # ---------------------------------------------------------
    # 1. General Stats & Sequence Lengths
    # ---------------------------------------------------------
    if 'labels' in df.columns:
        stats['class_counts'] = df['labels'].value_counts().to_dict()
        stats['class_distribution'] = df['labels'].value_counts(
            normalize=True).to_dict()

    if 'attention_mask' in df.columns:
        df['real_len'] = df['attention_mask'].apply(np.sum)
        stats['seq_len_general'] = df['real_len'].describe().to_dict()

        if 'labels' in df.columns:
            avg_len = df.groupby('labels')['real_len'].mean()
            stats['avg_len_by_label'] = avg_len.to_dict()

    # ---------------------------------------------------------
    # 2. Token Pre-processing (Explode & Deduplicate)
    # ---------------------------------------------------------
    unique_rows_per_token = pd.DataFrame()
    if 'ids' in df.columns and 'labels' in df.columns:
        work_df = df[['labels', 'ids']].copy()
        work_df['row_id'] = work_df.index

        exploded = work_df.explode('ids')
        unique_rows_per_token = exploded.drop_duplicates(
            subset=['row_id', 'ids'])

    # ---------------------------------------------------------
    # 3. Exclusive Token Tracking (Unique to 1 Label)
    # ---------------------------------------------------------
    if not unique_rows_per_token.empty:
        token_label_distribution = unique_rows_per_token.groupby('ids')[
            'labels'].nunique()
        tokens_unique_to_one_label = token_label_distribution[token_label_distribution == 1].index

        exclusive_data = unique_rows_per_token[unique_rows_per_token['ids'].isin(
            tokens_unique_to_one_label)]
        token_row_counts = exclusive_data.groupby(['labels', 'ids']).size()
        total_rows_per_label = df['labels'].value_counts()

        unique_token_stats = {}
        if not token_row_counts.empty:
            unique_labels = token_row_counts.index.get_level_values(
                'labels').unique()

            for lbl in unique_labels:
                lbl_key = str(lbl)
                unique_token_stats[lbl_key] = {}
                lbl_total_docs = total_rows_per_label[lbl]

                # Sort by frequency
                lbl_tokens = token_row_counts.loc[lbl].sort_values(
                    ascending=False)

                for token, count in lbl_tokens.items():
                    percentage = (count / lbl_total_docs) * 100
                    unique_token_stats[lbl_key][str(token)] = {
                        "row_count": count,
                        "percentage_of_label_rows": round(percentage, 4)
                    }

        stats["unique_label_tokens"] = unique_token_stats

    # ---------------------------------------------------------
    # 4. Common Token Stats (Global Top 100)
    # ---------------------------------------------------------
    top_100_global_ids = []

    if not unique_rows_per_token.empty:
        global_token_counts = unique_rows_per_token['ids'].value_counts()
        top_100_global = global_token_counts.head(100)
        top_100_global_ids = top_100_global.index.tolist()

        stats['top_100_common_tokens_global'] = {}
        for token_id, count in top_100_global.items():
            stats['top_100_common_tokens_global'][str(token_id)] = {
                'row_count': count,
                'share': round(count / len(df), 4)
            }

    # ---------------------------------------------------------
    # 5. Per-Label Top 50 & Distribution Differences
    # ---------------------------------------------------------
    if not unique_rows_per_token.empty:
        stats['top_50_tokens_per_label'] = {}

        label_token_counts = unique_rows_per_token.groupby(
            ['labels', 'ids']).size()
        unique_lbls = df['labels'].value_counts().index.tolist()
        label_shares_for_global_tokens = {}

        for lbl in unique_lbls:
            lbl_key = str(lbl)
            lbl_total = len(df[df['labels'] == lbl])

            if lbl in label_token_counts.index:
                lbl_counts = label_token_counts.loc[lbl]

                # A. Top 50 for this label
                top_50 = lbl_counts.sort_values(ascending=False).head(50)
                stats['top_50_tokens_per_label'][lbl_key] = {}

                for t_id, t_count in top_50.items():
                    stats['top_50_tokens_per_label'][lbl_key][str(t_id)] = {
                        'row_count': t_count,
                        'share': round(t_count / lbl_total, 4)
                    }

                # B. Pre-calculate shares for Global Top 100 within this label
                current_label_shares = {}
                for global_id in top_100_global_ids:
                    c = lbl_counts.get(global_id, 0)
                    current_label_shares[global_id] = c / lbl_total
                label_shares_for_global_tokens[lbl] = current_label_shares

        # C. Compare Global Top 100 distribution between Top 2 Labels
        if len(unique_lbls) >= 2:
            l1, l2 = unique_lbls[0], unique_lbls[1]
            diff_key = f"dist_diff_global_top_100_{l1}_vs_{l2}"
            stats[diff_key] = {}

            for t_id in top_100_global_ids:
                share_1 = label_shares_for_global_tokens[l1].get(t_id, 0.0)
                share_2 = label_shares_for_global_tokens[l2].get(t_id, 0.0)

                stats[diff_key][str(t_id)] = {
                    f'share_{l1}': round(share_1, 4),
                    f'share_{l2}': round(share_2, 4),
                    'diff_pct_points': round(share_1 - share_2, 4),
                    'abs_diff': round(abs(share_1 - share_2), 4)
                }

    # ---------------------------------------------------------
    # Save Output
    # ---------------------------------------------------------
    base_name = Path(file_path).stem
    out_name = f"{base_name}_stat.json"
    out_path = os.path.join(output_dir, out_name)

    os.makedirs(output_dir, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=4, cls=NpEncoder)

    print(f"Saved stats to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate merged stats for Parquet DBs")
    parser.add_argument("input_dir", type=str,
                        help="Directory containing .parquet files")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save .json stat files")

    args = parser.parse_args()

    search_path = os.path.join(args.input_dir, "*.parquet")
    files = glob.glob(search_path)

    if not files:
        print(f"No .parquet files found in {args.input_dir}")
    else:
        for f in files:
            calculate_stats(f, args.output_dir)
