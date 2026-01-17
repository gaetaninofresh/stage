import argparse
import json
import sys
import os
import pandas as pd
import numpy as np
from collections import Counter


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract Parquet stats to JSON")
    parser.add_argument('-i', '--input', required=True,
                        help="Input parquet file path")
    parser.add_argument('-o', '--output', required=True,
                        help="Output JSON file path")
    return parser.parse_args()


def get_token_counts(series_of_ids):
    # Flatten list of lists into a single numpy array of tokens
    all_tokens = np.concatenate(series_of_ids.values)
    return Counter(all_tokens)


def get_statistics(df):
    stats = {}
    stats['total_rows'] = len(df)

    # 1. Sequence Length Analysis
    if 'attention_mask' in df.columns:
        df['real_len'] = df['attention_mask'].apply(sum)
        stats['seq_len_general'] = df['real_len'].describe().to_dict()

        if 'labels' in df.columns:
            stats['avg_len_by_label'] = {
                str(k): v for k, v in df.groupby('labels')['real_len'].mean().items()
            }
            stats['seq_len_stats_by_label'] = {
                str(label): metrics.to_dict()
                for label, metrics in df.groupby('labels')['real_len'].describe().iterrows()
            }

    # 2. Token Distribution Analysis
    if 'ids' in df.columns:
        # General Counts
        global_counter = get_token_counts(df['ids'])
        stats['vocab_size'] = len(global_counter)
        stats['total_tokens'] = sum(global_counter.values())
        stats['top_100_tokens_general'] = {
            str(k): v for k, v in global_counter.most_common(100)
        }

        # Class-Specific and Comparative Analysis
        if 'labels' in df.columns:
            stats['token_stats_by_label'] = {}
            label_counters = {}
            label_totals = {}

            # First pass: Collect counts per label
            unique_labels = sorted(df['labels'].unique())
            for label in unique_labels:
                subset = df[df['labels'] == label]
                lbl_counter = get_token_counts(subset['ids'])
                total_tokens = sum(lbl_counter.values())

                label_counters[label] = lbl_counter
                label_totals[label] = total_tokens

                stats['token_stats_by_label'][str(label)] = {
                    'vocab_size': len(lbl_counter),
                    'total_tokens': total_tokens,
                    'top_50_tokens': {
                        str(k): v for k, v in lbl_counter.most_common(50)
                    }
                }

            top_100_global_ids = [t[0]
                                  for t in global_counter.most_common(100)]

            diff_list = []

            def get_prob(token, label):
                if label_totals[label] == 0:
                    return 0.0
                return label_counters[label][token] / label_totals[label]

            for token in top_100_global_ids:
                token_str = str(token)

                # Calculate probs per label
                probs = {lbl: get_prob(token, lbl) for lbl in unique_labels}

                entry = {
                    "token": token_str,
                    "probabilities": {str(k): v for k, v in probs.items()},
                    "diff_magnitude": 0.0  # Default
                }

                # Calculate Difference
                if len(unique_labels) == 2:
                    l0, l1 = unique_labels[0], unique_labels[1]
                    diff = probs[l0] - probs[l1]
                    entry['diff_L0_minus_L1'] = diff
                    entry['diff_magnitude'] = abs(diff)
                else:
                    # Fallback for multi-class: Max diff between any two labels
                    prob_values = list(probs.values())
                    entry['diff_magnitude'] = max(
                        prob_values) - min(prob_values)

                diff_list.append(entry)

            # 2. Sort by Magnitude (Greatest difference first)
            diff_list.sort(key=lambda x: x['diff_magnitude'], reverse=True)

            # 3. Take Top 10
            top_10_diffs = diff_list[:10]

            stats['top_10_highest_diff_tokens'] = top_10_diffs

            stats['exclusive_tokens'] = {}
            label_vocabs = {label: set(counter.keys())
                            for label, counter in label_counters.items()}

            for label in unique_labels:
                current_vocab = label_vocabs[label]
                other_vocabs = set().union(
                    *[v for l, v in label_vocabs.items() if l != label])
                unique_tokens = current_vocab - other_vocabs

                unique_tokens_with_counts = [
                    (token, label_counters[label][token]) for token in unique_tokens
                ]
                unique_tokens_sorted = sorted(
                    unique_tokens_with_counts, key=lambda x: x[1], reverse=True)

                stats['exclusive_tokens'][str(label)] = {
                    'unique_token_count': len(unique_tokens),
                    'top_50_exclusive_tokens': {
                        str(k): v for k, v in unique_tokens_sorted[:50]
                    }
                }

    if 'labels' in df.columns:
        stats['class_counts'] = {
            str(k): v for k, v in df['labels'].value_counts().items()
        }
        stats['class_distribution'] = {
            str(k): v for k, v in df['labels'].value_counts(normalize=True).items()
        }

    return stats


def main():
    args = get_args()

    if not os.path.exists(args.input):
        sys.exit(f"Error: Input file '{args.input}' not found.")

    print(f"Reading from: {args.input}")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        sys.exit(f"Error reading parquet file: {e}")

    print("Analyzing data...")
    final_stats = get_statistics(df)

    print(f"Saving to: {args.output}")
    try:
        with open(args.output, 'w') as f:
            json.dump(final_stats, f, cls=NpEncoder, indent=4)
        print("Done.")
    except Exception as e:
        sys.exit(f"Error saving JSON: {e}")


if __name__ == "__main__":
    main()
