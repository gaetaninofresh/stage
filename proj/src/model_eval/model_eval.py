import pandas as pd
from tabulate import tabulate
import os
import warnings
import json
import numpy as np
from argparse import ArgumentParser
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    fbeta_score,
    roc_curve,
    average_precision_score,
    matthews_corrcoef
)
from sklearn.exceptions import UndefinedMetricWarning

import eval_graphs


def recall_at_fpr(y_true, y_score, target_fpr):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        valid = fpr <= target_fpr
        if not np.any(valid):
            return 0.0, None
        idx = np.argmax(tpr[valid])
        return tpr[valid][idx], thresholds[valid][idx]
    except ValueError:
        return 0.0, None


def print_summary(summary, sort_by="f2", descending=True):
    df = summary.copy()
    float_cols = [
        "accuracy", "precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "mcc"
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(
                x, 4) if isinstance(x, (float, int)) else x)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    print("\n=== Evaluation Summary (Sorted by F2) ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print("=========================================\n")


class VulBERTaEvaluator:
    def __init__(self, model_path, dataset, batch_size=32, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.to(self.device).eval()

        if hasattr(torch, "compile"):
            pass

        self.dataset = dataset
        self.batch_size = batch_size

    def batch_iterator(self):
        for batch in self.dataset.to_batches(batch_size=self.batch_size):
            yield batch

    def preprocess_batch(self, batch):
        input_ids = [torch.tensor(x, dtype=torch.long)
                     for x in batch["ids"].to_pylist()]
        attention_mask = [torch.tensor(x, dtype=torch.long)
                          for x in batch["attention_mask"].to_pylist()]
        labels = torch.tensor(
            batch["labels"].to_pylist(), dtype=torch.long, device=self.device)

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=1).to(self.device)

        attention_mask = pad_sequence(
            attention_mask, batch_first=True, padding_value=0).to(self.device)

        return input_ids, attention_mask, labels

    def evaluate(self, fpr_targets=(0.01, 0.05, 0.10)):
        all_true, all_prob = [], []

        # Detect if we can use AMP
        use_amp = torch.cuda.is_available() and self.device == "cuda"
        dtype = torch.float16 if use_amp else torch.float32

        total_rows = self.dataset.count_rows()
        with tqdm(total=total_rows, desc="Evaluating", unit="samples", leave=False) as pbar:
            for batch in self.batch_iterator():
                input_ids, attention_mask, labels = self.preprocess_batch(
                    batch)

                with torch.no_grad():
                    # AMP Context for performance
                    with torch.amp.autocast('cuda', enabled=use_amp, dtype=dtype):
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                true = labels.cpu().tolist()

                all_true.extend(true)
                all_prob.extend(probs)
                pbar.update(len(true))

        y_true = np.array(all_true)
        y_prob = np.array(all_prob)

        # --- 1. Global Metrics (Threshold Independent) ---
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = 0.0

        try:
            pr_auc = average_precision_score(y_true, y_prob)
        except ValueError:
            pr_auc = 0.0

        # --- 2. Threshold Optimization (Maximize F2) ---
        threshold_range = np.linspace(0.01, 0.99, 99)
        best_f2 = 0.0
        best_thresh = 0.5
        stats_per_thresh = []

        for thresh in threshold_range:
            y_pred_t = (y_prob >= thresh).astype(int)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UndefinedMetricWarning)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred_t, average="binary", pos_label=1, zero_division=0
                )

            # Calc f2
            if (4 * precision + recall) == 0:
                f2_t = 0.0
            else:
                f2_t = (5 * precision * recall) / ((4 * precision) + recall)

            if f2_t > best_f2:
                best_f2 = f2_t
                best_thresh = thresh

            stats_per_thresh.append({
                "threshold": thresh,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f2": f2_t
            })

        # --- 3. Final Metrics at Optimal Threshold ---
        y_pred_opt = (y_prob >= best_thresh).astype(int)

        acc = accuracy_score(y_true, y_pred_opt)

        # Recalculate precision/recall at best_thresh
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            prec_opt, rec_opt, f1_opt, _ = precision_recall_fscore_support(
                y_true, y_pred_opt, average="binary", pos_label=1, zero_division=0
            )

        mcc = matthews_corrcoef(y_true, y_pred_opt)

        cm = confusion_matrix(y_true, y_pred_opt)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # --- 4. Recall at Fixed FPRs ---
        recall_fpr = {}
        thresholds_fpr = {}
        flat_metrics = {}

        for fpr in fpr_targets:
            r, t = recall_at_fpr(y_true, y_prob, fpr)
            recall_fpr[fpr] = r
            thresholds_fpr[fpr] = t
            flat_metrics[f"recall_fpr_{int(fpr*100)}"] = r
            flat_metrics[f"thresh_fpr_{
                int(fpr*100)}"] = t if t is not None else 0.0

        metrics = {
            "optimal_threshold": best_thresh,
            "accuracy": acc,
            "precision": prec_opt,
            "recall": rec_opt,
            "f1": f1_opt,
            "f2": best_f2,
            "mcc": mcc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fn": int(fn),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "recall_fpr": recall_fpr,
            "thresholds_fpr": thresholds_fpr,
            **flat_metrics
        }

        raw_data = {
            "y_true": y_true,
            "y_prob": y_prob,
            "stats_per_thresh": stats_per_thresh
        }

        return metrics, raw_data


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to Parquet DB")
    parser.add_argument("model_dir", type=str,
                        help="Root folder to recursively search for VulBERTa models")
    parser.add_argument("-o", "--out", default="./out",
                        type=str, help="Output folder for JSON metrics and Graphs")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    assert os.path.exists(args.db), f"DB not found: {args.db}"
    assert os.path.isdir(args.model_dir), f"Model dir not found: {
        args.model_dir}"

    os.makedirs(args.out, exist_ok=True)
    dataset = ds.dataset(args.db, format='parquet')

    valid_models = []
    print(f"Searching for models recursively in: {args.model_dir} ...")
    for root, dirs, files in os.walk(args.model_dir):
        files_set = set(files)
        if "config.json" in files_set and "model.safetensors" in files_set:
            rel_path = os.path.relpath(root, args.model_dir)
            if rel_path == ".":
                model_name = os.path.basename(os.path.normpath(args.model_dir))
            else:
                model_name = rel_path.replace(os.sep, "_")
            valid_models.append((root, model_name))

    if not valid_models:
        raise RuntimeError("No valid VulBERTa models found recursively.")

    print(f"Found {len(valid_models)} models.")
    summary = []
    all_models_raw_data = {}

    for model_path, model_name in tqdm(valid_models, desc="Evaluating models"):
        tqdm.write(f"Processing: {model_name}")

        evaluator = VulBERTaEvaluator(
            model_path=model_path, dataset=dataset, batch_size=args.batch)
        metrics, raw_data = evaluator.evaluate(fpr_targets=(0.01, 0.05, 0.10))

        # Save Metrics JSON
        out_file = os.path.join(args.out, f"{model_name}.json")
        with open(out_file, "w") as f:
            def default(o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                    return int(o)
                elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                    return float(o)
                raise TypeError(f"Object of type {
                                o.__class__.__name__} is not JSON serializable")

            json.dump(metrics, f, indent=2, default=default)

        # Generate Individual Diagnostic Graphs
        eval_graphs.plot_model_diagnostics(
            raw_data["y_true"],
            raw_data["y_prob"],
            raw_data["stats_per_thresh"],
            model_name,
            args.out
        )

        # Store for Comparison
        all_models_raw_data[model_name] = {
            "y_true": raw_data["y_true"],
            "y_prob": raw_data["y_prob"]
        }

        summary_row = {k: v for k, v in metrics.items(
        ) if not isinstance(v, (dict, list))}
        summary.append({"model": model_name, **summary_row})

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.out, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Generate Cross-Model Comparison Graphs
    if len(valid_models) > 1:
        print("Generating comparison graphs...")
        eval_graphs.plot_model_comparison(
            all_models_raw_data, summary_df, args.out)

    print(f"\nSummary saved to {summary_csv}")
    print_summary(summary_df, sort_by="f2")
