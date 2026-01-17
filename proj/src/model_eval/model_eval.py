import pandas as pd
from tabulate import tabulate
import os
import warnings
import json
from argparse import ArgumentParser
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning


def is_vulberta_mlp_folder(path: str) -> bool:
    files = set(os.listdir(path))
    return "config.json" in files and bool(files & {"model.safetensors"})


def print_summary(summary, sort_by="f1", descending=True):
    df = summary.copy()

    float_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 4))

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    print("\n=== Evaluation Summary ===")
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
    print("==========================\n")


class VulBERTaEvaluator:
    def __init__(self, model_path, dataset, batch_size=32, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.to(self.device).eval()
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

    def compute_metrics(self, y_true, y_pred, y_prob):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": p,
            "recall": r,
            "f1": f1,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "degenerate_predictions": len(set(y_pred)) == 1
        }

        if len(set(y_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        return metrics

    def evaluate(self):
        all_true, all_pred, all_prob = [], [], []

        total_rows = self.dataset.count_rows()
        with tqdm(total=total_rows, desc="Evaluating samples", unit="samples") as pbar:
            for batch in self.batch_iterator():
                input_ids, attention_mask, labels = self.preprocess_batch(
                    batch)
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                preds = logits.argmax(dim=-1).cpu().tolist()
                true = labels.cpu().tolist()

                all_true.extend(true)
                all_pred.extend(preds)
                all_prob.extend(probs)
                pbar.update(len(true))
        metrics = {
            "accuracy": accuracy_score(all_true, all_pred),
            "precision": precision_recall_fscore_support(all_true, all_pred, average="binary", zero_division=0)[0],
            "recall": precision_recall_fscore_support(all_true, all_pred, average="binary", zero_division=0)[1],
            "f1": precision_recall_fscore_support(all_true, all_pred, average="binary", zero_division=0)[2],
            "confusion_matrix": confusion_matrix(all_true, all_pred).tolist(),
        }

        if len(set(all_true)) == 2:
            metrics["roc_auc"] = roc_auc_score(all_true, all_prob)

        return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to Parquet DB")
    parser.add_argument("model_dir", type=str,
                        help="Folder with multiple VulBERTa-MLP subfolders")
    parser.add_argument("-o", "--out", default="./out",
                        type=str, help="Output folder for JSON metrics")
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    assert os.path.exists(args.db), f"DB not found: {args.db}"
    assert os.path.isdir(args.model_dir), f"Model dir not found: {
        args.model_dir}"

    os.makedirs(args.out, exist_ok=True)

    dataset = ds.dataset(args.db, format='parquet')

    # Find valid models
    model_paths = [os.path.join(args.model_dir, d) for d in os.listdir(args.model_dir)
                   if os.path.isdir(os.path.join(args.model_dir, d))]
    valid_models = [p for p in model_paths if is_vulberta_mlp_folder(p)]
    invalid_models = [p for p in model_paths if p not in valid_models]
    if invalid_models:
        print("Skipping non-VulBERTa-MLP folders:")
        for p in invalid_models:
            print("  -", p)
    if not valid_models:
        raise RuntimeError("No valid VulBERTa-MLP models found.")

    summary = []

    for model_path in tqdm(valid_models, desc="Evaluating models"):
        model_name = os.path.basename(model_path)
        evaluator = VulBERTaEvaluator(
            model_path=model_path, dataset=dataset, batch_size=args.batch)
        metrics = evaluator.evaluate()

        out_file = os.path.join(args.out, f"{model_name}.json")
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=2)
        tqdm.write(f"Results for {model_name} saved to {out_file}")

        summary.append({"model": model_name, **metrics})

    summary_df = pd.DataFrame(summary)
    summary_csv = os.path.join(args.out, "summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")
    print_summary(summary_df, sort_by="f1")
