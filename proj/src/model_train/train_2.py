import os
import json
import torch
import torch.nn as nn
import numpy as np

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    fbeta_score,
    roc_auc_score,
    roc_curve
)

from helpers import calculate_weights, FocalLoss

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def save_metrics(metrics, filename):
    with open(filename, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def recall_at_fpr(y_true, y_score, target_fpr):
    """
    Exact Recall@FPR using ROC curve
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        valid = fpr <= target_fpr
        if not np.any(valid):
            return 0.0, None
        idx = np.argmax(tpr[valid])
        return tpr[valid][idx], thresholds[valid][idx]
    except ValueError:
        return 0.0, None


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate(
    model,
    val_loader,
    device,
    fpr_targets=(0.01, 0.05, 0.10),
    fixed_threshold=0.5
):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # -------------------------
    # Recall@FPR metrics
    # -------------------------
    recall_fpr = {}
    thresholds_fpr = {}

    for fpr in fpr_targets:
        r, t = recall_at_fpr(all_labels, all_probs, fpr)
        recall_fpr[fpr] = r
        thresholds_fpr[fpr] = t

    # -------------------------
    # Diagnostic fixed-threshold metrics
    # -------------------------
    preds = (all_probs >= fixed_threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        preds,
        average="binary",
        pos_label=1,
        zero_division=0
    )
    f2 = fbeta_score(
        all_labels,
        preds,
        beta=2,
        pos_label=1,
        zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # -------------------------
    # Logging
    # -------------------------
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)
    for fpr in fpr_targets:
        print(
            f"Recall@FPR≤{int(fpr*100):2d}% : "
            f"{recall_fpr[fpr]:.4f} "
            f"(thr={thresholds_fpr[fpr]})"
        )
    print("-" * 72)
    print(
        f"F2@0.5: {f2:.4f} | "
        f"Recall: {recall:.4f} | "
        f"Precision: {precision:.4f}"
    )
    print(f"ROC-AUC: {auc:.4f}")
    print(f"FN: {fn} | TP: {tp}")
    print("=" * 72 + "\n")

    return {
        "recall_fpr": recall_fpr,
        "thresholds_fpr": thresholds_fpr,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "accuracy": acc,
        "auc": auc,
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
def train_vulberta(
    model,
    train_loader,
    val_loader,
    weights,
    loss_func="ce",
    epochs=20,
    warmup_epochs=3,
    device="cuda",
    out_dir="./results",
    fpr_targets=(0.01, 0.05, 0.10),
    main_fpr=0.05,
    patience=4,
):
    os.makedirs(out_dir, exist_ok=True)

    ACCUMULATION_STEPS = 16
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir=os.path.join(out_dir, "runs"))
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    total_steps = (len(train_loader) // ACCUMULATION_STEPS) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if loss_func == "ce":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss_func == "fl":
        criterion = FocalLoss()
    else:
        raise ValueError("Unsupported loss function")

    best_recall = 0.0
    patience_ctr = 0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        optimizer.zero_grad()

        for i, batch in enumerate(loop):
            input_ids = batch["ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUMULATION_STEPS

            if (i + 1) % ACCUMULATION_STEPS == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if global_step % 10 == 0:
                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar(
                    "Train/LR", scheduler.get_last_lr()[0], global_step)

            global_step += 1

        avg_loss = total_loss / len(train_loader)

        print(f"\nValidating Epoch {epoch+1}...")
        metrics = evaluate(
            model,
            val_loader,
            device,
            fpr_targets=fpr_targets
        )

        # TensorBoard logging
        for fpr in fpr_targets:
            writer.add_scalar(
                f"Val/Recall@FPR_{int(fpr*100)}",
                metrics["recall_fpr"][fpr],
                epoch
            )

        writer.add_scalar("Val/F2", metrics["f2"], epoch)
        writer.add_scalar("Val/AUC", metrics["auc"], epoch)
        writer.add_scalar("Val/FN", metrics["fn"], epoch)

        save_metrics(
            {
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **{
                    f"recall_fpr_{int(fpr*100)}": metrics["recall_fpr"][fpr]
                    for fpr in fpr_targets
                },
                "f2": metrics["f2"],
                "auc": metrics["auc"],
                "fn": metrics["fn"],
            },
            filename=os.path.join(out_dir, "training_logs.jsonl")
        )

        # -------------------------
        # Early stopping
        # -------------------------
        current_recall = metrics["recall_fpr"][main_fpr]

        if epoch < warmup_epochs:
            print(f"Warmup epoch {
                  epoch+1}/{warmup_epochs},  skipping early stop")
            continue

        if current_recall > best_recall + 1e-3:
            best_recall = current_recall
            patience_ctr = 0
            print(
                f">>> New best model "
                f"(Recall@FPR≤{int(main_fpr*100)}%: {best_recall:.4f})"
            )
            model.save_pretrained(os.path.join(
                out_dir, f"VulBERta_best_fpr_{main_fpr*100}"))
        else:
            patience_ctr += 1
            print(f"No improvement ({patience_ctr}/{patience})")

        if patience_ctr >= patience:
            print(
                f"\nEarly stopping triggered. "
                f"Best Recall@FPR≤{int(main_fpr*100)}% = {best_recall:.4f}"
            )
            break

    writer.close()
    print("Training complete.")
    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("db", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("-o", "--out", default="./out")
    parser.add_argument("-l", "--loss_function", default="ce")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        "parquet",
        data_files={
            "train": args.db + "train.parquet",
            "val": args.db + "validate.parquet",
        }
    )

    dataset.set_format(
        type="torch",
        columns=["ids", "attention_mask", "labels"]
    )

    weights = calculate_weights(dataset, device)

    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        dataset["val"],
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )

    train_vulberta(
        model,
        train_loader,
        val_loader,
        weights=weights,
        loss_func=args.loss_function,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        device=device,
        out_dir=args.out,
    )
