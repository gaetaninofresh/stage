import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, fbeta_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from argparse import ArgumentParser
from helpers import calculate_weights, FocalLoss
from torch.utils.tensorboard import SummaryWriter

os.makedirs("results", exist_ok=True)


def save_metrics(metrics, filename="results/training_logs.jsonl"):
    with open(filename, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def evaluate(model, val_loader, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            # Store only vulnerable probabilities
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Apply custom decision threshold
    preds = (all_probs >= threshold).astype(int)

    # --- Metrics ---
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    # Vulnerable metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, preds, average='binary', pos_label=1, zero_division=0
    )

    # F2 Score
    f2 = fbeta_score(all_labels, preds, beta=2, pos_label=1, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Average confidence on True Positives
    mask_tp = (all_labels == 1) & (preds == 1)
    avg_prob_tp = all_probs[mask_tp].mean() if mask_tp.sum() > 0 else 0.0

    print(f"\n" + "="*60)
    print(f"VALIDATION REPORT (Threshold: {threshold})")
    print("="*60)
    print(f"Confusion Matrix:")
    print(f"   [[ TN: {tn:<5}  FP: {fp:<5} ]]")
    print(f"   [[ FN: {fn:<5}  TP: {tp:<5} ]]")
    print("-" * 60)
    print(f"Class 1 (Vulnerable) Metrics:")
    print(f"   F2-Score:      {f2:.4f}")
    print(f"   Recall:        {recall:.4f}")
    print(f"   Precision:     {precision:.4f}")
    print(f"   ROC-AUC:       {auc:.4f}")
    print("-" * 60)
    print(f"Diagnostics:")
    print(f"   Avg Confidence (TP): {avg_prob_tp:.4f}")
    print(f"   Total Vulnerable:    {np.sum(all_labels == 1)}")
    print("="*60 + "\n")

    return {
        "accuracy": acc,
        "f1": f1,
        "f2": f2,
        "recall": recall,
        "precision": precision,
        "auc": auc,
        "fn": int(fn),
        "tp": int(tp)
    }


def train_vulberta(
        model,
        train_loader,
        val_loader,
        weights,
        loss_func='ce',
        epochs=20,
        device='cuda',
        out_dir='./results/',
        early_stop=False
):

    ACCUMULATION_BATCH_SIZE = 16
    model.to(device)
    scaler = torch.amp.GradScaler('cuda')

    # TENSORBOARD SETUP
    log_dir = os.path.join(out_dir, "runs/")
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging avviato su: {log_dir}")
    global_step = 0

    # Learning rate from VulBerta paper
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    steps_for_epoch = len(train_loader) // ACCUMULATION_BATCH_SIZE
    total_steps = steps_for_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    if loss_func == 'ce':
        if weights is not None:
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif loss_func == 'fl':
        criterion = FocalLoss()
    else:
        raise ValueError(f'{loss_func} isn\'t a supported mode')

    print(f"Starting Training on {device} with Gradient Accumulation ({
          ACCUMULATION_BATCH_SIZE} steps)...")

    best_f2 = 0.0

    with open(os.path.join(out_dir, 'loss.csv'), 'w') as f:
        f.write("step,epoch,loss\n")

    for epoch in range(epochs):

        total_train_loss = 0
        model.train()
        loop = tqdm(train_loader, leave=True)

        for i, batch in enumerate(loop):

            input_ids = batch['ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(
                    outputs, 'logits') else outputs

                loss = criterion(logits, labels) / ACCUMULATION_BATCH_SIZE

            scaler.scale(loss).backward()

            current_loss = loss.item() * ACCUMULATION_BATCH_SIZE
            total_train_loss += current_loss

            if global_step % 10 == 0:
                writer.add_scalar("Train/Loss", current_loss, global_step)
                writer.add_scalar(
                    "Train/LR", scheduler.get_last_lr()[0], global_step)

            global_step += 1

            if (i+1) % ACCUMULATION_BATCH_SIZE == 0 or i+1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            loop.set_description(f"Epoch {epoch+1}")

        avg_train_loss = total_train_loss / len(train_loader)

        print(f"\nValidating Epoch {epoch+1}...")

        # TODO: Temporary treshold, must back up choice
        metrics = evaluate(model, val_loader, device, threshold=0.5)

        writer.add_scalar("Val/F2", metrics['f2'], epoch)
        writer.add_scalar("Val/Recall", metrics['recall'], epoch)
        writer.add_scalar("Val/Precision", metrics['precision'], epoch)
        writer.add_scalar("Val/Accuracy", metrics['accuracy'], epoch)
        writer.add_scalar("Val/AUC", metrics['auc'], epoch)

        writer.add_scalar("Val/FalseNegatives", metrics['fn'], epoch)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **metrics
        }

        save_metrics(log_entry)
        print(f"Train Loss: {avg_train_loss:.4f} | Val F2: {
              metrics['f2']:.4f} | Val Recall: {metrics['recall']:.4f}")

        # Save based on F2-Score to minimize false negatives
        if metrics['f2'] > best_f2:
            best_f2 = metrics['f2']
            print(f">>> New Best Model (F2: {
                  best_f2:.4f})! Saving checkpoint...")
            model.save_pretrained("./results/vulberta_best_training")

    print("Training Complete. Saving final model...")
    model.save_pretrained(os.path.join(out_dir, "vulberta_best_final"))
    writer.close()

    return model


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to parquet db")
    parser.add_argument("model", type=str, help="Path to VulBerta")
    parser.add_argument("-o", "--out", default="./out", type=str, help="")
    args = parser.parse_args()

    db_path = args.db
    out_path = args.out
    model_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        "parquet",
        data_files={
            'train': db_path + 'train.parquet',
            'val': db_path + 'validate.parquet'
        }
    )
    dataset.set_format(
        type='torch',
        columns=['ids', 'attention_mask', 'labels']
    )

    weights = calculate_weights(dataset, device)

    # Optimized for my system
    train_loader = torch.utils.data.DataLoader(
        dataset['train'],
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset['val'],
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2
    )

    model = train_vulberta(
        model,
        train_loader,
        val_loader,
        weights=weights,
        epochs=5,
        device=device,
        out_dir=out_path
    )
