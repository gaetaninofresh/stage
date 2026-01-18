import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from argparse import ArgumentParser


os.makedirs("results", exist_ok=True)


def save_metrics(metrics, filename="results/training_logs.jsonl"):
    with open(filename, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def train_vulberta(model, train_loader, val_loader, epochs=3, device='cuda', out_dir='./results/', early_stop=False):
    model.to(device)
    scaler = torch.amp.GradScaler('cuda')

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()
    best_f1 = 0.0
    print(f"Starting Training on {device}...")

    glob_loss_i = 0
    f = open(out_dir + 'loss.csv', 'w')
    f.write("step,epoch,loss\n")

    for epoch in range(epochs):

        total_train_loss = 0
        loss_buffer = []

        model.train()
        loop = tqdm(train_loader, leave=True)

        i = 0  # useful later
        for i, batch in enumerate(loop):

            input_ids = batch['ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask=attention_mask)

                logits = outputs.logits if hasattr(
                    outputs, 'logits') else outputs

                loss = criterion(logits, labels)

                f.write(f"{glob_loss_i},{epoch+1},{loss.item()}\n")
                glob_loss_i += 1

                f.flush()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()

            if early_stop:
                # Stop on low loss to avoid overfitting on last 20 average
                if len(loss_buffer) < 20:
                    loss_buffer.append(loss.item())
                else:
                    if sum(loss_buffer) / 20 <= .1:
                        break
                    loss_buffer.pop(0)
                    loss_buffer.append(loss.item())

                loss_avg = sum(loss_buffer) / len(loss_buffer)

            loop.set_description(f"Epoch {epoch+1}")

            if early_stop:
                loop.set_postfix(train_loss=loss_avg)

        avg_train_loss = total_train_loss / i

        print(f"\nValidating Epoch {epoch+1}...")
        metrics = evaluate(model, val_loader, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            **metrics
        }

        save_metrics(log_entry)
        print(f"Train Loss: {avg_train_loss:.4f} |"
              f"Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            print(">>> New Best Model! Saving checkpoint...")
            model.save_pretrained("./results/vulberta_best_training")

    f.close()
    print("Training Complete. Saving final model...")
    model.save_pretrained(out_dir + "vulberta_best_final")


def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("db", type=str, help="Path to parquet db")
    parser.add_argument("model", type=str, help="Path to VulBerta")
    parser.add_argument("-o", "--out", default="./out",
                        type=str, help="")
    args = parser.parse_args()

    db_path = args.db
    out_path = args.out
    model_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("parquet", data_files={
                           'train': db_path + 'train.parquet', 'val': db_path + 'validate.parquet'})
    dataset.set_format(type='torch', columns=[
                       'ids', 'attention_mask', 'labels'])

    # Optimized for my system
    train_loader = torch.utils.data.DataLoader(
        dataset['train'], batch_size=4, shuffle=True, num_workers=3, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset['val'], batch_size=4, num_workers=3, pin_memory=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=2)

    train_vulberta(model, train_loader, val_loader,
                   epochs=5, device=device, out_dir=out_path)
