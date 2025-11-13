
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(
    'mahdin70/codebert-devign-code-vulnerability-detector')
model.to(device)
model.eval()

d = datasets.load_from_disk('./dbs/stack_buff_ofw/proc/tokenized_test')

in_ids = torch.tensor(d['input_ids'])
masks = torch.tensor(d['attention_mask'])
labels = torch.tensor(d['label'])

in_ids = in_ids.to(device)
masks = masks.to(device)
labels = labels.to(device)

dataset = TensorDataset(in_ids, masks, labels)
batch_size = 32  # reduce if still OOM
dataloader = DataLoader(dataset, batch_size=batch_size)

all_preds = []
all_labels = []

with torch.no_grad():  # disable gradient computation
    for batch in dataloader:
        b_input_ids, b_masks, b_labels = batch
        outputs = model(input_ids=b_input_ids, attention_mask=b_masks)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(b_labels.cpu().numpy())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print("\n=== Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
