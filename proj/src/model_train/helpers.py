import torch.nn as nn
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_weights(dataset, device):
    train_labels = dataset['train']['labels']

    if torch.is_tensor(train_labels):
        train_labels = train_labels.numpy()

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )

    weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32).to(device)
    return weights_tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
