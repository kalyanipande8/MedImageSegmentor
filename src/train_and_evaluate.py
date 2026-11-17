"""Real U-Net training loop (Adam, BCE-with-logits loss) on a
train/test split, with the standard Dice-score metric computed on the
held-out test set."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .unet_model import SmallUNet


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-6) -> float:
    intersection = (pred_mask * true_mask).sum()
    return float((2 * intersection + eps) / (pred_mask.sum() + true_mask.sum() + eps))


def train_and_evaluate(images: np.ndarray, masks: np.ndarray, test_frac: float = 0.2,
                        epochs: int = 15, seed: int = 967) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    n = len(images)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train = torch.tensor(images[train_idx]).unsqueeze(1)
    y_train = torch.tensor(masks[train_idx]).unsqueeze(1)
    X_test = torch.tensor(images[test_idx]).unsqueeze(1)
    y_test = torch.tensor(masks[test_idx]).unsqueeze(1)

    model = SmallUNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    batch_size = 32
    n_train = len(X_train)
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(X_train[batch_idx])
            loss = criterion(logits, y_train[batch_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_preds = (torch.sigmoid(test_logits) > 0.5).float()

    dice_scores = [dice_score(test_preds[i, 0].numpy(), y_test[i, 0].numpy()) for i in range(len(test_idx))]
    mean_dice = float(np.mean(dice_scores))

    return {"n_train": n_train, "n_test": len(test_idx), "mean_dice_score": round(mean_dice, 4),
            "median_dice_score": round(float(np.median(dice_scores)), 4)}
