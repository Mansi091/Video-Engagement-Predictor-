"""Engagement prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from torch.utils.data import DataLoader, TensorDataset


class EngagementMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class ModelResult:
    name: str
    auc: float
    f1: float
    accuracy: float


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 128,
    dropout: float = 0.3,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[EngagementMLP, ModelResult]:
    torch.manual_seed(seed)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * (1 - val_fraction))
    tr, va = idx[:split], idx[split:]

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]

    model = EngagementMLP(X.shape[1], hidden_dim, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_va, dtype=torch.float32))
        probs = torch.sigmoid(logits).numpy()

    auc = roc_auc_score(y_va, probs) if len(set(y_va)) > 1 else 0.5
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(y_va, preds, zero_division=0)
    acc = accuracy_score(y_va, preds)

    return model, ModelResult("pytorch_mlp", auc, f1, acc)


def train_xgboost_baseline(X: np.ndarray, y: np.ndarray, seed: int = 42) -> ModelResult:
    model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    probs = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(set(y)) > 1 else 0.5
    f1 = f1_score(y, preds, zero_division=0)
    acc = accuracy_score(y, preds)
    model.fit(X, y)
    return ModelResult("xgboost_baseline", auc, f1, acc)


def bias_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    segment: np.ndarray,
) -> dict:
    """Segment-level AUC for bias mitigation analysis."""
    report = {}
    for seg in np.unique(segment):
        mask = segment == seg
        if mask.sum() < 20 or len(set(y_true[mask])) < 2:
            continue
        report[str(seg)] = {
            "n": int(mask.sum()),
            "positive_rate": float(y_true[mask].mean()),
            "auc": float(roc_auc_score(y_true[mask], y_prob[mask])),
        }
    return report
