# -*- coding: utf-8 -*-

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from services.preprocessing import KickstarterPreprocessor
from models.nn_model import KickstarterNet

import json
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, classification_report,
    roc_curve, auc
)

MODELS_DIR = Path(__file__).parent / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────

df = pd.read_csv("kickstarter_data_with_features.csv")

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. EDA stats snapshot (computed BEFORE encoding — Pattern 5) ─────────────

df_clean = df.copy()
df_clean.dropna(inplace=True)
df_clean["succeeded"] = (df_clean["state"] == "successful").astype(int)

# Goal buckets per D-03 Claude's Discretion
goal_bins = [0, 1_000, 10_000, 100_000, float("inf")]
goal_labels = ["<$1k", "$1k-$10k", "$10k-$100k", ">$100k"]
df_clean["goal_bucket"] = pd.cut(
    df_clean["goal"], bins=goal_bins, labels=goal_labels, right=False
)
eda_stats = {
    "by_category": (
        df_clean.groupby("category")["succeeded"]
        .agg(success_rate="mean", count="count")
        .reset_index().to_dict(orient="records")
    ),
    "by_country": (
        df_clean.groupby("country")["succeeded"]
        .agg(success_rate="mean", count="count")
        .reset_index().to_dict(orient="records")
    ),
    "by_goal_bucket": (
        df_clean.groupby("goal_bucket", observed=True)["succeeded"]
        .agg(success_rate="mean", count="count")
        .reset_index().to_dict(orient="records")
    ),
}

# ── 3. Shared preprocessing (FND-01, FND-02, FND-05) ─────────────────────────

preprocessor = KickstarterPreprocessor()
X, y = preprocessor.fit_transform(df)

print(f"After preprocessing: {X.shape[0]} rows, {X.shape[1]} features")

# ── 4. Train/Test Split ──────────────────────────────────────────────────────

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

# ── 5. Model ─────────────────────────────────────────────────────────────────

model = KickstarterNet(num_features=X_train.shape[1])
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Class imbalance weighting
num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()
pos_weight = torch.tensor([num_negative / num_positive])
print(f"Class distribution — Negative: {num_negative}, Positive: {num_positive}, pos_weight: {pos_weight.item():.2f}")

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── 6. Training ───────────────────────────────────────────────────────────────

NUM_EPOCHS = 500
PATIENCE = 25  # Stop if val loss doesn't improve for this many epochs

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

t0 = time.time()

for epoch in range(NUM_EPOCHS):
    # ── Training phase ──
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X_batch.size(0)
        preds = (torch.sigmoid(output) > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = epoch_loss / total
    avg_acc = correct / total
    train_loss_history.append(avg_loss)
    train_acc_history.append(avg_acc)

    # ── Validation phase ──
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_t)
        v_loss = loss_fn(val_output, y_val_t).item()
        val_preds = (torch.sigmoid(val_output) > 0.5).float()
        v_acc = (val_preds == y_val_t).float().mean().item()

    val_loss_history.append(v_loss)
    val_acc_history.append(v_acc)

    # ── Reduce LR if val loss plateaus ──
    scheduler.step(v_loss)

    # ── Early stopping check ──
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict().copy()
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
              f'Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, '
              f'Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}', flush=True)

    if epochs_without_improvement >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1} (no val improvement for {PATIENCE} epochs)', flush=True)
        break

# Restore the best model weights
model.load_state_dict(best_model_state)

elapsed = time.time() - t0
print(f'Training complete in {elapsed:.2f}s', flush=True)

# ── 7. Evaluation ─────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_t)
    test_loss = loss_fn(y_pred_logits, y_test_t).item()
    y_pred_prob = torch.sigmoid(y_pred_logits).numpy()

y_pred = (y_pred_prob > 0.5).astype(int).flatten()
test_acc = (y_pred == y_test).mean()

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(train_acc_history, label='Training Accuracy')
axes[0].plot(val_acc_history, label='Validation Accuracy')
axes[0].set_title('Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(train_loss_history, color='orange', label='Training Loss')
axes[1].plot(val_loss_history, color='red', label='Validation Loss')
axes[1].set_title('Loss Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d', cmap='Blues', ax=ax)
ax.set_title("Confusion Matrix")
ax.grid(False)
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob.flatten())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ── 9. Save all 6 artifacts (FND-04) ─────────────────────────────────────────

# 1. Neural net state_dict
torch.save(model.state_dict(), MODELS_DIR / "kickstarter_nn.pt")

# 2. Scaler + feature columns (via preprocessor.save)
preprocessor.save(MODELS_DIR)

# 3. Balanced SHAP background sample — 100 rows (50 success + 50 failure)
rng = np.random.default_rng(42)
y_train_arr = y_train if isinstance(y_train, np.ndarray) else y_train.numpy()
X_train_arr = X_train if isinstance(X_train, np.ndarray) else X_train.numpy()
success_idx = np.where(y_train_arr == 1)[0]
failure_idx = np.where(y_train_arr == 0)[0]
bg_idx = np.concatenate([
    rng.choice(success_idx, 50, replace=False),
    rng.choice(failure_idx, 50, replace=False),
])
background_tensor = torch.tensor(X_train_arr[bg_idx], dtype=torch.float32)
torch.save(background_tensor, MODELS_DIR / "background.pt")

# 4. EDA stats
with open(MODELS_DIR / "eda_stats.json", "w") as f:
    json.dump(eda_stats, f, indent=2, default=str)

# 5. Model metadata
metadata = {
    "trained_at": pd.Timestamp.now().isoformat(),
    "accuracy": float(test_acc),
    "auc": float(roc_auc),
    "num_features": int(X_train.shape[1]),
    "feature_columns": preprocessor.feature_columns,
}
with open(MODELS_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Saved 6 artifacts to {MODELS_DIR}")
