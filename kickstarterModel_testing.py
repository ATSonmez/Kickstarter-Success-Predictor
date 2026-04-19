# -*- coding: utf-8 -*-
# Version with only Step 1 (Adam optimizer) and Step 2 (Validation split + Early stopping)
# No BatchNorm, no class imbalance weighting

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from services.preprocessing import KickstarterPreprocessor
from models.nn_model import KickstarterNet

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

# ── 1. Load Data ─────────────────────────────────────────────────────────────

df = pd.read_csv("kickstarter_data_with_features.csv")

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Shared preprocessing (FND-05) ─────────────────────────────────────────

preprocessor = KickstarterPreprocessor()
X, y = preprocessor.fit_transform(df)

print(f"After preprocessing: {X.shape[0]} rows, {X.shape[1]} features")

# ── 3. Train/Test/Val Split (Step 2) ─────────────────────────────────────────

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ── 4. Model (Original architecture, no BatchNorm — testing variant) ──────────
# Note: uses canonical KickstarterNet from backend/models/nn_model.py (with BatchNorm).
# This file is for exploratory/testing runs only; no artifacts are saved.

num_features = X_train.shape[1]
model = KickstarterNet(num_features)
print(model)

# Step 1: Adam optimizer instead of SGD
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

# ── 5. Training with Early Stopping (Step 2) ─────────────────────────────────

NUM_EPOCHS = 500
PATIENCE = 15

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

# ── 6. Evaluation ─────────────────────────────────────────────────────────────

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

# ── 7. Plots ──────────────────────────────────────────────────────────────────

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
