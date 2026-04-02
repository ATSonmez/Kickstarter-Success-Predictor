# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, classification_report,
    roc_curve, auc
)

# ── 1. Load & Clean Data ─────────────────────────────────────────────────────

df = pd.read_csv(r"C:\Users\popul\Desktop\All Projects\Startup-analysis project\kickstarter_data_with_features.csv")

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop columns that are identifiers, metadata, or redundant time breakdowns
columns_to_drop = [
    "Unnamed: 0", "id", "photo", "name", "blurb", "slug",
    "currency", "currency_symbol", "currency_trailing_code",
    "state_changed_at", "created_at", "creator", "location",
    "profile", "urls", "source_url", "friends", "is_starred",
    "is_backing", "permissions",
    "deadline_weekday", "state_changed_at_weekday", "created_at_weekday",
    "launched_at_weekday", "deadline_day", "deadline_hr",
    "state_changed_at_month", "state_changed_at_day", "state_changed_at_yr",
    "state_changed_at_hr", "created_at_month", "created_at_day",
    "created_at_yr", "created_at_hr", "launched_at_day", "launched_at_hr",
    "launch_to_state_change",
    "deadline", "launched_at", "name_len_clean", "blurb_len_clean",
]
df.drop(columns=columns_to_drop, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Feature Engineering ────────────────────────────────────────────────────

# Binary target: 1 = successful, 0 = everything else
df['succeeded'] = (df['state'] == 'successful').astype(int)
df.drop(columns='state', inplace=True)

# Extract days from timedelta strings
df['create_to_launch'] = pd.to_timedelta(df['create_to_launch']).dt.days
df['launch_to_deadline'] = pd.to_timedelta(df['launch_to_deadline']).dt.days

# One-hot encode categorical and year columns
df = pd.get_dummies(df, columns=['country', 'category', 'deadline_yr', 'launched_at_yr'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Scale continuous columns
continuous_cols = [
    'backers_count', 'goal', 'pledged', 'static_usd_rate',
    'usd_pledged', 'name_len', 'blurb_len',
    'create_to_launch', 'launch_to_deadline'
]
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

# ── 3. EDA ────────────────────────────────────────────────────────────────────

corMat = df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Drop data-leakage columns (unknowable before campaign ends)
leakage_cols = ['pledged', 'usd_pledged', 'backers_count', 'spotlight']
df.drop(columns=leakage_cols, inplace=True)

# ── 4. Train/Test Split ──────────────────────────────────────────────────────

X = df.drop(columns='succeeded').values.astype(np.float32)
y = df['succeeded'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# ── 5. Model Definition ──────────────────────────────────────────────────────

class KickstarterNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

num_features = X_train.shape[1]
model = KickstarterNet(num_features)
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# ── 6. Training ───────────────────────────────────────────────────────────────

NUM_EPOCHS = 150
train_acc_history = []
train_loss_history = []

t0 = time.time()

for epoch in range(NUM_EPOCHS):
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
        preds = (output > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    avg_loss = epoch_loss / total
    avg_acc = correct / total
    train_loss_history.append(avg_loss)
    train_acc_history.append(avg_acc)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}', flush=True)

elapsed = time.time() - t0
print(f'Training complete in {elapsed:.2f}s', flush=True)

# ── 7. Evaluation ─────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_t).numpy()
    test_loss = loss_fn(torch.tensor(y_pred_prob), y_test_t).item()

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
axes[0].set_title('Training Accuracy Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].grid(True)
axes[0].legend()

axes[1].plot(train_loss_history, color='orange', label='Training Loss')
axes[1].set_title('Training Loss Over Epochs')
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
