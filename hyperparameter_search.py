# -*- coding: utf-8 -*-
# Hyperparameter search for KickstarterNet
# Tests combinations of learning rate, hidden layer sizes, dropout, and batch size
# Reports the best configuration by F1 score

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# ── 1. Data Preparation (same as main model) ─────────────────────────────────

df = pd.read_csv("kickstarter_data_with_features.csv")

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
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df['succeeded'] = (df['state'] == 'successful').astype(int)
df.drop(columns='state', inplace=True)

df['create_to_launch'] = pd.to_timedelta(df['create_to_launch']).dt.days
df['launch_to_deadline'] = pd.to_timedelta(df['launch_to_deadline']).dt.days

df = pd.get_dummies(df, columns=['country', 'category', 'deadline_yr', 'launched_at_yr'])
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

continuous_cols = [
    'backers_count', 'goal', 'pledged', 'static_usd_rate',
    'usd_pledged', 'name_len', 'blurb_len',
    'create_to_launch', 'launch_to_deadline'
]
scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

leakage_cols = ['pledged', 'usd_pledged', 'backers_count', 'spotlight']
df.drop(columns=leakage_cols, inplace=True)

X = df.drop(columns='succeeded').values.astype(np.float32)
y = df['succeeded'].values.astype(np.float32)

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_val_t = torch.tensor(X_val)
y_val_t = torch.tensor(y_val).unsqueeze(1)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test).unsqueeze(1)

num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()
pos_weight = torch.tensor([num_negative / num_positive])

num_features = X_train.shape[1]

# ── 2. Model Definition ──────────────────────────────────────────────────────

class KickstarterNet(nn.Module):
    def __init__(self, num_features, hidden_sizes, dropouts):
        super().__init__()
        layers = []
        in_size = num_features
        for h_size, drop in zip(hidden_sizes, dropouts):
            layers.extend([
                nn.Linear(in_size, h_size),
                nn.BatchNorm1d(h_size),
                nn.ReLU(),
                nn.Dropout(drop),
            ])
            in_size = h_size
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ── 3. Training Function ─────────────────────────────────────────────────────

def train_and_evaluate(lr, hidden_sizes, dropouts, batch_size):
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = KickstarterNet(num_features, hidden_sizes, dropouts)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    patience = 25

    for epoch in range(500):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_t)
            v_loss = loss_fn(val_output, y_val_t).item()

        scheduler.step(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        y_pred_prob = torch.sigmoid(model(X_test_t)).numpy()

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    test_acc = (y_pred == y_test).mean()
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'epochs': epoch + 1,
        'best_val_loss': best_val_loss,
    }

# ── 4. Search Space ──────────────────────────────────────────────────────────

search_space = {
    'lr':           [0.0005, 0.001, 0.005],
    'hidden_sizes': [(64, 32, 16), (128, 64, 32), (128, 64, 16)],
    'dropouts':     [(0.3, 0.2, 0.1), (0.2, 0.1, 0.05), (0.4, 0.2, 0.1)],
    'batch_size':   [128, 256],
}

combos = list(product(
    search_space['lr'],
    search_space['hidden_sizes'],
    search_space['dropouts'],
    search_space['batch_size'],
))

print(f"Testing {len(combos)} hyperparameter combinations...\n")

# ── 5. Run Search ────────────────────────────────────────────────────────────

results = []
best_f1 = 0
best_config = None

t0 = time.time()

for i, (lr, hidden_sizes, dropouts, batch_size) in enumerate(combos):
    config = {
        'lr': lr,
        'hidden_sizes': hidden_sizes,
        'dropouts': dropouts,
        'batch_size': batch_size,
    }

    metrics = train_and_evaluate(lr, hidden_sizes, dropouts, batch_size)
    config.update(metrics)
    results.append(config)

    status = "*** NEW BEST ***" if metrics['f1'] > best_f1 else ""
    if metrics['f1'] > best_f1:
        best_f1 = metrics['f1']
        best_config = config

    print(f"[{i+1}/{len(combos)}] LR={lr}, Hidden={hidden_sizes}, "
          f"Drop={dropouts}, Batch={batch_size} → "
          f"F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, "
          f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f} "
          f"{status}", flush=True)

elapsed = time.time() - t0
print(f"\nSearch complete in {elapsed:.1f}s")

# ── 6. Results ────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)
print(f"Learning Rate:  {best_config['lr']}")
print(f"Hidden Sizes:   {best_config['hidden_sizes']}")
print(f"Dropouts:       {best_config['dropouts']}")
print(f"Batch Size:     {best_config['batch_size']}")
print(f"Epochs Trained: {best_config['epochs']}")
print(f"─────────────────────────────")
print(f"Test Accuracy:  {best_config['accuracy']:.4f}")
print(f"Precision:      {best_config['precision']:.4f}")
print(f"Recall:         {best_config['recall']:.4f}")
print(f"F1 Score:       {best_config['f1']:.4f}")

# Show top 5
print("\n" + "=" * 70)
print("TOP 5 CONFIGURATIONS BY F1 SCORE")
print("=" * 70)

results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)

for i, r in enumerate(results_sorted[:5]):
    print(f"\n#{i+1}: F1={r['f1']:.4f}, Acc={r['accuracy']:.4f}, "
          f"Prec={r['precision']:.4f}, Rec={r['recall']:.4f}")
    print(f"    LR={r['lr']}, Hidden={r['hidden_sizes']}, "
          f"Drop={r['dropouts']}, Batch={r['batch_size']}, "
          f"Epochs={r['epochs']}")
