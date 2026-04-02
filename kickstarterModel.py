# -*- coding: utf-8 -*-

# Import necessary packages
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
    mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, classification_report,
    roc_curve, auc
)

# Load dataset 
df = pd.read_csv(r"C:\Users\popul\Desktop\All Projects\Startup-analysis project\kickstarter_data_with_features.csv")

df.head()

# Shape and columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.columns)

# Data types
print(df.dtypes)

# Initial drop for completely unnecessary columns
to_drop = [
    "Unnamed: 0", "id", "photo", "name", "blurb", "slug",
    "currency", "currency_symbol", "currency_trailing_code",
    "state_changed_at", "created_at", "creator", "location",
    "profile", "urls", "source_url", "friends", "is_starred",
    "is_backing", "permissions", "deadline_weekday",
    "state_changed_at_weekday", "created_at_weekday",
    "launched_at_weekday", "deadline_day", "deadline_hr",
    "state_changed_at_month", "state_changed_at_day",
    "state_changed_at_yr", "state_changed_at_hr",
    "created_at_month", "created_at_day", "created_at_yr",
    "created_at_hr", "launched_at_day", "launched_at_hr",
    "launch_to_state_change"
]

df.drop(to_drop, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# Second round of refining columns/dataset
to_drop_2 = [
    "deadline", "launched_at", "name_len_clean", "blurb_len_clean",
]

df.drop(to_drop_2, axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)

# Check for missing values
nan_count = np.sum(df.isnull(), axis=0)
print(nan_count)

# Drop rows with missing values
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

nan_count = np.sum(df.isnull(), axis=0)
print(nan_count)

# Convert state to binary classification
df['state'] = (df['state'] == 'successful').astype(int)
df = df.rename(columns={'state': 'succeeded'})

# One-hot encode categorical columns
to_encode = ['country', 'category']
df = pd.get_dummies(df, columns=to_encode)

# Convert bool cols to int64
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Extract days from timedelta columns
df['create_to_launch'] = pd.to_timedelta(df['create_to_launch']).dt.days
df['launch_to_deadline'] = pd.to_timedelta(df['launch_to_deadline']).dt.days

df.head()

# One-hot encode year columns
year_to_encode = ['deadline_yr', 'launched_at_yr']
df = pd.get_dummies(df, columns=year_to_encode)

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

df.head()

# Scale continuous columns
continuous_cols = [
    'backers_count', 'goal', 'pledged', 'static_usd_rate',
    'usd_pledged', 'name_len', 'blurb_len',
    'create_to_launch', 'launch_to_deadline'
]

scaler = StandardScaler()
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])

df.head()

# EDA
df.describe(include='all')

# Correlation heatmap
corMat = df.corr(method='pearson')
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("Correlation matrix using heatmap")
plt.show()

# Drop leakage columns
red_flags = ['pledged', 'usd_pledged', 'backers_count', 'spotlight']
df = df.drop(columns=red_flags)

# Prepare features and labels
X = df.loc[:, df.columns != 'succeeded'].values.astype(np.float32)
y = df['succeeded'].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test).unsqueeze(1)

# Wrap in DataLoader for batching
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# Define Neural Network
num_features = X_train.shape[1]

class NeuralNet(nn.Module):
    def __init__(self, num_features):
        super(NeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),   # Hidden layer 1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),             # Hidden layer 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),             # Hidden layer 3
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),              # Output layer
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

nn_model = NeuralNet(num_features)
print(nn_model)

# Optimizer and loss function
optimizer = optim.SGD(nn_model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

# Training loop
num_epochs = 150
train_acc_history = []
train_loss_history = []

t0 = time.time()

for epoch in range(num_epochs):
    nn_model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = nn_model(X_batch)
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}', flush=True)

t1 = time.time()
print('Elapsed time: %.2fs' % (t1 - t0), flush=True)

# Evaluate on test set
nn_model.eval()
with torch.no_grad():
    y_pred_prob = nn_model(X_test_t).numpy()
    test_loss = loss_fn(torch.tensor(y_pred_prob), y_test_t).item()

y_pred = (y_pred_prob > 0.5).astype(int)
test_acc = (y_pred.flatten() == y_test).mean()

print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Training accuracy vs final test accuracy
plt.figure(figsize=(6, 4))
plt.plot(train_acc_history, marker='o', label='Training Accuracy')
plt.scatter(len(train_acc_history)-1, train_acc_history[-1], s=150, color='red', label='Final Accuracy')
plt.title('Training Accuracy Over Epochs and Final Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Training loss vs final test loss
plt.figure(figsize=(6, 4))
plt.plot(train_loss_history, marker='o', label='Training Loss')
plt.scatter(len(train_loss_history)-1, train_loss_history[-1], s=150, color='orange', label='Final Loss')
plt.title('Training Loss Over Epochs and Final Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

plt.figure(figsize=(6, 5))
disp.plot(values_format='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()