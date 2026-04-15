# Testing Patterns

**Analysis Date:** 2026-04-14

## Test Framework

**Runner:**
- Python: No test framework configured (pytest or unittest not in dependencies)
- Frontend: No test framework configured (no Jest or Vitest in devDependencies)
- **Current approach:** Manual testing scripts only

**Assertion Library:**
- Python models use `sklearn.metrics` for assertions (e.g., `confusion_matrix`, `precision_score`, `recall_score`, `f1_score`)
- No assertion library framework for formal unit tests

**Run Commands:**
```bash
# Frontend
npm run dev              # Start development server
npm run build          # Build for production
npm run lint           # Run ESLint
npm run preview        # Preview production build

# Python models (manual)
python kickstarterModel.py          # Run main training
python kickstarterModel_testing.py  # Run testing variant
python hyperparameter_search.py     # Run hyperparameter search
```

## Test File Organization

**Location:**
- Separate files with `_testing.py` suffix: `kickstarterModel_testing.py` is a variant of main model
- No co-located test files (test files are in same directory as source)
- No dedicated `tests/` directory

**Naming:**
- Testing variant files: `{module}_testing.py` (e.g., `kickstarterModel_testing.py`)
- Hyperparameter search: `hyperparameter_search.py`
- Main model: `kickstarterModel.py`

**Structure:**
```
Kickstarter-Success-Predictor/
├── kickstarterModel.py              # Production model training
├── kickstarterModel_testing.py      # Testing variant (Step 1 & 2 only)
├── hyperparameter_search.py         # Hyperparameter tuning
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api.js
    │   ├── main.jsx
    │   └── pages/
    │       ├── PredictPage.jsx
    │       ├── DashboardPage.jsx
    │       ├── PerformancePage.jsx
    │       └── HistoryPage.jsx
```

## Test Structure

**Suite Organization:**
The codebase uses a procedural testing approach with inline assertions during training. No formal test suites exist.

**Main testing pattern in `kickstarterModel.py` (lines 1-285):**
```python
# ── 1. Load & Clean Data ─────────────────────────────────────────────────────
# Read CSV and drop irrelevant columns
df = pd.read_csv("kickstarter_data_with_features.csv")
df.drop(columns=columns_to_drop, inplace=True)
df.dropna(inplace=True)
print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
# Transform target and features
df['succeeded'] = (df['state'] == 'successful').astype(int)

# ── 3. EDA ────────────────────────────────────────────────────────────────────
# Visualize with correlation heatmap
corMat = df.corr(method='pearson')
plt.figure(figsize=(12, 10))
sns.heatmap(corMat, square=True)
plt.show()

# ── 4. Train/Test Split ──────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# ── 5. Model Definition ──────────────────────────────────────────────────────
class KickstarterNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

# ── 6. Training ───────────────────────────────────────────────────────────────
# Early stopping, learning rate scheduling, class imbalance weighting
for epoch in range(NUM_EPOCHS):
    model.train()
    # ... training loop
    model.eval()
    with torch.no_grad():
        # ... validation loop

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")

# ── 8. Plots ──────────────────────────────────────────────────────────────────
# Accuracy/loss curves, confusion matrix, ROC curve
plt.show()
```

**Patterns:**
- **Setup:** Data loading and preprocessing in sections 1-2
- **Teardown:** No explicit cleanup; models saved via state dict and reloaded
- **Assertion:** Metrics printed to console at epoch intervals and final evaluation
- **Visualization:** Matplotlib plots for manual verification (accuracy, loss, confusion matrix, ROC)

## Mocking

**Framework:** None configured

**Patterns:** No mocking in codebase. Models train on actual CSV data (`kickstarter_data_with_features.csv`)

**What to Mock:**
- Database calls (if testing backend routes without live DB)
- External API calls (not yet implemented in frontend)
- Model inference (if unit testing API endpoints)

**What NOT to Mock:**
- Data loading (use real CSV for training)
- Model layers (PyTorch components are low-level tested separately)
- Metrics calculation (sklearn metrics are well-tested libraries)

## Fixtures and Factories

**Test Data:**
No fixtures or factories configured. Models directly load CSV:
```python
df = pd.read_csv("kickstarter_data_with_features.csv")
```

**Location:**
- CSV data: `kickstarter_data_with_features.csv` (root directory)
- Test models: `kickstarterModel_testing.py` (simplified variant with fewer steps)
- Hyperparameter configs: Defined inline in `hyperparameter_search.py`

**Example from `hyperparameter_search.py`:**
```python
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [64, 128, 256]
hidden_sizes_options = [[64, 32, 16], [128, 64, 32], [256, 128, 64]]
dropouts_options = [[0.3, 0.2, 0.1], [0.5, 0.3, 0.1], [0.4, 0.3, 0.2]]

for lr, batch_size, hidden_sizes, dropouts in product(
    learning_rates, batch_sizes, hidden_sizes_options, dropouts_options
):
    # Train and evaluate model
```

## Coverage

**Requirements:** None enforced

**View Coverage:**
- No coverage tool configured
- Manual inspection of test files and model output

## Test Types

**Unit Tests:**
- **Scope:** Individual PyTorch layer tests implicit in model training
- **Approach:** Not formalized; layers tested as part of training loop
- **Current:** No isolated unit tests for utilities or API endpoints

**Integration Tests:**
- **Scope:** Full training pipeline tested via `kickstarterModel.py` and `kickstarterModel_testing.py`
- **Approach:** End-to-end data → preprocessing → training → evaluation
- **Validation:** Metrics printed (accuracy, precision, recall, F1, AUC)

**E2E Tests:**
- **Framework:** Not configured
- **Current status:** Manual testing of frontend pages (render, navigation)
- **Backend:** No API endpoint tests; endpoints not yet implemented beyond `/` and `/health`

## Common Patterns

**Async Testing:**
- Not applicable (no async code in training loops)
- Frontend uses synchronous axios calls but no formal async testing

**Error Testing:**
- Models print metrics on convergence
- Early stopping halts training if validation loss plateaus (PATIENCE=25 epochs)
- No explicit error assertions; models fail silently if data is malformed

**Example early stopping pattern from `kickstarterModel.py`:**
```python
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # ... training and validation ...
    
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        epochs_without_improvement = 0
        best_model_state = model.state_dict().copy()
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f'\nEarly stopping at epoch {epoch+1}...')
        break

model.load_state_dict(best_model_state)
```

**Metric evaluation pattern from `kickstarterModel_testing.py` (lines 220-230):**
```python
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
test_acc = (y_pred == y_test).mean()

print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss:     {test_loss:.4f}")
print(f"Precision:     {precision_score(y_test, y_pred):.4f}")
print(f"Recall:        {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
```

**Visualization pattern for validation:**
```python
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(values_format='d', cmap='Blues', ax=ax)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob.flatten())
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.show()
```

---

*Testing analysis: 2026-04-14*
