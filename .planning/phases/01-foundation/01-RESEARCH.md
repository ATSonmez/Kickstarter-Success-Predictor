# Phase 1: Foundation - Research

**Researched:** 2026-04-18
**Domain:** Python ML preprocessing refactor — scikit-learn / PyTorch / joblib artifact persistence
**Confidence:** HIGH (all findings grounded in direct codebase inspection and locked prior research)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — Refactor depth:** Minimal refactor only — replace inline preprocessing code with the shared import. Matplotlib/seaborn EDA blocks, print statements, and exploratory plots stay in place. Scripts remain usable as standalone notebooks for future exploratory runs.
- **D-02 — File locations:** Training scripts stay at project root. They import backend modules via `sys.path.insert(0, str(Path(__file__).parent / "backend"))`.
- **D-03 — nn.Module location:** The PyTorch architecture class lives in `backend/models/nn_model.py` — importable by both root training scripts and the FastAPI lifespan loader.

### Claude's Discretion

- **`static_usd_rate` handling:** Drop it from the feature set. Users cannot supply a currency exchange rate at form time. Default to dropping; if the model needs it, hardcode 1.0 in `transform_single()` and document clearly.
- **EDA goal buckets:** 4 ranges — `<$1,000`, `$1,000–$10,000`, `$10,000–$100,000`, `>$100,000`. Used consistently in `eda_stats.json` and Phase 3 dashboard.
- **Background sample size:** 100 rows (balanced: 50 success + 50 failure) saved as `background.pt` for SHAP `DeepExplainer`.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FND-01 | Single `backend/services/preprocessing.py` module, canonical source for all feature transforms, imported by training AND inference | `KickstarterPreprocessor` class design with `fit_transform` / `transform_single` / `save` / `load` methods fully specified in ARCHITECTURE.md |
| FND-02 | Scaler fitted only on non-leaky columns (scaler ordering bug fixed) | Bug confirmed at `kickstarterModel.py` line 71 — scaler called before leakage drop at line 85 |
| FND-03 | Explicit `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES` documented; `static_usd_rate` audited | Decision locked: drop `static_usd_rate`; document both lists as module-level constants |
| FND-04 | Training saves all 6 artifacts to `backend/models/`: `kickstarter_nn.pt`, `scaler.pkl`, `feature_columns.pkl`, `eda_stats.json`, `background.pt`, `model_metadata.json` | `torch.save`, `joblib.dump`, `json.dump` patterns identified; `torch.save` never called in current code |
| FND-05 | Three training scripts refactored to import shared preprocessing — no duplicated code | Verbatim duplication confirmed across all three files; exact import pattern specified |
| FND-06 | `*.pt`, `*.pkl`, `backend/models/*.json` gitignored; only `.gitkeep` tracked under `backend/models/` | `.gitignore` audited — `*.pt` and `*.pkl` are covered; `backend/models/*.json` is NOT yet covered |
</phase_requirements>

---

## Summary

Phase 1 is a pure Python refactor with no new library dependencies. All three training scripts (`kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`) contain verbatim copies of ~60 lines of preprocessing code. A `KickstarterPreprocessor` class in `backend/services/preprocessing.py` replaces all three copies. The class has a strict interface: `fit_transform(df)` for training (fits and saves the scaler), `transform_single(raw_dict)` for inference (loads the saved scaler and zero-fills against the saved column list), and `save(dir)` / `load(dir)` for artifact I/O.

The scaler leakage bug is concrete and located: `kickstarterModel.py` line 71 calls `scaler.fit_transform(df[continuous_cols])` where `continuous_cols` includes `backers_count`, `pledged`, and `usd_pledged` — but those leakage columns are not dropped until line 85. The fix is to reorder: drop leakage columns first, then fit the scaler on the remaining continuous columns (`goal`, `static_usd_rate`-or-dropped, `name_len`, `blurb_len`, `create_to_launch`, `launch_to_deadline`).

The gitignore has a gap: `*.pt` and `*.pkl` are covered, but `backend/models/*.json` is not. The `backend/models/` directory itself also needs a `.gitkeep` (both `backend/models/` and `backend/services/` are currently empty with no tracked files). The `nn.Module` class must move to `backend/models/nn_model.py` so the FastAPI lifespan can import it without depending on the root training scripts.

**Primary recommendation:** Build `KickstarterPreprocessor` first; plug it into `kickstarterModel.py` second; verify end-to-end artifact output before touching the other two scripts.

---

## Standard Stack

### Core

| Library | Version (pinned) | Purpose | Source |
|---------|-----------------|---------|--------|
| scikit-learn | 1.7.0 | `StandardScaler` fit/transform, `train_test_split` | `[VERIFIED: backend/requirements.txt]` |
| torch | 2.7.1 | `nn.Module`, `torch.save` / `torch.load`, `TensorDataset` | `[VERIFIED: backend/requirements.txt]` |
| joblib | 1.5.1 | Serialize `StandardScaler` and `feature_columns.pkl` via `joblib.dump` / `joblib.load` | `[VERIFIED: backend/requirements.txt]` |
| pandas | 3.0.2 | `pd.get_dummies`, `pd.to_timedelta`, `dropna` | `[VERIFIED: backend/requirements.txt]` |
| numpy | 2.4.4 | `np.float32` casts, array operations | `[VERIFIED: backend/requirements.txt]` |
| shap | 0.47.2 | `DeepExplainer` background sample saved as `background.pt` | `[VERIFIED: backend/requirements.txt]` |

> **Note on system Python vs. project Python:** The machine's system Python 3.14/3.13 has torch 2.11 and sklearn 1.8, but the project's `backend/requirements.txt` pins torch 2.7.1 and sklearn 1.7.0. Training scripts should be run inside a venv that matches `requirements.txt`, not the system Python. `shap` is not installed on the system Python — must use the project venv. [VERIFIED: direct probing]

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | — | Write `eda_stats.json` and `model_metadata.json` | Artifact saving in training script |
| pathlib (stdlib) | — | `Path(__file__).parent` for portable paths | All file I/O in the module |
| pickle (via joblib) | — | Underlying format for `.pkl` artifacts | Used transparently by joblib |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `joblib.dump` for scaler | `pickle.dump` directly | joblib is already a dependency and handles numpy arrays more efficiently — use joblib |
| `torch.save(state_dict)` | `torch.save(whole_model)` | State dict is smaller and architecture-independent — always use state dict |
| Module-level `LAUNCH_TIME_FEATURES` constant | Inline comments only | Constant is machine-readable; Phase 2 can validate inputs against it |

**Installation:** No new dependencies. All required libraries are already in `backend/requirements.txt`. [VERIFIED: backend/requirements.txt]

---

## Architecture Patterns

### Recommended Project Structure (Phase 1 additions)

```
backend/
├── models/
│   ├── .gitkeep              # NEW: track empty dir; all *.pt, *.pkl, *.json gitignored
│   └── nn_model.py           # NEW: KickstarterNet class (extracted from kickstarterModel.py)
├── services/
│   └── preprocessing.py      # NEW: KickstarterPreprocessor class
kickstarterModel.py            # MODIFIED: import KickstarterPreprocessor, add artifact saves
kickstarterModel_testing.py    # MODIFIED: import KickstarterPreprocessor (minimal)
hyperparameter_search.py       # MODIFIED: import KickstarterPreprocessor (minimal)
.gitignore                     # MODIFIED: add backend/models/*.json pattern
```

### Pattern 1: KickstarterPreprocessor Class Interface

**What:** A stateful class that owns all feature transform logic. `fit_transform` is the training path (fits scaler, captures column list). `transform_single` is the inference path (uses saved state, zero-fills missing columns).

**When to use:** Any code that touches feature columns — training or inference — goes through this class.

```python
# Source: .planning/research/ARCHITECTURE.md Pattern 2
# backend/services/preprocessing.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# ── Feature category documentation ───────────────────────────────────────────
# Columns the user CAN know before launching a campaign
LAUNCH_TIME_FEATURES = [
    'goal', 'name_len', 'blurb_len', 'create_to_launch', 'launch_to_deadline',
    'country', 'category', 'deadline_yr', 'launched_at_yr',
]

# Columns that are only knowable AFTER a campaign ends — never used at inference
POST_CAMPAIGN_FEATURES = [
    'backers_count', 'pledged', 'usd_pledged', 'spotlight',
]

# Dropped entirely — post-campaign exchange rate, not knowable at launch time
# (Decision D-discretion: drop static_usd_rate; not supplied at form time)
AMBIGUOUS_DROP = ['static_usd_rate']

METADATA_COLS = [
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

CONTINUOUS_COLS = [
    'goal', 'name_len', 'blurb_len', 'create_to_launch', 'launch_to_deadline'
]


class KickstarterPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Training path. Fits the scaler AFTER leakage columns are dropped.
        Returns (X, y) as float32 numpy arrays.
        """
        df = df.copy()
        df.drop(columns=METADATA_COLS, errors='ignore', inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Target
        df['succeeded'] = (df['state'] == 'successful').astype(int)
        df.drop(columns='state', inplace=True)

        # Timedelta → days
        df['create_to_launch'] = pd.to_timedelta(df['create_to_launch']).dt.days
        df['launch_to_deadline'] = pd.to_timedelta(df['launch_to_deadline']).dt.days

        # One-hot encode
        df = pd.get_dummies(df, columns=['country', 'category', 'deadline_yr', 'launched_at_yr'])
        bool_cols = df.select_dtypes(include='bool').columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Drop post-campaign leakage AND ambiguous columns BEFORE fitting scaler
        leakage_and_ambiguous = POST_CAMPAIGN_FEATURES + AMBIGUOUS_DROP
        df.drop(columns=[c for c in leakage_and_ambiguous if c in df.columns], inplace=True)

        # Fit scaler only on clean continuous columns
        present_continuous = [c for c in CONTINUOUS_COLS if c in df.columns]
        df[present_continuous] = self.scaler.fit_transform(df[present_continuous])

        # Capture post-encoding column list (excludes target)
        self.feature_columns = [c for c in df.columns if c != 'succeeded']

        y = df['succeeded'].values.astype(np.float32)
        X = df[self.feature_columns].values.astype(np.float32)
        return X, y

    def transform_single(self, raw: dict) -> np.ndarray:
        """
        Inference path. Zero-fills columns not present in this row,
        scales continuous fields via the FITTED scaler.
        Never calls pd.get_dummies() on a one-row DataFrame.
        """
        row = {col: 0 for col in self.feature_columns}

        # Continuous fields — scale
        cont_vals = np.array([[
            raw.get('goal', 0),
            raw.get('name_len', 0),
            raw.get('blurb_len', 0),
            raw.get('create_to_launch', raw.get('prep_days', 0)),
            raw.get('launch_to_deadline', raw.get('duration_days', 0)),
        ]], dtype=np.float32)
        scaled = self.scaler.transform(cont_vals)[0]
        for i, col in enumerate(CONTINUOUS_COLS):
            if col in row:
                row[col] = float(scaled[i])

        # One-hot fields
        if raw.get('country'):
            col = f"country_{raw['country']}"
            if col in row:
                row[col] = 1
        if raw.get('category'):
            col = f"category_{raw['category']}"
            if col in row:
                row[col] = 1

        return np.array([list(row.values())], dtype=np.float32)

    def save(self, models_dir: Path) -> None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, models_dir / "scaler.pkl")
        joblib.dump(self.feature_columns, models_dir / "feature_columns.pkl")

    @classmethod
    def load(cls, models_dir: Path) -> "KickstarterPreprocessor":
        models_dir = Path(models_dir)
        preprocessor = cls()
        preprocessor.scaler = joblib.load(models_dir / "scaler.pkl")
        preprocessor.feature_columns = joblib.load(models_dir / "feature_columns.pkl")
        return preprocessor
```

### Pattern 2: Training Script Adapter (minimal refactor, D-01)

**What:** Replace the inline preprocessing block with a three-line import + call. EDA blocks and plot code stay in place.

**When to use:** In all three training scripts at their section 1–3 blocks.

```python
# At top of kickstarterModel.py (replace sklearn StandardScaler import for scaler usage)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from services.preprocessing import KickstarterPreprocessor

# Replace sections 1-3 (load/clean/engineer/encode/scale/leak-drop) with:
df = pd.read_csv("kickstarter_data_with_features.csv")
preprocessor = KickstarterPreprocessor()
X, y = preprocessor.fit_transform(df)
# X and y are float32 numpy arrays — proceed directly to train/test split
```

### Pattern 3: nn.Module Extraction (D-03)

**What:** Move `KickstarterNet` class definition from `kickstarterModel.py` to `backend/models/nn_model.py`. Both training scripts and the Phase 2 lifespan loader import from that single location.

```python
# backend/models/nn_model.py
import torch.nn as nn

class KickstarterNet(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_features, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),          nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)
```

### Pattern 4: Artifact Saving Block

**What:** After training completes in `kickstarterModel.py`, save all 6 required artifacts.

```python
import json, torch, joblib
from pathlib import Path

MODELS_DIR = Path("backend/models")

# 1. Neural net weights
torch.save(model.state_dict(), MODELS_DIR / "kickstarter_nn.pt")

# 2. Scaler + feature columns (already written by preprocessor.save)
preprocessor.save(MODELS_DIR)

# 3. SHAP background sample — 100 rows balanced (50 success + 50 failure)
success_idx = np.where(y_train == 1)[0]
failure_idx = np.where(y_train == 0)[0]
bg_idx = np.concatenate([
    np.random.choice(success_idx, 50, replace=False),
    np.random.choice(failure_idx, 50, replace=False),
])
background_tensor = torch.tensor(X_train[bg_idx])
torch.save(background_tensor, MODELS_DIR / "background.pt")

# 4. EDA stats — goal buckets per locked discretion decision
goal_bins   = [0, 1_000, 10_000, 100_000, float('inf')]
goal_labels = ["<$1k", "$1k-$10k", "$10k-$100k", ">$100k"]
# (computed from original df before fit_transform; see pitfall note below)

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
```

### Pattern 5: EDA Stats Computation

**What:** `eda_stats.json` must be computed from the original pre-encoding DataFrame — not from the one-hot-encoded matrix.

```python
# Computed BEFORE calling preprocessor.fit_transform, from the raw df
# (raw df must be cleaned but not yet encoded)

# Example: success rate by category
cat_stats = (
    df_clean.groupby('category')['succeeded']
    .agg(success_rate='mean', count='count')
    .reset_index()
    .to_dict(orient='records')
)

# Goal bucket success rates (locked discretion decision)
goal_bins   = [0, 1_000, 10_000, 100_000, float('inf')]
goal_labels = ["<$1k", "$1k-$10k", "$10k-$100k", ">$100k"]
df_clean['goal_bucket'] = pd.cut(df_clean['goal'], bins=goal_bins, labels=goal_labels, right=False)
goal_stats = (
    df_clean.groupby('goal_bucket', observed=True)['succeeded']
    .agg(success_rate='mean', count='count')
    .reset_index()
    .to_dict(orient='records')
)

eda_stats = {
    "by_category": cat_stats,
    "by_country": country_stats,
    "by_goal_bucket": goal_stats,
}
with open(MODELS_DIR / "eda_stats.json", "w") as f:
    json.dump(eda_stats, f, indent=2)
```

### Anti-Patterns to Avoid

- **Fitting scaler before leakage drop:** The bug being fixed. Order MUST be: drop metadata → drop leakage → one-hot encode → fit scaler → save. [VERIFIED: kickstarterModel.py lines 65-86]
- **Calling `pd.get_dummies()` on a one-row inference input:** Produces only columns present in that row. Use zero-fill from saved `feature_columns` list instead.
- **Computing EDA stats AFTER `fit_transform`:** `fit_transform` destroys the original column structure. Compute EDA stats from the cleaned-but-not-encoded df, before calling `fit_transform`.
- **Putting `KickstarterNet` import in training scripts only:** Phase 2 needs to import the class to reconstruct the model for inference. It must live in `backend/models/nn_model.py`.
- **Saving `feature_columns` as JSON:** Phase 2 uses `joblib.load` to get a Python list back. Save as `.pkl` for round-trip fidelity (joblib handles it; JSON would require re-ordering guarantees).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scaler serialization | Custom pickle wrapper | `joblib.dump` / `joblib.load` | joblib is already a dep; handles numpy internals correctly |
| Feature column alignment at inference | Manual dict-building logic | Zero-fill dict keyed on saved `feature_columns` list | Exact pattern already specified in ARCHITECTURE.md; any deviation risks shape mismatch |
| Model artifact directory management | Custom dir-creation logic | `Path.mkdir(parents=True, exist_ok=True)` | One line, idiomatic Python |
| EDA aggregations at serve time | Live pandas query in FastAPI | Precomputed JSON written at train time | FastAPI never loads the CSV; zero latency |

**Key insight:** This phase has no novel algorithmic challenges. Every pattern is standard sklearn/PyTorch serialization. The value is in the correct ordering (leakage fix) and the interface contract (`fit_transform` vs `transform_single`) that Phase 2 depends on.

---

## Common Pitfalls

### Pitfall 1: Scaler Fitted on Leaky Columns (Real Bug)

**What goes wrong:** `StandardScaler` computes mean/variance over `backers_count`, `pledged`, `usd_pledged` — post-campaign values the model should never see. The scaler's learned statistics are polluted by leakage. When inference is later wired up, those columns are absent — the scaler operates on a differently-shaped input, producing silently wrong scaled values.

**Why it happens:** In `kickstarterModel.py`, `fit_transform` is called at line 71 on `continuous_cols` which includes the three leaky columns. The leakage drop happens at line 85, after scaling.

**How to avoid:** `fit_transform` in `KickstarterPreprocessor` must drop all leakage and `static_usd_rate` columns from the DataFrame *before* instantiating or fitting the scaler.

**Warning signs:** If `scaler.mean_` has 9 values (matching original `continuous_cols`), the bug is still present. The fixed scaler should have 5 values (matching `CONTINUOUS_COLS` = goal, name_len, blurb_len, create_to_launch, launch_to_deadline). [VERIFIED: kickstarterModel.py direct inspection]

### Pitfall 2: EDA Stats Computed From Encoded DataFrame

**What goes wrong:** If EDA stats are computed after `fit_transform`, the column `goal` has been replaced by scaled values and `category` columns have been one-hot-encoded. `groupby('category')` fails. Goal bucket ranges applied to standardized floats produce meaningless groupings.

**Why it happens:** Doing everything in sequence — load, clean, fit_transform, then compute stats.

**How to avoid:** Compute all EDA aggregations from `df_clean` (after `dropna` and target creation, but before calling `KickstarterPreprocessor.fit_transform`). Pass `df_clean` separately to an EDA stats function.

### Pitfall 3: `backend/models/*.json` Not Gitignored

**What goes wrong:** `eda_stats.json` and `model_metadata.json` get committed to git. These files contain aggregated training data statistics and could be large or sensitive. More practically, they will conflict on every retrain.

**Why it happens:** The current `.gitignore` covers `*.pkl` and `*.pt` globally, but not `*.json` files inside `backend/models/`. [VERIFIED: .gitignore direct inspection]

**How to avoid:** Add `backend/models/*.json` to `.gitignore`. Also confirm `backend/models/` is otherwise empty except `.gitkeep`. [VERIFIED: backend/models/ is empty, no .gitkeep exists yet]

### Pitfall 4: Missing `__init__.py` in `backend/services/`

**What goes wrong:** `from services.preprocessing import KickstarterPreprocessor` raises `ModuleNotFoundError` because Python does not recognise `services/` as a package without `__init__.py`.

**Why it happens:** `backend/services/` is a freshly created empty directory with no `__init__.py`.

**How to avoid:** Create `backend/services/__init__.py` (empty file is sufficient). [VERIFIED: backend/services/ is empty, no __init__.py exists]

### Pitfall 5: `nn_model.py` in `backend/models/` Without `__init__.py`

**What goes wrong:** Same as Pitfall 4 — `from models.nn_model import KickstarterNet` fails.

**How to avoid:** Create `backend/models/__init__.py` (empty). Note the directory is `backend/models/`, not `backend/model/`.

### Pitfall 6: `torch.load` Without `weights_only` Argument (PyTorch 2.x Warning)

**What goes wrong:** In PyTorch >= 2.0, `torch.load(path)` emits a `FutureWarning` and in later versions requires explicit `weights_only=True` or `weights_only=False`. For `background.pt` (a plain tensor), use `weights_only=True`.

**How to avoid:**
```python
background = torch.load(MODELS_DIR / "background.pt", weights_only=True)
model.load_state_dict(torch.load(MODELS_DIR / "kickstarter_nn.pt", weights_only=True))
```
[ASSUMED — PyTorch 2.7.1 behavior; consistent with PyTorch 2.x migration docs pattern]

---

## Gitignore Gap — Action Required

The current `.gitignore` covers:
- `*.pkl` — global ✓
- `*.pt` — global ✓
- `*.csv` — global ✓

**Missing:**
- `backend/models/*.json` — `eda_stats.json` and `model_metadata.json` are NOT covered

**Required addition to `.gitignore`:**
```
# Model artifacts — JSON outputs
backend/models/*.json
```

[VERIFIED: .gitignore direct inspection, 2026-04-18]

---

## Code Examples

### Correct Scaler Ordering (Fixed)

```python
# Source: direct inspection of kickstarterModel.py + PITFALLS.md §Pitfall 1
# WRONG (current code):
continuous_cols = ['backers_count', 'goal', 'pledged', ...]  # leaky cols included
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])   # line 71
# ... leakage drop happens AFTER at line 85

# CORRECT (in KickstarterPreprocessor.fit_transform):
df.drop(columns=POST_CAMPAIGN_FEATURES + AMBIGUOUS_DROP, errors='ignore', inplace=True)
# NOW fit the scaler — only clean columns remain
df[CONTINUOUS_COLS] = self.scaler.fit_transform(df[CONTINUOUS_COLS])
```

### sys.path Import Pattern

```python
# Source: CONTEXT.md D-02 / ARCHITECTURE.md Integration Points
# At top of kickstarterModel.py, kickstarterModel_testing.py, hyperparameter_search.py:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from services.preprocessing import KickstarterPreprocessor
```

### Balanced Background Sample

```python
# Source: CONTEXT.md Claude's Discretion — 100 rows, 50/50 split
rng = np.random.default_rng(42)
success_idx = np.where(y_train == 1)[0]
failure_idx = np.where(y_train == 0)[0]
bg_idx = np.concatenate([
    rng.choice(success_idx, 50, replace=False),
    rng.choice(failure_idx, 50, replace=False),
])
background_tensor = torch.tensor(X_train[bg_idx], dtype=torch.float32)
torch.save(background_tensor, MODELS_DIR / "background.pt")
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `scaler.fit_transform` before leakage drop | Drop leakage first, then `fit_transform` | Removes data leakage from scaler statistics |
| Inline preprocessing duplicated in 3 files | Single `KickstarterPreprocessor` class | Single change point for any future feature changes |
| No `torch.save` calls in training scripts | Save 6 artifacts on every training run | Enables Phase 2 inference |
| `KickstarterNet` defined only in root scripts | `backend/models/nn_model.py` | Importable by FastAPI lifespan without root dependency |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | PyTorch 2.7.1 requires `weights_only` argument to `torch.load` to suppress warnings | Common Pitfalls §6 | Low risk — if wrong, code still works, just emits a warning |
| A2 | `backend/models/__init__.py` and `backend/services/__init__.py` are required for Python package imports | Anti-Patterns | Medium — if wrong (namespace packages), imports may still work without them; but adding empty `__init__.py` is harmless |

---

## Open Questions

1. **EDA stats DataFrame scope** — `eda_stats.json` requires access to the original cleaned df (before encoding). The cleanest architecture has `fit_transform` return `(X, y, df_clean)` or the training script computes EDA stats separately before calling `fit_transform`. The latter (compute EDA stats inline in the training script from `df_clean`, pass to a separate function) keeps `KickstarterPreprocessor` single-responsibility.
   - What we know: EDA stats need `goal` (unscaled float) and `category` / `country` (string columns before one-hot).
   - Recommendation: Compute EDA stats in the training script from a snapshot of `df_clean` taken after `dropna` / target creation, but before `fit_transform`. Pass the snapshot dict to a `compute_eda_stats(df_clean)` helper.

2. **`hyperparameter_search.py` refactor scope** — This script has a different `KickstarterNet` signature (takes `hidden_sizes` and `dropouts` parameters, not a fixed architecture). After refactoring preprocessing, it still trains many model variants and does not save a final model. The minimal refactor (D-01) means only the preprocessing block changes; the parametric model class can remain local to that script.
   - Recommendation: Do not extract `hyperparameter_search.py`'s `KickstarterNet` variant to `nn_model.py` — it is a different signature used only for search.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | backend/requirements.txt target | Partial — system has 3.14/3.13; project venv needed | System: 3.14.3 | Use project venv matching requirements.txt |
| torch 2.7.1 | Training, artifact save | Partial — system has 2.11 (newer); venv likely has 2.7.1 | System: 2.11.0+cpu | Use project venv |
| scikit-learn 1.7.0 | StandardScaler | Partial — system has 1.8.0 | System: 1.8.0 | Use project venv |
| joblib 1.5.1 | Artifact serialization | Partial — system has 1.5.3 | System: 1.5.3 | Compatible; venv preferred |
| shap 0.47.2 | background.pt creation | NOT available on system Python | — | Must use project venv |
| pandas 3.0.2 | DataFrame operations | Available (system matches) | 3.0.2 | — |

**Missing dependencies with no fallback:**
- `shap` — not installed on system Python. Background sample creation (`background.pt`) requires `shap` only indirectly (the tensor itself is created without shap; shap is only needed at inference/Phase 2). Phase 1 can create `background.pt` as a plain PyTorch tensor without importing shap. The tensor is merely saved for use by `DeepExplainer` in Phase 2.

**Note:** Training scripts must be run with the project venv (`pip install -r backend/requirements.txt`), not the system Python. [VERIFIED: direct probing]

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed — Wave 0 gap) |
| Config file | None — needs `pytest.ini` or `pyproject.toml` [pytest] section |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

> No test files, no pytest config, and no `tests/` directory exist in the project. [VERIFIED: filesystem scan]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FND-01 | `preprocessing.py` exists and is importable | smoke | `python -c "from backend.services.preprocessing import KickstarterPreprocessor"` | Wave 0 |
| FND-02 | Scaler `mean_` has exactly 5 values (not 9) after fit | unit | `pytest tests/test_preprocessing.py::test_scaler_excludes_leakage -x` | Wave 0 |
| FND-03 | `LAUNCH_TIME_FEATURES` and `POST_CAMPAIGN_FEATURES` exist as module constants | unit | `pytest tests/test_preprocessing.py::test_feature_constants -x` | Wave 0 |
| FND-04 | All 6 artifact files exist in `backend/models/` after training | integration | `pytest tests/test_artifacts.py::test_all_artifacts_saved -x` | Wave 0 |
| FND-05 | No `StandardScaler()` import in the three root training scripts | static | `grep -r "StandardScaler" kickstarterModel.py kickstarterModel_testing.py hyperparameter_search.py` (expect 0 matches) | manual |
| FND-06 | `backend/models/test.json` is gitignored | smoke | `git check-ignore -v backend/models/test.json` | manual |

### Sampling Rate

- **Per task commit:** `pytest tests/test_preprocessing.py -x -q` (unit tests for the new module)
- **Per wave merge:** `pytest tests/ -v` (full suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/__init__.py` — make tests a package
- [ ] `tests/test_preprocessing.py` — covers FND-01, FND-02, FND-03
- [ ] `tests/test_artifacts.py` — covers FND-04 (runs training, checks file existence)
- [ ] `pytest` install: add to `backend/requirements.txt` or a separate `requirements-dev.txt`
- [ ] `pytest.ini` or `[tool.pytest.ini_options]` in `pyproject.toml` — set `testpaths = ["tests"]`

---

## Security Domain

This phase has no API endpoints, no user input, no authentication surface, and no network calls. It is offline Python script execution producing local files.

Applicable ASVS checks for this phase:

| ASVS Category | Applies | Note |
|---------------|---------|------|
| V5 Input Validation | No | No user input in Phase 1 |
| V6 Cryptography | No | No secrets handled |
| V2–V4 Auth/Session/Access | No | No HTTP surface |

**One security-adjacent action:** `backend/models/*.json` must be gitignored (FND-06) to prevent training statistics from leaking into the public repo. This is addressed in the gitignore gap section above.

---

## Sources

### Primary (HIGH confidence)

- Direct inspection: `kickstarterModel.py` — scaler bug confirmed at lines 65-86
- Direct inspection: `kickstarterModel_testing.py` — identical preprocessing block confirmed
- Direct inspection: `hyperparameter_search.py` lines 1-66 — identical preprocessing block confirmed
- Direct inspection: `.gitignore` — `backend/models/*.json` gap confirmed
- Direct inspection: `backend/models/` and `backend/services/` — both empty, no `__init__.py`, no `.gitkeep`
- `.planning/research/ARCHITECTURE.md` — `KickstarterPreprocessor` class interface, build order, anti-patterns
- `.planning/research/PITFALLS.md` — Pitfalls 1, 2, 3, 7 directly applicable to Phase 1
- `.planning/codebase/CONCERNS.md` — Priority debt items confirmed
- `.planning/codebase/STACK.md` — Pinned library versions
- `backend/requirements.txt` — Dependency versions verified

### Secondary (MEDIUM confidence)

- `.planning/phases/01-foundation/01-CONTEXT.md` — Locked decisions D-01, D-02, D-03, discretion decisions

### Tertiary (LOW confidence — training knowledge)

- PyTorch 2.x `weights_only` deprecation pattern [ASSUMED — not verified against 2.7.1 changelog]

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all versions verified against requirements.txt and filesystem
- Architecture: HIGH — `KickstarterPreprocessor` interface specified in prior research, locked in CONTEXT.md
- Pitfalls: HIGH — all critical pitfalls grounded in direct code inspection
- Gitignore gap: HIGH — verified by reading .gitignore directly

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (stable domain; no fast-moving external dependencies for this phase)
