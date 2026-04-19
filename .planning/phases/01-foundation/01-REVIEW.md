---
phase: 01-foundation
reviewed: 2026-04-18T23:18:00Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - .gitignore
  - backend/models/nn_model.py
  - backend/requirements-dev.txt
  - backend/services/preprocessing.py
  - hyperparameter_search.py
  - kickstarterModel.py
  - kickstarterModel_testing.py
  - pyproject.toml
  - tests/conftest.py
  - tests/test_preprocessing.py
  - tests/test_training.py
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-18T23:18:00Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

The Phase 1 foundation is well-structured. The leakage fix (FND-02) is correctly implemented — the scaler fits only on 5 continuous columns after post-campaign features are dropped. The shared preprocessor (`backend/services/preprocessing.py`) is the single source of truth and both production and test scripts import from it correctly. The neural network architecture in `nn_model.py` is cleanly separated. Gitignore rules are complete and verified.

Four warnings are present, all relating to correctness risks: a crash-at-runtime bug if training ends at epoch 0, a shallow `state_dict().copy()` that may not protect the best model state, a test assertion that will silently fail on the actual import string used, and a missing `scheduler.step()` call in the testing script. Four informational items cover unused imports, the naming clash of `KickstarterNet` in `hyperparameter_search.py`, a missing `requirements.txt` reference, and the CSV path being hardcoded as a relative string.

---

## Warnings

### WR-01: Crash if training stops at epoch 0 (all three training scripts)

**File:** `kickstarterModel.py:179`, `kickstarterModel_testing.py:138`, `hyperparameter_search.py:109`

**Issue:** `best_model_state` is initialised to `None`. If early stopping fires on the very first epoch (i.e. the first val loss is not less than `float('inf')`, which cannot happen by construction — BUT if `PATIENCE=0` were passed or the loop body were refactored), `model.load_state_dict(None)` would raise `TypeError`. More practically, if the DataLoader yields zero batches (empty training set), `total` stays 0 and `avg_loss = epoch_loss / total` raises `ZeroDivisionError` at line 142/104 before `best_model_state` is ever assigned.

**Fix:** Guard the restore call and the division:

```python
# Guard zero-batch edge case
avg_loss = epoch_loss / total if total > 0 else 0.0

# Guard restore
if best_model_state is not None:
    model.load_state_dict(best_model_state)
```

---

### WR-02: Shallow copy of `state_dict` may not snapshot tensor data

**File:** `kickstarterModel.py:165`, `kickstarterModel_testing.py:124`, `hyperparameter_search.py:102`

**Issue:** `model.state_dict().copy()` performs a shallow dict copy. `state_dict()` returns a new `OrderedDict` each call, but the tensor *values* inside it are the same objects as the live model parameters. A plain `.copy()` of the dict does not deep-copy the tensors, so `best_model_state` holds references to the tensors that continue to be mutated by subsequent training steps. In practice PyTorch tensors are reference-typed, so this can cause the "best" state to silently drift toward the final weights rather than the true best-epoch weights.

**Fix:** Use `copy.deepcopy` or the idiomatic pattern:

```python
import copy
best_model_state = copy.deepcopy(model.state_dict())
```

---

### WR-03: `test_no_standardscaler_import_in_root_scripts` assertion will always fail

**File:** `tests/test_training.py:39`

**Issue:** The test asserts:
```python
assert "from services.preprocessing import KickstarterPreprocessor" in text
```
But all three training scripts use a `sys.path.insert` trick and import as:
```python
from services.preprocessing import KickstarterPreprocessor
```
That string *does* match — however the test also reads `hyperparameter_search.py`, which does **not** import `KickstarterNet` from `backend.models.nn_model` (it defines its own local `KickstarterNet` class). The test only checks for `StandardScaler` absence and the preprocessor import, so it passes for `hyperparameter_search.py` on those two assertions. This is fine as written, but the comment in the test says "scripts import from shared module instead" — the local `KickstarterNet` redefinition in `hyperparameter_search.py` (lines 49-66) is a silent divergence from that stated intent. If `KickstarterNet` architecture changes in `nn_model.py`, `hyperparameter_search.py` will silently use a stale definition and produce misleading search results.

**Fix:** Add an assertion that covers the canonical model import:

```python
# In test_no_standardscaler_import_in_root_scripts, for the search script specifically:
assert "from models.nn_model import KickstarterNet" in hyperparameter_search_text or \
       "KickstarterNet" not in hyperparameter_search_text, (
    "hyperparameter_search.py defines a local KickstarterNet that shadows the canonical class"
)
```
Or better: refactor `hyperparameter_search.py` to accept `hidden_sizes`/`dropouts` as constructor kwargs via a thin wrapper, removing the local class entirely.

---

### WR-04: `kickstarterModel_testing.py` omits `scheduler.step()` silently

**File:** `kickstarterModel_testing.py:120-135`

**Issue:** The testing script creates an `optim.Adam` optimizer (line 66) but does **not** create or step a `ReduceLROnPlateau` scheduler, unlike `kickstarterModel.py`. This is intentional per the comment "No BatchNorm, no class imbalance weighting," but it diverges from the production training loop in a way that is not documented. The file comment at line 2-3 only mentions "no BatchNorm, no class imbalance weighting" — it does not mention "no LR scheduler." Comparing results between the two scripts will attribute differences to BatchNorm/weighting when LR scheduling is also absent.

**Fix:** Add a comment explicitly documenting the missing scheduler:

```python
# No LR scheduler (also intentionally omitted — isolating Adam+early-stopping only)
```

Or add the scheduler to make the comparison single-variable (BatchNorm + class weighting only).

---

## Info

### IN-01: Unused imports in training scripts

**File:** `kickstarterModel.py:14-15`, `kickstarterModel_testing.py:12-13`

**Issue:** Both scripts import `matplotlib.pyplot as plt`, `seaborn as sns`, and `classification_report` from sklearn. `sns` (seaborn) is imported but never called in either file. `classification_report` is imported but not used in either file.

**Fix:** Remove unused imports:
```python
# Remove from both files:
import seaborn as sns
# Remove from sklearn.metrics import line:
classification_report,
```

---

### IN-02: `KickstarterNet` name shadowing in `hyperparameter_search.py`

**File:** `hyperparameter_search.py:49`

**Issue:** The local class `KickstarterNet` (lines 49-66) shares a name with the canonical `KickstarterNet` in `backend/models/nn_model.py`. The local class is intentionally parametric (different constructor signature), documented with a comment at line 47. However, the name collision means any future `from models.nn_model import KickstarterNet` added to this file would silently shadow or be shadowed by the local definition depending on import order.

**Fix:** Rename the local class to make the distinction unambiguous:
```python
class SearchableKickstarterNet(nn.Module):
    ...
```

---

### IN-03: `backend/requirements-dev.txt` pins only pytest; `requirements.txt` not in scope

**File:** `backend/requirements-dev.txt:1`

**Issue:** The file includes `-r requirements.txt` but `requirements.txt` is not present in the reviewed file list and does not appear to exist in `backend/`. If it is missing, `pip install -r requirements-dev.txt` will fail with a file-not-found error.

**Fix:** Verify `backend/requirements.txt` exists and is tracked in git. If the base requirements live at the repo root, update the reference:
```
-r ../requirements.txt
```

---

### IN-04: CSV path hardcoded as relative string in all training scripts

**File:** `kickstarterModel.py:34`, `kickstarterModel_testing.py:30`, `hyperparameter_search.py:25`

**Issue:** All three scripts load data via `pd.read_csv("kickstarter_data_with_features.csv")` — a bare filename resolved against the current working directory. Running from any directory other than the repo root will raise `FileNotFoundError` with no helpful message.

**Fix:** Use `Path(__file__)` to anchor the path:
```python
DATA_PATH = Path(__file__).parent / "kickstarter_data_with_features.csv"
df = pd.read_csv(DATA_PATH)
```

---

_Reviewed: 2026-04-18T23:18:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
