---
phase: 01-foundation
verified: 2026-04-18T23:14:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification: false
gaps: []
deferred: []
---

# Phase 1: Foundation Verification Report

**Phase Goal:** A single shared preprocessing module exists, the scaler leakage bug is fixed, and training produces all required artifacts so the rest of the stack can be wired up.
**Verified:** 2026-04-18T23:14:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running the training script produces all 6 artifacts in `backend/models/` | VERIFIED | All 6 files exist on disk: `kickstarter_nn.pt`, `scaler.pkl`, `feature_columns.pkl`, `eda_stats.json`, `background.pt`, `model_metadata.json`. `model_metadata.json` shows `accuracy: 0.7679`, `num_features: 70`. `eda_stats.json` contains `by_category`, `by_country`, `by_goal_bucket` keys. |
| 2 | The scaler is fitted only after post-campaign leakage columns are dropped | VERIFIED | In `preprocessing.py`, `POST_CAMPAIGN_FEATURES + AMBIGUOUS_DROP` drop occurs at byte offset 3799; `self.scaler.fit_transform` occurs at byte offset 4155. Correct order confirmed. `CONTINUOUS_COLS` has exactly 5 entries. |
| 3 | All three training scripts import from `backend/services/preprocessing.py` — no duplicated preprocessing remains | VERIFIED | `kickstarterModel.py` line 6: `from services.preprocessing import KickstarterPreprocessor`. `kickstarterModel_testing.py` line 8: same. `hyperparameter_search.py` line 9: same. No `StandardScaler` found in any of the three scripts. |
| 4 | `*.pt`, `*.pkl`, and `backend/models/*.json` are gitignored; only `.gitkeep` is tracked under `backend/models/` | VERIFIED | `.gitignore` line 35: `backend/models/*.json`. `git check-ignore` exits 0 for `eda_stats.json` and `model_metadata.json`. `*.pkl` and `*.pt` covered globally. `git ls-files backend/models/.gitkeep` returns the file as tracked. `git status backend/models/` shows clean working tree. |
| 5 | `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES` are explicitly documented in the preprocessing module | VERIFIED | `preprocessing.py` exports `LAUNCH_TIME_FEATURES` (9 items), `POST_CAMPAIGN_FEATURES` (4 items: `backers_count`, `pledged`, `usd_pledged`, `spotlight`), `AMBIGUOUS_DROP` (`static_usd_rate`). All annotated with explanatory comments. `static_usd_rate` is not in `LAUNCH_TIME_FEATURES`. |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `backend/services/preprocessing.py` | KickstarterPreprocessor class + feature constants | VERIFIED | 152 lines. Contains `class KickstarterPreprocessor`, `LAUNCH_TIME_FEATURES`, `POST_CAMPAIGN_FEATURES`, `AMBIGUOUS_DROP`, `CONTINUOUS_COLS`. All 4 methods implemented: `fit_transform`, `transform_single`, `save`, `load`. |
| `backend/services/__init__.py` | Python package marker | VERIFIED | Exists (1 byte). |
| `backend/models/nn_model.py` | KickstarterNet nn.Module | VERIFIED | 29 lines. Contains `class KickstarterNet(nn.Module)` with 4-layer architecture (`Linear 64/32/16/1`, BatchNorm, ReLU, Dropout 0.3/0.2/0.1). Instantiable: 3,553 params confirmed. |
| `backend/models/__init__.py` | Python package marker | VERIFIED | Exists. |
| `backend/models/.gitkeep` | Preserve empty dir in git | VERIFIED | Exists and tracked: `git ls-files` confirms. |
| `.gitignore` | `backend/models/*.json` rule | VERIFIED | Line 35 contains `backend/models/*.json`. |
| `tests/__init__.py` | Python package marker | VERIFIED | Exists (1 byte). |
| `tests/conftest.py` | Shared fixtures | VERIFIED | Contains `def sample_raw_df` (line 10) and `def tmp_models_dir` (line 82). |
| `tests/test_preprocessing.py` | FND-01/02/03/04 test stubs | VERIFIED | 6 test functions present: `test_preprocessor_importable`, `test_feature_constants_exist`, `test_scaler_excludes_leakage`, `test_transform_single_zero_fills`, `test_save_creates_artifacts`, `test_load_roundtrip`. |
| `tests/test_training.py` | FND-04/05/06 test stubs | VERIFIED | 4 test functions present: `test_all_six_artifacts_exist_after_training` (xfail), `test_no_standardscaler_import_in_root_scripts`, `test_json_artifacts_gitignored`, `test_gitkeep_exists`. |
| `pyproject.toml` | `[tool.pytest.ini_options]` with testpaths=tests | VERIFIED | Contains `testpaths = ["tests"]`, `pythonpath = ["."]`, `addopts = "-ra"`. |
| `backend/requirements-dev.txt` | pytest pin | VERIFIED | Contains `-r requirements.txt` and `pytest==8.3.3`. |
| `backend/models/kickstarter_nn.pt` | Trained neural net state_dict | VERIFIED | File exists on disk; gitignored. |
| `backend/models/scaler.pkl` | Fitted StandardScaler | VERIFIED | File exists on disk; gitignored. |
| `backend/models/feature_columns.pkl` | Post-encoding column list | VERIFIED | File exists on disk; gitignored. |
| `backend/models/eda_stats.json` | Precomputed dashboard aggregations | VERIFIED | File exists, 38 lines. Keys: `by_category`, `by_country`, `by_goal_bucket`. Gitignored. |
| `backend/models/background.pt` | SHAP background tensor | VERIFIED | File exists on disk; gitignored. |
| `backend/models/model_metadata.json` | Training metadata | VERIFIED | Keys: `trained_at`, `accuracy` (0.7679), `auc`, `num_features` (70), `feature_columns`. Gitignored. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `backend/services/preprocessing.py` | `sklearn.preprocessing.StandardScaler` | `self.scaler = StandardScaler()` fitted after drop | WIRED | Drop at offset 3799, `fit_transform` at 4155 — correct order confirmed |
| `tests/test_preprocessing.py` | `backend/services/preprocessing.py` | `from backend.services.preprocessing import KickstarterPreprocessor` | WIRED | Line 10 in test file; module importable from project root via `pythonpath = ["."]` |
| `kickstarterModel.py` | `backend/services/preprocessing.py` | `sys.path.insert` + `from services.preprocessing import KickstarterPreprocessor` | WIRED | Line 5: `sys.path.insert(0, ...)`, Line 6: import. Preprocessor used at line 70. |
| `kickstarterModel.py` | `backend/models/nn_model.py` | `from models.nn_model import KickstarterNet` | WIRED | Line 7 confirmed. No local `class KickstarterNet` definition remains in the file. |
| `kickstarterModel.py` | `backend/models/` (6 artifact files) | `torch.save` / `preprocessor.save` / `json.dump` | WIRED | Lines 251, 254, 267, 270, 271, 281, 282 all confirmed. EDA stats block (`df_clean = df.copy()`) at line 40, before preprocessor at line 70 — ordering invariant holds. |
| `kickstarterModel_testing.py` | `backend/services/preprocessing.py` | `sys.path.insert` + import | WIRED | Lines 7–8 confirmed. |
| `hyperparameter_search.py` | `backend/services/preprocessing.py` | `sys.path.insert` + import | WIRED | Lines 8–9 confirmed. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|-------------------|--------|
| `eda_stats.json` | `eda_stats` dict | `df_clean.groupby(...)` aggregations in `kickstarterModel.py` | Yes — 38 lines with real category/country/goal-bucket records | FLOWING |
| `model_metadata.json` | `metadata` dict | `test_acc`, `roc_auc`, `X_train.shape[1]`, `preprocessor.feature_columns` from training run | Yes — accuracy 0.7679, num_features 70 confirmed real values | FLOWING |
| `scaler.pkl` | `self.scaler` | `StandardScaler().fit_transform(df[present_continuous])` on 5 columns only | Yes — fitted from real training data, leakage-free | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `KickstarterPreprocessor` importable from project root | `python3 -c "from backend.services.preprocessing import KickstarterPreprocessor, CONTINUOUS_COLS; print(len(CONTINUOUS_COLS))"` | `5` | PASS |
| `CONTINUOUS_COLS` has exactly 5 entries | Same as above | `5` | PASS |
| `static_usd_rate` in `AMBIGUOUS_DROP`, not in `LAUNCH_TIME_FEATURES` | Python import check | `True` / `False` confirmed | PASS |
| `KickstarterNet` importable and instantiable | `python3 -c "sys.path.insert(0,'backend'); from models.nn_model import KickstarterNet; KickstarterNet(10)"` | `params: 3553` | PASS |
| All 6 model artifacts exist | `test -f` for each of 6 files | All 6 confirmed present | PASS |
| JSON artifacts gitignored | `git check-ignore -v backend/models/eda_stats.json` | Exit 0, matched `.gitignore:35` | PASS |
| `.gitkeep` tracked in git | `git ls-files backend/models/.gitkeep` | `backend/models/.gitkeep` | PASS |
| No `StandardScaler` in any of 3 training scripts | `grep StandardScaler kickstarterModel.py kickstarterModel_testing.py hyperparameter_search.py` | No matches in any file | PASS |
| EDA stats ordering invariant | `df_clean = df.copy()` byte offset < `preprocessor = KickstarterPreprocessor()` offset | 40 < 70 (lines) confirmed | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FND-01 | 01-02-PLAN.md | Single `backend/services/preprocessing.py` as canonical source for both training and inference | SATISFIED | `preprocessing.py` exists, importable, contains full `KickstarterPreprocessor` class with `fit_transform`, `transform_single`, `save`, `load` |
| FND-02 | 01-02-PLAN.md | `StandardScaler` fitted only on non-leaky columns (scaler ordering bug fixed) | SATISFIED | Drop of `POST_CAMPAIGN_FEATURES + AMBIGUOUS_DROP` confirmed at offset before `scaler.fit_transform`; `CONTINUOUS_COLS` = 5 |
| FND-03 | 01-02-PLAN.md | `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES` explicitly documented; `static_usd_rate` audited | SATISFIED | All three constants exported with explanatory comments; `static_usd_rate` in `AMBIGUOUS_DROP`, absent from `LAUNCH_TIME_FEATURES` |
| FND-04 | 01-03-PLAN.md | Training produces all 6 artifacts in `backend/models/` | SATISFIED | All 6 files confirmed on disk with real content from a completed training run (76.8% accuracy, 70 features, early stop epoch 43) |
| FND-05 | 01-03-PLAN.md | Three training scripts refactored to import shared preprocessor; no duplicated code | SATISFIED | All 3 scripts: `sys.path.insert` + `from services.preprocessing import KickstarterPreprocessor`. Zero `StandardScaler` occurrences found. |
| FND-06 | 01-02-PLAN.md | `*.pt`, `*.pkl`, `backend/models/*.json` gitignored; only `.gitkeep` tracked | SATISFIED | `.gitignore` rules confirmed; `git check-ignore` exits 0 for JSON artifacts; `.gitkeep` tracked; `git status backend/models/` is clean |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No TODOs, stubs, empty returns, or placeholder patterns found in phase deliverables | — | — |

Notes:
- `hyperparameter_search.py` retains a local parametric `KickstarterNet` variant with `hidden_sizes`/`dropouts` kwargs. This is intentional per 01-RESEARCH.md Open Question #2 and the 01-03-PLAN.md design decision — the parametric variant is incompatible with the canonical fixed-architecture class. Not a stub; not a gap.
- `test_all_six_artifacts_exist_after_training` is marked `@pytest.mark.xfail` but per 01-03-SUMMARY.md resolved as XPASS (all 6 artifacts exist). The xfail marker is now stale but harmless — it causes pytest to report XPASS, which is correct behavior.

---

### Human Verification Required

None. All must-haves are verifiable programmatically from the codebase state. The human checkpoint in Plan 03 (Task 4) was already completed and signed off (`checkpoint: approved` in 01-03-SUMMARY.md).

---

### Gaps Summary

No gaps. All 5 ROADMAP success criteria are fully satisfied. All 6 FND requirements (FND-01 through FND-06) are implemented, wired, and backed by real artifacts from a completed training run. The pytest infrastructure (Wave 0) is correctly configured and all 10 test functions are present. The leakage bug fix is verified at the source code level (ordering invariant) and at the artifact level (scaler fitted on 5 columns only). The gitignore rules are correct and verified via `git check-ignore`.

---

_Verified: 2026-04-18T23:14:00Z_
_Verifier: Claude (gsd-verifier)_
