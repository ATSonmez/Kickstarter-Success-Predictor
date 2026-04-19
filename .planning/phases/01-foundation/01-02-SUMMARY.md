---
plan: 01-02
phase: 01-foundation
status: complete
completed: 2026-04-18
commits:
  - fee8bc7
  - 2444af2
  - 48f914d
---

## What was built

Shared preprocessing module with leakage bug fixed, KickstarterNet extracted to canonical location, gitignore gap closed for JSON artifacts.

## Key files created

- `backend/services/__init__.py` — package marker
- `backend/services/preprocessing.py` — KickstarterPreprocessor class + LAUNCH_TIME_FEATURES / POST_CAMPAIGN_FEATURES / AMBIGUOUS_DROP / CONTINUOUS_COLS constants. Scaler fitted AFTER leakage drop (FND-02 fix).
- `backend/models/__init__.py` — package marker
- `backend/models/nn_model.py` — KickstarterNet(nn.Module), 4-layer architecture matching original kickstarterModel.py
- `backend/models/.gitkeep` — tracked so empty dir survives clone
- `.gitignore` — appended `backend/models/*.json` rule (FND-06)

## Verification

- `python -m pytest tests/test_preprocessing.py -q` → 6 passed
- `python -m pytest tests/test_training.py::test_json_artifacts_gitignored tests/test_training.py::test_gitkeep_exists -q` → 2 passed
- `test_no_standardscaler_import_in_root_scripts` → RED (expected — Plan 03 closes this)
- `test_all_six_artifacts_exist_after_training` → XFAIL (expected — requires full training run in Plan 03)
- `python -c "from backend.services.preprocessing import KickstarterPreprocessor, CONTINUOUS_COLS; print(len(CONTINUOUS_COLS))"` → 5

## Self-Check: PASSED
