---
plan: 01-03
phase: 01-foundation
status: complete
completed: 2026-04-18
commits:
  - 4ce1ee1
  - 002d2b5
checkpoint: approved
---

## What was built

Three root training scripts refactored to use shared KickstarterPreprocessor (FND-05). Primary script extended with artifact-saving block producing all 6 required artifacts (FND-04). Full training run executed and human-verified.

## Key files modified

- `kickstarterModel.py` — removed inline preprocessing + KickstarterNet class; imports shared preprocessor and canonical nn_model; adds EDA stats snapshot before preprocessing; saves 6 artifacts after training
- `kickstarterModel_testing.py` — replaced manual preprocessing with KickstarterPreprocessor; imports canonical KickstarterNet
- `hyperparameter_search.py` — replaced manual preprocessing with KickstarterPreprocessor; kept local parametric KickstarterNet variant (incompatible signature, per research guidance)

## Artifacts produced (FND-04)

- `backend/models/kickstarter_nn.pt` — trained state_dict (best val loss epoch)
- `backend/models/scaler.pkl` — fitted StandardScaler (5 cols, leakage-free)
- `backend/models/feature_columns.pkl` — post-encoding column list (70 features)
- `backend/models/eda_stats.json` — by_category / by_country / by_goal_bucket aggregations
- `backend/models/background.pt` — 100-row balanced SHAP background tensor
- `backend/models/model_metadata.json` — trained_at, accuracy, auc, num_features

## Training results

- Dataset: 20,632 rows → 18,738 after cleaning; 70 features post-encoding
- Early stopping: epoch 43 / 500 (patience=25)
- Test accuracy: 76.8% | Precision: 0.577 | Recall: 0.680 | F1: 0.624
- Scaler mean_.shape: (5,) — leakage fix confirmed end-to-end

## Verification

- `python -m pytest tests/ -v` → 9 passed, 1 xpassed (FND-04 xfail promoted to xpass)
- `git status backend/models/` → clean (artifacts gitignored)
- `scaler.mean_.shape == (5,)` → confirmed
- Human checkpoint: approved

## Self-Check: PASSED
