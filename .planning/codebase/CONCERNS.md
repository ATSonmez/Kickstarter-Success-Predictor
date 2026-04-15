# Concerns

Technical debt, known issues, security gaps, performance risks, and fragile areas in the Kickstarter Success Predictor codebase.

## Tech Debt

- **Duplicated preprocessing logic** across `kickstarterModel.py`, `kickstarterModel_testing.py`, and `hyperparameter_search.py` (roughly lines 1–85 of each). Any schema/feature change must be made in three places.
- **Unused dependencies** declared in `backend/requirements.txt`: `xgboost` and `shap` are listed but never imported anywhere in `backend/`.
- **Empty scaffolding directories**: `backend/services/` and `backend/models/` exist but contain no files. Phase 3 deliverables are missing.
- **No model persistence**: training in `kickstarterModel.py` produces a PyTorch model in-memory only — no `torch.save(...)`, no `scaler.pkl`, no `feature_columns.pkl`.
- **Frontend pages are stubs**: `frontend/src/pages/PredictPage.jsx`, `DashboardPage.jsx`, `PerformancePage.jsx`, `HistoryPage.jsx` are placeholder components with no real logic.
- **`frontend/src/components/` is empty** — no reusable UI components extracted yet.

## Known Bugs / Latent Issues

- **Missing `DATABASE_URL` validation** in `backend/database.py` — no guard if env var is absent, will fail at connection time with a cryptic error.
- **No input validation** for the future `/predict` endpoint — no Pydantic request model in `backend/main.py`.
- **Feature scaler not saved** in `kickstarterModel.py` around the fit/transform step — inference would use an unscaled or refit scaler, producing wrong predictions.
- **Hardcoded column drops** repeated in three training files must stay in sync manually — easy to break silently.

## Security

- **CORS `allow_headers=["*"]`** in `backend/main.py` — overly permissive; should be an explicit allowlist before deployment.
- **Env vars loaded without validation** — `os.getenv("FRONTEND_URL", "http://localhost:5173")` silently falls back to dev default in production.
- **No API authentication or rate limiting** planned for `/predict` — acceptable for MVP behind a personal URL, risky if exposed publicly.
- **Hardcoded dataset path** in training scripts — not a direct vulnerability but leaks filesystem structure.

## Performance

- **Full DataFrame loaded into memory** in `kickstarterModel.py` (~line 25) — fine for current dataset, will break on larger data.
- **No inference batching** planned — one request = one model forward pass.
- **CPU-only** training and inference; no `.to(device)` logic for GPU acceleration.
- **Model artifacts will be loaded per-request** unless startup caching is explicitly implemented in FastAPI lifespan hooks.

## Fragile Areas

- **Feature count assumptions** are implicit: if the category/country one-hot space changes, the saved model's input layer won't match incoming requests, and there's no guard.
- **Training scripts assume CWD** contains `kickstarter_data_with_features.csv` — breaks when invoked from `backend/`.
- **Docker Compose references postgres** but backend has no actual DB usage yet — wiring is half-done and will confuse new contributors.
- **Three separate training entrypoints** (`kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`) with no shared module — any refactor risks divergence.

## Missing Critical Features

- **No `/predict` endpoint** — the core product is not yet wired up.
- **No `/models`, `/predict/{model_name}`, `/eda/stats`** endpoints.
- **No SHAP integration** despite being listed in requirements.
- **No model comparison** (XGBoost, LogReg) — only the neural net exists.
- **No frontend ↔ backend wiring** beyond `frontend/src/api.js` scaffold.

## Test Coverage Gaps (all currently zero)

- No unit tests for preprocessing functions.
- No integration tests for the training pipeline.
- No API endpoint tests (pytest + httpx not yet set up).
- No frontend component tests (React Testing Library not installed).
- No end-to-end tests covering form → API → prediction → render.

## Priority Recommendations

1. **P0 — Unblock core flow**: save model/scaler/feature columns, build `/predict` endpoint, wire `PredictPage.jsx`.
2. **P0 — Consolidate preprocessing** into a single `backend/services/preprocessing.py` module imported by both training and inference.
3. **P1 — Decide database scope**: either use the postgres scaffold or remove it from `docker-compose.yml` and `backend/database.py` to reduce confusion.
4. **P1 — Tighten CORS and validate env vars** before any public deployment.
5. **P2 — Add pytest baseline** with at least one endpoint test and one preprocessing test to establish the testing foundation.
