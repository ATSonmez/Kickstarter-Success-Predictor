# Research Summary

**Project:** Kickstarter Success Predictor
**Synthesized:** 2026-04-14
**Sources:** `STACK.md`, `FEATURES.md`, `ARCHITECTURE.md`, `PITFALLS.md`

---

## Stack (what's locked, what to add)

The React/Vite + FastAPI + PyTorch + Postgres + Docker stack is locked and — critically — **already installed**. Audit of `backend/requirements.txt` and `frontend/package.json` shows every major library the project needs is present: `torch 2.7.1`, `shap 0.47.2`, `xgboost 3.0.2`, `scikit-learn 1.7.0`, `joblib`, `chart.js 4.5.1` + `react-chartjs-2`, `tailwindcss 4.2.2`, `axios`, `react-router-dom`. **The only new dependency is `react-hook-form`.**

Key library decisions:
- **Model loading:** FastAPI `lifespan` context manager, artifacts into `app.state` — never per-request.
- **Artifact format:** `torch.save(state_dict)` + importable architecture class; `joblib.dump` for sklearn/XGBoost.
- **SHAP:** `DeepExplainer` (NN) / `TreeExplainer` (XGBoost) / `LinearExplainer` (LogReg) — all built at startup with a 50–200 row background sample. Never `KernelExplainer`.
- **Charting:** Keep Chart.js — do NOT add Recharts or Plotly.
- **Styling:** Tailwind 4 only — do NOT add MUI.
- **Forms:** `react-hook-form` (uncontrolled inputs, no per-keystroke re-renders).
- **State:** Component-local `useState` — no Redux/Zustand.

---

## Table Stakes (v1 MVP)

| Feature | Notes |
|---|---|
| Prediction form | 6–8 fields with plain-English labels, `react-hook-form` validation |
| Probability output | Horizontal gradient bar + rounded whole percent + verdict badge (never gauge, never decimals) |
| Top-3 plain-English SHAP sentences | Server-side templated sentences from top features by `|shap_value|` |
| EDA dashboard | ≥3 charts: success by category, country, goal range |
| Prediction history | Postgres-backed, anonymous session ID in localStorage |
| `docker compose up` works on a fresh clone | The portability floor |
| Mobile-responsive layout | Tailwind responsive utilities from the start |

## Differentiators (portfolio polish)

| Feature | Notes |
|---|---|
| Plain-English SHAP sentence generation | The highest-leverage differentiator — every competitor omits XAI or shows only raw force plots |
| Model comparison page | Threshold slider + plain-English confusion matrix (not ROC-first) |
| Confidence-language tier labels | High / Moderate / Low — makes "67%" meaningful |
| Category context annotation | "Technology campaigns with goals over $50K succeed 28% of the time" next to the prediction |
| Collapsible SHAP bar chart | Behind a toggle, for curious users and portfolio reviewers |
| Per-history SHAP replay | Store `shap_values JSONB` column from day one to avoid later migration |

## Anti-Features (deliberately NOT building)

- Raw SHAP waterfall/force plots as primary output
- Semicircle/speedometer probability gauges
- All three model predictions shown simultaneously on the Predict page
- Decimal-precision probabilities (73.4%)
- Confidence intervals for casual users
- User accounts / login
- Kickstarter API scraping/autofill
- Real-time probability updates as user types
- "Similar campaigns" recommendation engine

---

## Architecture (at a glance)

**Shared preprocessing** (`backend/services/preprocessing.py`) is the single source of truth, used by BOTH training scripts and the inference path. Critical to avoid training/serving skew.

**Backend structure:**
```
backend/
├── main.py                      # FastAPI + lifespan loads artifacts into app.state
├── schemas.py                   # Pydantic v2 request/response
├── services/
│   ├── preprocessing.py         # CANONICAL; fit_transform() + transform_single()
│   ├── prediction_service.py    # Model forward pass + SHAP orchestration
│   └── eda_service.py           # Serves precomputed eda_stats.json
├── models/                      # gitignored artifacts
│   ├── kickstarter_nn.pt  scaler.pkl  feature_columns.pkl
│   ├── xgboost_model.pkl  logreg_model.pkl
│   ├── eda_stats.json     background.pt
├── database.py, db_models.py    # exists
```

**Build order (strict):**
1. `preprocessing.py`
2. Refactor training scripts → import it → save artifacts
3. FastAPI lifespan + `PredictionService`
4. `schemas.py` + `/predict` endpoint
5. `PredictPage.jsx` wired to `/predict` ← **first demoable milestone**
6. `/eda/stats`, `/models/metrics`, `/history` + their pages (parallelizable after step 2)
7. Docker Compose end-to-end validation
8. Resume polish (responsive styling, README with screenshots)

**EDA strategy:** Precompute aggregations into `eda_stats.json` during training; load once at startup and serve from memory. Never live-query Postgres — the dataset is static.

**History strategy:** Insert on every `/predict`. Query paginated `ORDER BY created_at DESC`. Anonymous — no user identity.

---

## Watch Out For — Top Pitfalls

1. **Scaler fitted on leaky columns (real bug in current `kickstarterModel.py`).** The scaler is fitted before leakage columns are dropped. Must be fixed in Phase 1.
2. **Three diverging preprocessing copies → guaranteed training/serving skew** unless consolidated before any new code is written. `kickstarterModel_testing.py` already diverges.
3. **Model artifacts not saved.** `torch.save` is never called. Inference is impossible. First deliverable.
4. **SHAP latency will break the 2–3s SLA** unless explainers are built at startup with a tiny background sample.
5. **`VITE_API_URL` is broken in Docker** — Vite embeds `VITE_*` vars at build time, not runtime. Fix with `ARG`/`ENV` in the frontend Dockerfile.
6. **`database.py` crashes on fresh clone** if `DATABASE_URL` is unset (`create_engine(None)` → cryptic error). One-line guard needed.
7. **PyTorch default install pulls CUDA wheels** → 3–5 GB Docker image. Force CPU-only wheel.
8. **Uncalibrated sigmoid displayed as "probability"** — show confidence tiers and category base rates, or label it a "success score".
9. **Model comparison invalid if preprocessing differs per model.** Single shared train/test split, `random_state=42`, identical feature set.
10. **Empty-scaffolding trap.** Predict first, end-to-end, before any other page is started. No placeholder pages in the router during demo.

---

## Roadmap Implications (guidance for `/gsd-plan-phase`)

- **Phase 1 — Foundation.** `backend/services/preprocessing.py` (with leakage audit + scaler fix), training script refactor, artifact save (`.pt`, `.pkl`, `.json`, `background.pt`), `eda_stats.json` generation, `.gitignore` updates for artifacts. **Nothing else is possible without this.**
- **Phase 2 — Core predict loop.** FastAPI lifespan + `PredictionService` + SHAP explainers at startup + Pydantic v2 schemas + `/predict` endpoint + `database.py` guard + `PredictPage.jsx` wired end-to-end. **First demoable milestone.**
- **Phase 3 — Supporting pages.** `/eda/stats` + DashboardPage, `/models/metrics` + PerformancePage (threshold slider, plain-English confusion matrix), `/history` + HistoryPage. Parallelizable.
- **Phase 4 — Integration + polish.** Multi-model training (XGBoost + LogReg) with shared split, model comparison wired, calibration plot, confidence tiers, responsive styling, Docker fixes (CPU-only torch, `ARG VITE_API_URL`, postgres healthcheck), clean-clone `docker compose up` smoke test, README with screenshots.

---

## Open Questions for Requirements/Planning

- `static_usd_rate` — drop from inference input (default to 1.0) or derive from currency?
- Class imbalance ratio in the CSV — determines calibration urgency.
- SHAP background sample size — start at 50 rows, benchmark, scale up.
- `background.pt` gitignored alongside other artifacts? Assumed yes.
- Prediction table needs `shap_values JSONB` column from day one to support history replay without migration.
