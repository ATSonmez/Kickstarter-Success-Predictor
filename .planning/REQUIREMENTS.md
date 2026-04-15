# Requirements — Kickstarter Success Predictor v1

**Project:** Kickstarter Success Predictor
**Defined:** 2026-04-14
**Scope:** v1 MVP — working local product demo-ready for a resume + portfolio

Every requirement below is a capability the user (or the system, at the system boundary) must exhibit. Each maps to exactly one phase in `ROADMAP.md`.

---

## v1 Requirements

### Foundation — model persistence and shared preprocessing

- [ ] **FND-01** — A single `backend/services/preprocessing.py` module exists and is the canonical source of truth for all feature transforms, imported by BOTH the training scripts and the FastAPI inference path.
- [ ] **FND-02** — The preprocessing module fits its `StandardScaler` only on non-leaky columns (scaler ordering bug in the current `kickstarterModel.py` is fixed).
- [ ] **FND-03** — The preprocessing module explicitly documents `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES`, and `static_usd_rate` is audited and either dropped or documented as launch-time-knowable.
- [ ] **FND-04** — Training produces and saves these artifacts to `backend/models/`: `kickstarter_nn.pt` (state_dict), `scaler.pkl`, `feature_columns.pkl`, `eda_stats.json`, `background.pt` (50–100 row SHAP background sample), and `model_metadata.json` (training date, accuracy, AUC, feature count). v1 trains only the neural net; XGBoost and Logistic Regression are deferred to v2. v1 trains only the neural net; XGBoost and Logistic Regression are deferred to v2.
- [ ] **FND-05** — The three existing training scripts (`kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`) are refactored to import the shared preprocessing module — no duplicated preprocessing code remains.
- [ ] **FND-06** — `*.pt`, `*.pkl`, `backend/models/*.json` are gitignored; only `.gitkeep` (or nothing) is tracked.

### Prediction — the core loop

- [ ] **PRD-01** — User can open the Predict page and see a form with plain-English labels for: campaign category, country, funding goal (USD), name length, blurb length, duration in days, prep days. No ML jargon in the labels.
- [ ] **PRD-02** — Form submission sends a `POST /predict` request validated by a Pydantic v2 `PredictRequest` model; invalid input returns a 422 with a human-readable error message.
- [ ] **PRD-03** — `POST /predict` returns a success probability (rounded to a whole percent), a boolean verdict, the model name, and a top-8 list of SHAP feature contributions with signed values.
- [ ] **PRD-04** — Model artifacts and SHAP explainers are loaded exactly once at FastAPI startup via a `lifespan` context manager; nothing is loaded per-request.
- [ ] **PRD-05** — `/predict` end-to-end response time is under 3 seconds in local single-request testing.
- [ ] **PRD-06** — The Predict page displays the probability as a horizontal gradient bar (red → green) with the rounded whole-percent number above it. No gauge, no decimals.
- [ ] **PRD-07** — The Predict page displays a verdict badge ("Likely to succeed" / "At risk") above the bar.
- [ ] **PRD-08** — The Predict page displays the top 3 SHAP contributions as plain-English sentences (magnitude bucket + direction), generated server-side.
- [ ] **PRD-09** — The Predict page displays a confidence-tier label ("High confidence" / "Moderate" / "Low") derived from the probability.
- [ ] **PRD-10** — The Predict page displays the historical base rate for the selected category next to the prediction, sourced from the same precomputed EDA stats used by the Dashboard.
- [ ] **PRD-11** — The Predict page shows a loading state while the request is in flight.
- [ ] **PRD-12** — Only the neural net is trained and served in v1; no model selector on the Predict page.

### EDA Dashboard

- [ ] **EDA-01** — `GET /eda/stats` returns precomputed aggregations loaded from `backend/models/eda_stats.json` at startup; no live Postgres aggregation, no CSV loaded at request time.
- [ ] **EDA-02** — The Dashboard page renders a horizontal bar chart of success rate by category.
- [ ] **EDA-03** — The Dashboard page renders a horizontal bar chart of success rate by country.
- [ ] **EDA-04** — The Dashboard page renders a bar/histogram chart of success rate by funding goal range.

### Prediction History

- [ ] **HST-01** — Every call to `POST /predict` persists a row to the `predictions` Postgres table including: model name, input fields, probability, verdict, SHAP values (`JSONB` column), and `created_at` timestamp.
- [ ] **HST-02** — The Prediction table schema includes a `shap_values JSONB` column from the first migration — no later ALTER.
- [ ] **HST-03** — `GET /history?limit=50&offset=0` returns paginated history ordered by `created_at DESC`, without requiring authentication.
- [ ] **HST-04** — The History page renders a table showing timestamp, category, country, goal, probability, and verdict badge for each past prediction.
- [ ] **HST-05** — Clicking a history row expands it to show the original SHAP explanation (replayed from the stored `shap_values`).

### Integration and portability

- [ ] **INT-01** — `docker compose up` from a freshly cloned repo with only a provided `.env.example` copied to `.env` brings the entire stack online (Postgres + backend + frontend) and reaches a working Predict page in the browser.
- [ ] **INT-02** — The backend Dockerfile installs PyTorch with the CPU-only wheel (`--index-url https://download.pytorch.org/whl/cpu`); final backend image is under 1.5 GB.
- [ ] **INT-03** — The frontend Dockerfile declares `ARG VITE_API_URL` and `ENV VITE_API_URL=$VITE_API_URL`; `docker-compose.yml` passes it via `build.args`, not runtime env.
- [ ] **INT-04** — `docker-compose.yml` adds a `healthcheck` to the `db` service; backend `depends_on.db.condition` is `service_healthy`.
- [ ] **INT-05** — `backend/database.py` raises a clear `RuntimeError` at startup if `DATABASE_URL` is unset (no cryptic `create_engine(None)` error).
- [ ] **INT-06** — `backend/main.py` CORS is tightened from `allow_headers=["*"]` to an explicit allowlist appropriate for this app.
- [ ] **INT-07** — `frontend/vite.config.js` declares a dev proxy from `/api` to the backend so local (non-Docker) development does not hit CORS.

### Quality and polish

- [ ] **QLT-01** — The app renders correctly and usably at phone (≤640px), tablet, and desktop widths using Tailwind responsive utilities.
- [ ] **QLT-02** — A baseline pytest suite covers at least: one preprocessing unit test (feature count stability), one `/predict` endpoint test (via httpx), and one `/eda/stats` endpoint test.
- [ ] **QLT-03** — The `README.md` includes: project description, architecture diagram (or link to `.planning/research/ARCHITECTURE.md`), one-command setup (`docker compose up`), at least two screenshots of the running product, and notes about the ML approach and SHAP explanations.
- [ ] **QLT-04** — All frontend pages render real data from real endpoints before the demo — no placeholder "coming soon" stubs remain in the routing.

---

## v2 / Deferred (Nice-to-Have, Not Blocking MVP)

- [ ] **V2-01** — Collapsible full SHAP bar chart behind a "See full breakdown" toggle on the Predict page.
- [ ] **V2-02** — Temperature-scaling calibration if a reliability diagram deviates from the diagonal.
- [ ] **V2-03** — `slowapi` rate limiting on `/predict`.
- [ ] **V2-04** — CI pipeline (GitHub Actions) running pytest on push.
- [ ] **V2-05** — End-to-end frontend tests with React Testing Library.
- [ ] **V2-06** — Train XGBoost and Logistic Regression on the same shared split as the neural net; persist metrics.
- [ ] **V2-07** — Model comparison / Performance page: threshold slider, plain-English confusion matrix, ROC curves, AUC summary, calibration diagram.

---

## Out of Scope — Explicit Exclusions (with Reasoning)

- **Live public deployment** — deferred per PROJECT.md. The bar for v1 is a working local demo, not a hosted URL.
- **Automated data refresh / scraping** — the static CSV is the only data source; scraping violates Kickstarter ToS and is a scope explosion.
- **User accounts / authentication** — no auth for v1. History is anonymous; session-based grouping is itself out of scope.
- **Mobile-native apps** — responsive web only.
- **Real-time Kickstarter API integration** — users paste data manually.
- **Recommendation engine / "similar campaigns"** — weeks of scope for marginal value relative to the core loop.
- **Showing all three model predictions simultaneously on the Predict page** — destroys user trust; comparison lives on the Performance page, framed as portfolio context.
- **Decimal-precision probabilities** — spurious precision; always round to whole percent.
- **Confidence intervals shown to casual users** — encoded instead via confidence-tier labels.
- **Raw SHAP waterfall/force plots as the primary output** — unreadable to non-ML users; plain-English sentences first, bar chart behind a toggle.
- **Semicircle / speedometer probability gauge** — misread risk; horizontal gradient bar is the mental model.
- **Real-time probability updates as the user types** — single submit button only.

---

## Traceability

*(Filled in by the roadmapper — every REQ-ID above maps to exactly one phase in `ROADMAP.md`.)*

| REQ-ID | Phase |
|---|---|
| FND-01 | 1 |
| FND-02 | 1 |
| FND-03 | 1 |
| FND-04 | 1 |
| FND-05 | 1 |
| FND-06 | 1 |
| PRD-01 | 2 |
| PRD-02 | 2 |
| PRD-03 | 2 |
| PRD-04 | 2 |
| PRD-05 | 2 |
| PRD-06 | 2 |
| PRD-07 | 2 |
| PRD-08 | 2 |
| PRD-09 | 2 |
| PRD-10 | 2 |
| PRD-11 | 2 |
| PRD-12 | 2 |
| EDA-01 | 3 |
| EDA-02 | 3 |
| EDA-03 | 3 |
| EDA-04 | 3 |
| HST-01 | 3 |
| HST-02 | 3 |
| HST-03 | 3 |
| HST-04 | 3 |
| HST-05 | 3 |
| INT-01 | 4 |
| INT-02 | 4 |
| INT-03 | 4 |
| INT-04 | 4 |
| INT-05 | 2 |
| INT-06 | 2 |
| INT-07 | 4 |
| QLT-01 | 4 |
| QLT-02 | 4 |
| QLT-03 | 4 |
| QLT-04 | 4 |
