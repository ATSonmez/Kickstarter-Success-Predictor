# Roadmap: Kickstarter Success Predictor

## Overview

This roadmap wires up an existing brownfield scaffold into a working, demo-ready prediction product. The build order is strictly dependency-ordered: shared preprocessing must exist before training can save artifacts, artifacts must exist before the API can serve predictions, and the predict loop must work end-to-end before any supporting pages are worth building. Docker polish and the quality gate close out the milestone.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Foundation** — Build shared preprocessing, fix scaler bug, save all model artifacts
- [ ] **Phase 2: Core Predict Loop** — Wire FastAPI lifespan, prediction service, SHAP, and PredictPage end-to-end
- [ ] **Phase 3: Supporting Pages** — EDA dashboard and prediction history (backend + frontend)
- [ ] **Phase 4: Integration and Polish** — Docker hardening, dev proxy, responsive styling, README

## Phase Details

### Phase 1: Foundation
**Goal**: A single shared preprocessing module exists, the scaler leakage bug is fixed, and training produces all required artifacts so the rest of the stack can be wired up
**Depends on**: Nothing (first phase)
**Requirements**: FND-01, FND-02, FND-03, FND-04, FND-05, FND-06
**Success Criteria** (what must be TRUE):
  1. Running the training script produces `kickstarter_nn.pt`, `scaler.pkl`, `feature_columns.pkl`, `eda_stats.json`, `background.pt`, and `model_metadata.json` in `backend/models/`
  2. The scaler is fitted only after post-campaign leakage columns are dropped (no leaky columns in the fitted scaler)
  3. All three training scripts (`kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`) import from `backend/services/preprocessing.py` — no duplicated preprocessing code remains in those files
  4. `*.pt`, `*.pkl`, and `backend/models/*.json` are gitignored; only `.gitkeep` (or nothing) is tracked under `backend/models/`
  5. `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES` are explicitly documented in the preprocessing module
**Plans**: TBD

### Phase 2: Core Predict Loop
**Goal**: Users can open the Predict page, fill in a campaign form, submit it, and see a success probability with plain-English SHAP explanations — the first demoable milestone
**Depends on**: Phase 1
**Requirements**: PRD-01, PRD-02, PRD-03, PRD-04, PRD-05, PRD-06, PRD-07, PRD-08, PRD-09, PRD-10, PRD-11, PRD-12, INT-05, INT-06
**Success Criteria** (what must be TRUE):
  1. A form with plain-English labels (category, country, goal, name length, blurb length, duration, prep days) renders on the Predict page — no ML jargon in any label
  2. Submitting the form returns a result in under 3 seconds showing: a horizontal gradient bar with a whole-percent probability, a verdict badge ("Likely to succeed" / "At risk"), a confidence-tier label, the category base rate, and 3 plain-English SHAP sentences
  3. A loading spinner or skeleton is visible while the request is in flight
  4. Model artifacts and SHAP explainer are loaded once at startup — restarting the server and sending one request does not trigger a second load
  5. Submitting an invalid payload (e.g., missing `goal`) returns a 422 with a human-readable error
  6. `backend/database.py` raises a clear `RuntimeError` at startup if `DATABASE_URL` is unset
**Plans**: TBD
**UI hint**: yes

### Phase 3: Supporting Pages
**Goal**: Users can explore historical success patterns on the EDA Dashboard and review past predictions on the History page, with all data sourced from the live backend
**Depends on**: Phase 1
**Requirements**: EDA-01, EDA-02, EDA-03, EDA-04, HST-01, HST-02, HST-03, HST-04, HST-05
**Success Criteria** (what must be TRUE):
  1. The Dashboard page renders three charts — success rate by category, by country, and by funding goal range — loaded from `GET /eda/stats` (no CSV read, no live DB query at request time)
  2. After submitting a prediction, the History page shows that prediction in the table with timestamp, category, country, goal, probability, and verdict badge
  3. Clicking a row in the history table expands it to show the plain-English SHAP explanation replayed from the stored `shap_values` JSONB column
  4. The `predictions` table has a `shap_values JSONB` column from the initial schema — no ALTER TABLE migration is needed
**Plans**: TBD
**UI hint**: yes

### Phase 4: Integration and Polish
**Goal**: `docker compose up` from a fresh clone works end-to-end, the UI is mobile-responsive and styled, a baseline test suite passes, and the README is demo-ready
**Depends on**: Phase 3
**Requirements**: INT-01, INT-02, INT-03, INT-04, INT-07, QLT-01, QLT-02, QLT-03, QLT-04
**Success Criteria** (what must be TRUE):
  1. On a fresh clone, `cp .env.example .env && docker compose up` brings backend, frontend, and Postgres online with a working Predict page — no manual setup beyond the `.env` copy
  2. The built backend Docker image is under 1.5 GB (PyTorch CPU-only wheel used)
  3. All pages render real data from real endpoints — no "coming soon" placeholder text remains
  4. The app renders correctly at phone (≤640px), tablet, and desktop widths
  5. `pytest` passes a suite covering: at least one preprocessing unit test, one `/predict` endpoint test, and one `/eda/stats` endpoint test
  6. `README.md` contains project description, one-command setup, at least two screenshots, and notes on the ML approach and SHAP explanations
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/TBD | Not started | - |
| 2. Core Predict Loop | 0/TBD | Not started | - |
| 3. Supporting Pages | 0/TBD | Not started | - |
| 4. Integration and Polish | 0/TBD | Not started | - |
