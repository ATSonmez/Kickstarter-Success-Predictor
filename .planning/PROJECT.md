# Kickstarter Success Predictor

## What This Is

A web app where a casual Kickstarter backer (or a more serious investor) can paste in the details of a live campaign and see a data-driven prediction of whether it will succeed, along with an explanation of *why*. The goal is to reduce the gut-feel risk of pledging to campaigns by grounding the decision in historical patterns from thousands of past projects.

## Core Value

**Paste a campaign → get a trustworthy, explained success probability.** If the predict-and-explain loop doesn't feel credible to a user in under 30 seconds, nothing else in the product matters.

## Requirements

### Validated

<!-- Inferred from existing code in the brownfield codebase map. -->

- ✓ PyTorch neural network trained on `kickstarter_data_with_features.csv` with hyperparameter search — existing (`kickstarterModel.py`, `hyperparameter_search.py`)
- ✓ FastAPI backend skeleton with CORS configured for the React frontend — existing (`backend/main.py`)
- ✓ Vite + React frontend skeleton with routing and page stubs for Predict / Dashboard / Performance / History — existing (`frontend/src/`)
- ✓ Docker Compose scaffold for backend + frontend + Postgres — existing (`docker-compose.yml`)

### Active

<!-- Current scope. These are hypotheses until shipped. -->

- [ ] **Predict page** — form input for a campaign (category, country, goal, duration, etc.) → success probability + SHAP explanation of top contributing features
- [ ] **Model serving pipeline** — persisted PyTorch model, scaler, and feature columns loaded once at FastAPI startup and reused across requests
- [ ] **Consolidated preprocessing module** — single `backend/services/preprocessing.py` used by both training and inference, replacing the current three-way duplication
- [ ] **EDA dashboard page** — interactive charts of success rate by category, country, and goal range, sourced from a `/eda/stats` endpoint
- [ ] **Model comparison** — train and serve XGBoost and Logistic Regression alongside the neural net; expose a model selector and side-by-side metrics (ROC/AUC, confusion matrix, threshold slider)
- [ ] **Prediction history** — persist each prediction to Postgres so users can revisit past results (uses the existing docker-compose Postgres scaffold)
- [ ] **Resume-grade polish** — styled, responsive UI and a clean README with screenshots, good enough to demo in an interview and link from a resume
- [ ] **`docker compose up` works end-to-end on a fresh clone** — the minimum bar for "deployable"

### Out of Scope

- **Live public deployment (Vercel/Railway)** — deferred. Decide after the product feels demo-ready locally. The goal is a working MVP first, a URL second.
- **Automated data refresh / scraping pipeline** — deferred. Start with the static CSV already in the repo. Revisit only if the project reaches the polish stage and still has time.
- **User accounts / auth** — not needed. Prediction history is keyed anonymously (session or local ID) for now.
- **Mobile-native apps** — responsive web is the only client.
- **Real-time Kickstarter API integration** — out of scope for MVP; users paste details manually.
- **Recommendation engine ("similar successful campaigns")** — nice idea, not core to the predict-and-explain loop.

## Context

- **Brownfield project.** Training code, data, and empty backend/frontend scaffolding already exist. See `.planning/codebase/` for the full map. The plan is to *wire up* what's scaffolded, not rebuild it.
- **Resume angle is "both equally"**: the ML story (3 models, SHAP explainability, proper evaluation) AND the full-stack story (React + FastAPI + Postgres + Docker, end-to-end) both need to be showcase-quality. Neither side can be a token demo.
- **Primary user is a casual backer** who is curious and a little risk-averse. The UI must be legible to someone who has never heard of SHAP — explanations need to be readable, not just technically correct.
- **Known tech debt (see `.planning/codebase/CONCERNS.md`)**: preprocessing is duplicated across three training scripts; the model isn't saved; `backend/services/` and `backend/models/` are empty; pages are placeholders; unused `xgboost` and `shap` deps are declared but not used.
- **Single developer, no deadline.** The constraint is attention and motivation, not calendar time — favor finishing one slice end-to-end over scaffolding many half-done layers.

## Constraints

- **Tech stack**: React (Vite) + FastAPI + Postgres + PyTorch — locked. Already scaffolded; switching would throw away real work.
- **Explainability**: SHAP must be the source of the "why" behind each prediction — it's already declared as a dependency and it's the resume story.
- **Model quality**: comparison must be honest — the three models (NN, XGBoost, LogReg) get the same train/test split and the same evaluation metrics, with results shown to the user.
- **Performance**: prediction response < 2–3 seconds end-to-end. Model artifacts loaded once at startup, never per-request.
- **Portability**: `docker compose up` from a fresh clone must bring the entire stack online with no manual setup beyond `.env` values.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| React (Vite) + FastAPI split, not a single full-stack framework (Next/Django) | Already scaffolded; also makes the resume story "I built two services that talk" rather than "I used a batteries-included framework" | — Pending |
| PyTorch neural net as the primary model | Already trained and tuned; swapping to Keras/sklearn only would erase sunk work | ✓ Good |
| Use Postgres for prediction history despite no auth | Scaffold already exists in `docker-compose.yml`; gives the project a real database story for the resume without inventing user accounts | — Pending |
| Deployment deferred until product is demo-ready | Avoids rabbit-holing on hosting decisions before the core loop works | — Pending |
| Static CSV as the only data source for MVP | Matches "resume piece first, live product later" framing; scraping is a huge scope explosion | — Pending |

---
*Last updated: 2026-04-14 after initial /gsd-new-project questioning*
