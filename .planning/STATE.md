# Project State: Kickstarter Success Predictor

**Last updated:** 2026-04-14
**Current phase:** Phase 1 — Foundation (not started)
**Mode:** interactive / standard / parallel

## Where we are

- Project initialized via `/gsd-new-project` on a brownfield scaffold (React+Vite frontend, FastAPI backend, PyTorch training scripts at repo root, docker-compose with Postgres).
- Research complete: `.planning/research/{STACK,FEATURES,ARCHITECTURE,PITFALLS,SUMMARY}.md`.
- Codebase mapped: `.planning/codebase/` (7 docs).
- Requirements locked: 38 v1 requirements (FND×6, PRD×12, EDA×4, HST×5, INT×7, QLT×4). CMP dropped entirely; XGBoost/LogReg/Performance page deferred to v2.
- Roadmap locked: 4 phases, strict numeric order.

## Phase status

| Phase | Status | Plan |
|---|---|---|
| 1. Foundation | Not started | TBD |
| 2. Core Predict Loop | Not started | TBD |
| 3. Supporting Pages | Not started | TBD |
| 4. Integration and Polish | Not started | TBD |

## Next action

Run `/gsd-plan-phase 1` to plan Phase 1 (Foundation).

## Key decisions

- v1 = neural net only (no model comparison)
- Shared `backend/services/preprocessing.py` is the single source of truth
- FastAPI lifespan loads all artifacts once into `app.state`
- SHAP: DeepExplainer with 50–100 row background sample, built at startup
- Chart.js (already installed), Tailwind 4, react-hook-form (only new dep)
- Deployment deferred; local `docker compose up` is the v1 bar
