# Pitfalls Research

**Domain:** ML prediction web app — Kickstarter classification, FastAPI + React + PyTorch + SHAP
**Researched:** 2026-04-14
**Confidence:** HIGH — all pitfalls grounded in direct inspection of this codebase.

---

## Critical Pitfalls

### 1. Scaler fitted on leaky columns (real bug in current code)

`kickstarterModel.py` calls `scaler.fit_transform(df[continuous_cols])` on a list that includes `pledged`, `usd_pledged`, and `backers_count`, then drops those as leakage afterward. The scaler's mean/variance is computed over post-campaign data. Fix ordering in the new `preprocessing.py`: (1) drop leakage, (2) drop metadata, (3) fit scaler, (4) save.
**Phase:** 1 (preprocessing consolidation).

### 2. Training/serving skew via preprocessing duplication

Three diverging copies in `kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`. Testing script already lacks `pos_weight` and BatchNorm. Adding a 4th inference copy guarantees silent wrong predictions.
**Fix:** one canonical `backend/services/preprocessing.py` imported by both training and inference.
**Phase:** 1.

### 3. Feature leakage audit incomplete

`static_usd_rate` is retained — may be a post-campaign exchange rate depending on the CSV snapshot. Document explicit `LAUNCH_TIME_FEATURES` vs `POST_CAMPAIGN_FEATURES` lists. Warning sign: accuracy > 85% on balanced test set → suspect leakage.
**Phase:** 1.

### 4. SHAP explainer choice and latency

`KernelExplainer` is unusably slow. Use `DeepExplainer(model, 50–100 row background_sample.pt)` for the NN, `TreeExplainer` for XGBoost, `LinearExplainer` for LogReg. Build all at FastAPI startup, not per-request. If concurrency matters, wrap `.shap_values(...)` in `asyncio.to_thread`.
**Phase:** 2 (serving).

### 5. Uncalibrated sigmoid displayed as "probability"

NN uses `BCEWithLogitsLoss` with `pos_weight`. Raw sigmoid is a monotone score, not calibrated frequency. Show a reliability diagram; if the curve deviates from the diagonal, either apply temperature scaling or label the output "success score" and show confidence tiers (High >0.80, Moderate 0.60–0.80, Low <0.60).
**Phase:** 3 (evaluation + predict UI).

### 6. Model comparison with different preprocessing per model

XGBoost doesn't need scaling, LogReg does. If the train/test split or column order differs across models, comparison is invalid. Single `X_train/X_test/y_train/y_test` artifact, shared `random_state=42`, identical feature set.
**Phase:** 3 (multi-model training).

### 7. Model artifacts not saved

`kickstarterModel.py` trains in memory and never calls `torch.save`. Save four artifacts: `nn_model.pt` (state_dict), `scaler.pkl`, `feature_columns.json`, `model_metadata.json`.
**Phase:** 1 (model persistence).

### 8. Docker image bloat from default PyTorch install

Default pip install pulls CUDA wheel (~2.5 GB). Image hits 3–5 GB. In `backend/Dockerfile`, install torch CPU-only first:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Add `.dockerignore` excluding `node_modules`, `__pycache__`, CSV, `*.pt`.
**Phase:** 4 (Docker finalization).

### 9. `VITE_API_URL` broken in Docker + CORS misconfiguration

Vite embeds `VITE_` env vars at **build time**, not runtime. `docker-compose.yml` currently passes it at runtime — `import.meta.env.VITE_API_URL` will be `undefined`. Fix: declare `ARG VITE_API_URL` / `ENV VITE_API_URL=$VITE_API_URL` in `frontend/Dockerfile` and pass via `build.args`. Also: add a Vite dev proxy in `vite.config.js` (`server.proxy`) to eliminate CORS during local dev.
**Phase:** 4.

### 10. Pydantic v2 breaking changes

Project uses Pydantic 2.11.3. SQLAlchemy response schemas need `model_config = ConfigDict(from_attributes=True)` (not v1 `orm_mode`). Use `@field_validator`, not `@validator`.
**Phase:** 2 (first `/predict` schema).

### 11. Empty-scaffolding trap

Five pages and multiple endpoints tempting early stubs. Rule: no new endpoint or page is started until the previous one returns real data in the browser. Predict first → EDA → Performance → History.
**Phase:** enforced by roadmap ordering.

### 12. Probability displayed without context

"67%" is meaningless without baseline comparison. Show confidence tier, historical category base rate alongside the number, and flag model disagreement explicitly if models split by more than ~20 points.
**Phase:** 2 (predict UI).

### 13. `database.py` crashes on fresh clone

`create_engine(None)` raises a cryptic `ArgumentError` if `DATABASE_URL` is missing. Add a one-line guard:
```python
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required but not set")
```
Plus postgres `healthcheck` in `docker-compose.yml` and `depends_on: db: condition: service_healthy`.
**Phase:** 2 (first DB wiring).

---

## Technical Debt Patterns

| Shortcut | Long-term Cost | When Acceptable |
|---|---|---|
| Scaler fitted before leakage drop | Wrong predictions, silently | Never |
| Three separate training entrypoints | Divergence inevitable | Never — Phase 1 |
| Skip `torch.save` | Inference impossible | Never |
| Default PyTorch install (CUDA) | 3–5 GB image | Fix before sharing |
| `allow_headers=["*"]` CORS | Overly permissive | OK for localhost MVP only |
| No input validation on `/predict` | Garbage-in errors | Never — Pydantic from day 1 |

---

## Integration Gotchas

| Integration | Mistake | Correct Approach |
|---|---|---|
| SHAP + PyTorch | `KernelExplainer` | `DeepExplainer` w/ 50–100 row background, loaded at startup |
| SHAP + XGBoost | `KernelExplainer` | `TreeExplainer` — exact, ms latency, no background |
| Vite + Docker | Runtime `VITE_API_URL` | `ARG`/`ENV` at build time |
| FastAPI + SQLAlchemy + Pydantic v2 | Forgetting `from_attributes=True` | `ConfigDict(from_attributes=True)` on every ORM-wrapping schema |
| FastAPI startup | `@app.on_event("startup")` | `@asynccontextmanager` lifespan |
| Compose backend + Postgres | Backend starts before DB ready | `healthcheck` + `condition: service_healthy` |

---

## Performance Traps

| Trap | Symptoms | Prevention |
|---|---|---|
| SHAP computed per request | 5–30s latency | Pre-load background, cache explainer, startup init |
| Model loaded per request | Cold-start per prediction | Lifespan hook → `app.state` |
| Sync SHAP in async endpoint | Event loop blocked | `asyncio.to_thread(...)` |
| Non-deterministic `pd.get_dummies` column order | Feature vector drift | Save and reindex against `feature_columns.json` |
| Full CSV loaded into FastAPI | 100+ MB RAM, slow startup | Precompute EDA aggregates; never load raw CSV at serve time |

---

## Security Mistakes

| Mistake | Prevention |
|---|---|
| `allow_headers=["*"]` | Explicit allowlist before deployment |
| No `DATABASE_URL` guard | `RuntimeError` on startup if absent |
| No rate limiting on `/predict` | `slowapi` before public exposure |
| Model artifacts in git | `.gitignore`: `*.pt`, `*.pkl`, `backend/models/` |

---

## UX Pitfalls

| Pitfall | Better Approach |
|---|---|
| Raw sigmoid labeled "probability" | Label as "success score"; show confidence tier |
| SHAP values as raw numbers | Bar chart + plain-English sentences |
| Models disagree, only one shown | Surface disagreement explicitly |
| No baseline comparison | Show historical category success rate next to the prediction |
| Long load with no feedback | Skeleton/spinner on form submit |

---

## "Looks Done But Isn't" Checklist

- [ ] `/predict` returns from weights loaded from disk (not a fresh random-init model)
- [ ] Inference feature count == `model.network[0].in_features` (startup assertion)
- [ ] SHAP values sum to approximately `log_odds(prediction) − log_odds(base_rate)` on one known example
- [ ] All three models evaluated on the same held-out test split with `random_state=42`
- [ ] `docker compose up` from a freshly cloned repo (no cached layers, no local `.env`) reaches a working predict form
- [ ] Reliability diagram rendered on Performance page
- [ ] Prediction history persists across backend container restarts
- [ ] Browser DevTools shows no CORS errors on any API call

---

## Pitfall-to-Phase Mapping

| Pitfall | Phase |
|---|---|
| Scaler ordering / preprocessing duplication / leakage audit | 1 — Preprocessing |
| Model artifacts missing | 1 — Model persistence |
| `DATABASE_URL` crash guard | 2 — First DB wiring |
| Pydantic v2 syntax | 2 — `/predict` endpoint |
| SHAP latency / explainer choice | 2 — SHAP integration |
| Empty-scaffolding trap | All phases (ordering discipline) |
| Uncalibrated probabilities / probability w/o context | 2–3 — Predict UI + evaluation |
| Multi-model fairness | 3 — Multi-model training |
| Docker image bloat / CORS / `VITE_API_URL` | 4 — Docker finalization |

---

## Open Questions

1. Class imbalance ratio in `kickstarter_data_with_features.csv` — determines how aggressive calibration correction needs to be.
2. Is `static_usd_rate` actually post-campaign in this snapshot? Needs temporal audit.
3. SHAP 0.47.2 thread-safety with PyTorch under concurrent requests — test before shipping.

---

## Sources

Direct inspection of `kickstarterModel.py`, `kickstarterModel_testing.py`, `backend/database.py`, `backend/main.py`, `docker-compose.yml`, `frontend/vite.config.js`, `backend/Dockerfile`; `.planning/codebase/CONCERNS.md`; FastAPI + Pydantic v2 migration docs; SHAP library design notes.
