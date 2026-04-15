# Architecture Research

**Domain:** ML-powered prediction web app (React + FastAPI + PyTorch + SHAP + Postgres)
**Researched:** 2026-04-14
**Confidence:** HIGH (grounded in existing codebase + established FastAPI/PyTorch/SHAP patterns)

---

## Key Findings

- **Preprocessing consolidation is the single highest-leverage action.** Three training scripts share ~60 lines of verbatim preprocessing. A `backend/services/preprocessing.py` module with `fit_transform()` (training) and `transform_single()` (inference) eliminates skew. Scaler must be fitted once and saved as `scaler.pkl` alongside `feature_columns.pkl` — without the saved column list, one-hot inference silently produces wrong-shaped tensors.
- **FastAPI `lifespan` is the correct model-loading pattern.** Load all artifacts once into `app.state` at startup. The current `backend/main.py` has no lifespan hook and no model-save calls — both missing, both blocking the product.
- **SHAP explainers must be built at startup.** `DeepExplainer` requires a background dataset (~100 rows stored as `backend/models/background.pt`). Built once, it adds ~100–200ms per request — well within the 2–3s budget.
- **EDA stats should be precomputed JSON, not live Postgres queries.** Dataset is static; writing `eda_stats.json` in the training script and serving from memory is zero-latency and requires no schema expansion.
- **Build order is strictly dependency-ordered.** `preprocessing.py` → training/save → FastAPI lifespan → `/predict` endpoint → PredictPage. Nothing else is demoable.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     BROWSER (React + Vite)                       │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────┐  │
│  │ PredictPage  │ │ DashboardPage│ │Performance │ │ History  │  │
│  └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └────┬─────┘  │
│         │                │               │              │        │
│  ┌──────▼────────────────▼───────────────▼──────────────▼─────┐  │
│  │              api.js (Axios instance, baseURL env)           │  │
│  └──────────────────────────────┬──────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────┘
                                  │ HTTP/JSON
┌─────────────────────────────────▼───────────────────────────────┐
│                    FASTAPI BACKEND                                │
│  main.py  (lifespan → load artifacts; route mounting)             │
│                                                                   │
│  /predict            /eda/stats                                   │
│  /predict/{model}    /models/metrics                              │
│  /history            /health                                      │
│                                                                   │
│  services/                                                        │
│    preprocessing.py   prediction_service.py   eda_service.py      │
│                                                                   │
│  models/ (artifacts on disk, gitignored)                          │
│    kickstarter_nn.pt  scaler.pkl  feature_columns.pkl             │
│    xgboost_model.pkl  logreg_model.pkl  eda_stats.json            │
│    background.pt                                                  │
│                                                                   │
│  database.py + db_models.py (SQLAlchemy → Postgres)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                    POSTGRES (Docker)
                 predictions  /  model_metrics
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `main.py` lifespan hook | Load all model artifacts once at startup into `app.state` | `backend/main.py` |
| `preprocessing.py` | Single source of truth for feature transforms; used by training AND inference | `backend/services/preprocessing.py` |
| `prediction_service.py` | Accept raw input → preprocess → forward pass → SHAP → structured result | `backend/services/prediction_service.py` |
| `eda_service.py` | Serve precomputed EDA stats from memory | `backend/services/eda_service.py` |
| `schemas.py` | Pydantic request/response models | `backend/schemas.py` |
| `db_models.py` | SQLAlchemy ORM: `Prediction`, `ModelMetric` (exists) | `backend/db_models.py` |
| Training scripts | Import `preprocessing.py`, train, save artifacts | root `kickstarterModel.py` etc. |

---

## Recommended Project Structure

```
backend/
├── main.py                          # FastAPI app, lifespan, routes
├── database.py                      # SQLAlchemy engine + get_db (exists)
├── db_models.py                     # ORM: Prediction, ModelMetric (exists)
├── schemas.py                       # Pydantic request/response models
├── services/
│   ├── preprocessing.py             # THE shared preprocessing module
│   ├── prediction_service.py        # Inference + SHAP orchestration
│   └── eda_service.py               # EDA stats loading/serving
├── models/                          # gitignored artifacts
│   ├── kickstarter_nn.pt
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   ├── xgboost_model.pkl
│   ├── logreg_model.pkl
│   ├── eda_stats.json
│   └── background.pt
├── requirements.txt
└── Dockerfile

frontend/src/
├── pages/
│   ├── PredictPage.jsx              # Form + result
│   ├── DashboardPage.jsx            # EDA charts
│   ├── PerformancePage.jsx          # Metrics, ROC, confusion matrix
│   └── HistoryPage.jsx              # Past predictions table
├── components/
│   ├── PredictionForm.jsx
│   ├── PredictionResult.jsx         # Probability bar + SHAP sentences
│   ├── ShapChart.jsx                # Collapsible horizontal bar
│   ├── MetricsCard.jsx
│   └── HistoryTable.jsx
├── api.js                           # Axios instance (exists)
├── App.jsx                          # Router (exists)
└── main.jsx                         # Entry (exists)

# Root — offline training only
kickstarterModel.py
kickstarterModel_testing.py
hyperparameter_search.py
```

---

## Pattern 1: FastAPI Lifespan for Model Loading

```python
# backend/main.py
from contextlib import asynccontextmanager
import torch, joblib, json
from pathlib import Path
from services.preprocessing import KickstarterPreprocessor
from services.prediction_service import PredictionService

MODELS_DIR = Path(__file__).parent / "models"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.preprocessor = KickstarterPreprocessor.load(MODELS_DIR)
    app.state.models = {
        "neural_net": _load_nn(MODELS_DIR / "kickstarter_nn.pt", app.state.preprocessor.num_features),
        "xgboost":    joblib.load(MODELS_DIR / "xgboost_model.pkl"),
        "logreg":     joblib.load(MODELS_DIR / "logreg_model.pkl"),
    }
    background = torch.load(MODELS_DIR / "background.pt")
    app.state.prediction_service = PredictionService(
        app.state.models, app.state.preprocessor, background
    )
    app.state.eda_stats = json.loads((MODELS_DIR / "eda_stats.json").read_text())
    yield

app = FastAPI(title="Kickstarter Predictor API", lifespan=lifespan)
```

## Pattern 2: Shared Preprocessing (Training/Serving Parity)

`KickstarterPreprocessor` exposes:
- `fit_transform(df)` — training path; fits scaler, captures post-encoding column list
- `transform_single(raw_dict)` — inference path; builds zero-filled vector keyed on saved `feature_columns`, sets matching one-hot columns to 1
- `save(dir)` / `load(dir)` — serialize `scaler.pkl` and `feature_columns.pkl`

**Never call `pd.get_dummies()` on a one-row DataFrame at inference** — it produces only the columns present in that row. Zero-fill from the saved column list instead.

Training script adapter:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent / "backend"))
from services.preprocessing import KickstarterPreprocessor

preprocessor = KickstarterPreprocessor()
X, y = preprocessor.fit_transform(df)
# ... train ...
preprocessor.save(Path("backend/models"))
torch.save(model.state_dict(), "backend/models/kickstarter_nn.pt")
```

## Pattern 3: SHAP Explainers at Startup

Instantiate all three explainers in `PredictionService.__init__()`:

- `shap.DeepExplainer(nn_model, background_tensor)` — PyTorch NN
- `shap.TreeExplainer(xgboost_model)` — XGBoost (fast, no background)
- `shap.LinearExplainer(logreg_model, masker)` — LogReg

Save 100–200 random training rows to `backend/models/background.pt` during training. Return only top-8 features by `|shap_value|`.

**Latency budget (CPU):** preprocessing ~5ms + NN forward ~10ms + DeepExplainer ~100–200ms = ~300ms. Well under 2–3s.

---

## Data Flow — Prediction Request

```
PredictPage form
  │
POST /predict  {category, country, goal, name_len, blurb_len,
                duration_days, prep_days, model_name}
  ├─ Pydantic validates PredictRequest
  ├─ prediction_service.predict(raw_dict, model_name)
  │     ├─ preprocessor.transform_single(raw_dict)
  │     │     ├─ scale continuous cols via saved scaler
  │     │     └─ align to feature_columns (zero-fill unknowns)
  │     ├─ model forward pass → probability float
  │     └─ explainer.shap_values(X) → top-8 contributions
  ├─ db.add(Prediction(...)); db.commit()
  └─ return PredictResponse JSON
  │
PredictionResult.jsx: gradient bar + SHAP sentences + (collapsible) chart
```

## Data Flow — EDA Stats

Training script writes `backend/models/eda_stats.json` (groupby aggregations on the same CSV used for training). Lifespan loads it into `app.state.eda_stats`. `GET /eda/stats` returns the dict — zero DB, zero compute per request.

## Data Flow — History

`POST /predict` inserts a row into `predictions`. `GET /history?limit=50&offset=0` returns paginated rows ordered by `created_at DESC`. Anonymous — no user identity.

---

## Pydantic Schemas

```python
# backend/schemas.py

class PredictRequest(BaseModel):
    category: str
    country: str
    goal: float
    name_len: int
    blurb_len: int
    duration_days: int
    prep_days: int
    model_name: str = "neural_net"

class ShapFeature(BaseModel):
    feature: str
    value: float

class PredictResponse(BaseModel):
    probability: float
    prediction: bool
    model: str
    shap_top_features: list[ShapFeature]

class PredictionHistoryItem(BaseModel):
    id: int
    model_name: str
    category: str
    country: str
    goal: float
    probability: float
    prediction: bool
    created_at: datetime

class ModelMetricItem(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    trained_at: datetime
```

---

## Build Order

```
1. backend/services/preprocessing.py                [no deps]
        │
        ▼
2. Refactor training scripts → import preprocessing
   Run training → save artifacts to backend/models/
   (nn.pt, scaler.pkl, feature_columns.pkl, xgb.pkl, logreg.pkl,
    eda_stats.json, background.pt, ModelMetric rows in Postgres)
        │
        ▼
3. FastAPI lifespan + PredictionService instantiation       [needs 2]
        │
        ▼
4. schemas.py + /predict endpoint                           [needs 3]
        │
        ▼
5. PredictPage.jsx wired to /predict    ← FIRST DEMOABLE MILESTONE
        │
        ├──── (parallel after step 2) ────
        ▼         ▼              ▼
     6a /eda    6b /metrics    6c /history
        │         │              │
        └─────────┼──────────────┘
                  ▼
      7. Docker Compose E2E validation (fresh clone → up → all pages)
                  │
                  ▼
      8. Resume polish: responsive styling, README with screenshots
```

Steps 1–5 are the critical path. 6a/6b/6c can be built independently after step 2.

---

## Anti-Patterns

1. **Model loading per-request** — 500ms–3s deserialization, blocks event loop. → Lifespan hook.
2. **Duplicating preprocessing between training and inference** — silent divergence, wrong predictions without errors. → Shared module.
3. **`pd.get_dummies()` at inference time on a one-row DataFrame** — shape mismatch (only produces columns present in that row). → Zero-fill from saved `feature_columns`.
4. **SHAP explainers built per-request** — DeepExplainer runs background samples through the network on every request. → Build once at startup.
5. **Live EDA queries from Postgres** — requires importing ~100K rows; aggregation on every request. → Precomputed JSON.

---

## Integration Points

| Boundary | Mechanism |
|---|---|
| Training scripts → preprocessing.py | `sys.path.insert(0, "backend")` + import |
| lifespan → route handlers | `request.app.state` (no module globals) |
| Route handlers → PredictionService | Direct sync call (ML is CPU-bound) |
| PredictionService → DB | SQLAlchemy session via `Depends(get_db)` |
| Backend ↔ Frontend | REST/JSON over HTTP via axios `api.js` |

Docker Compose: add `healthcheck` on postgres; backend `depends_on: { db: { condition: service_healthy } }`.

---

## Scaling Targets

Portfolio project — 1–10 concurrent users. Current design is correct.

If later scaled: wrap sync ML inference in `run_in_executor` (100 users) or extract dedicated inference workers + Redis SHAP cache (1000+ users).

---

## Open Questions

1. **Feature name mapping.** `db_models.Prediction` stores `prep_days`/`duration_days`; raw CSV uses timedelta strings `create_to_launch`/`launch_to_deadline`. Confirm `transform_single()` maps user-friendly fields to the correct feature columns.
2. **`static_usd_rate`** is in the training continuous columns but is a currency conversion rate the user cannot supply at form time. Decide: drop and default to 1.0, or derive from currency.
3. **`background.pt` gitignore.** 100 rows × ~120 features is tiny — confirm this artifact is gitignored alongside the other model files.

---

## Sources

- FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/ — HIGH
- SHAP docs: https://shap.readthedocs.io/en/latest/ — HIGH
- Sculley et al., *Hidden Technical Debt in ML Systems* — training/serving skew — HIGH
- Existing `backend/database.py`, `backend/db_models.py` — HIGH
- Standard sklearn one-hot inference alignment — HIGH
