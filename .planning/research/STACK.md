# Technology Stack

**Project:** Kickstarter Success Predictor
**Researched:** 2026-04-14
**Stack status:** Brownfield — core stack LOCKED (React/Vite + FastAPI + PyTorch + Postgres + Docker)

---

## Key Findings

1. **Nothing core needs replacing — only one npm package needs adding.** Audit of `backend/requirements.txt` and `frontend/package.json` shows Chart.js 4.5.1, react-chartjs-2 5.3.1, Tailwind 4.2.2, axios, torch 2.7.1, shap 0.47.2, xgboost 3.0.2, joblib 1.5.1, scikit-learn 1.7.0 all already installed. Only missing piece: a form library. **Add `react-hook-form`.**
2. **FastAPI `lifespan` context manager is the correct model-loading pattern.** `@app.on_event("startup")` is deprecated. Load all model + scaler + feature columns + SHAP explainers once into a module-level `ml_models` dict.
3. **Keep Chart.js — do not add Recharts or Plotly.** Chart.js is already installed and the app already uses Tailwind. Recharts adds zero functionality; Plotly is 3–4× heavier.
4. **SHAP: use `GradientExplainer` (PyTorch), `TreeExplainer` (XGBoost), `LinearExplainer` (LogReg).** Initialize once at startup with a 100–200 row background dataset. Do NOT use `KernelExplainer` per request — O(n²) on 100+ one-hot features will breach the 2–3s budget.
5. **Tailwind 4 + component-local state is sufficient.** No MUI (class conflicts, ~200KB overhead), no Redux/Zustand (4 pages, no cross-page shared state).

---

## What Is Already Installed

### Backend (`backend/requirements.txt`)
| Package | Version |
|---|---|
| fastapi | 0.115.12 |
| uvicorn | 0.34.2 |
| pydantic | 2.11.3 |
| sqlalchemy | 2.0.40 |
| psycopg2-binary | 2.9.10 |
| python-dotenv | 1.1.0 |
| torch | 2.7.1 |
| scikit-learn | 1.7.0 |
| xgboost | 3.0.2 |
| shap | 0.47.2 |
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| joblib | 1.5.1 |

### Frontend (`frontend/package.json`)
| Package | Version |
|---|---|
| react | 19.2.4 |
| react-dom | 19.2.4 |
| react-router-dom | 7.14.0 |
| axios | 1.14.0 |
| chart.js | 4.5.1 |
| react-chartjs-2 | 5.3.1 |
| tailwindcss | 4.2.2 (dev) |
| @tailwindcss/vite | 4.2.2 (dev) |
| vite | 8.0.1 (dev) |
| @vitejs/plugin-react | 6.0.1 (dev) |

---

## New Packages to Add

| Package | Version | Purpose | Install |
|---|---|---|---|
| react-hook-form | ^7.54 | Form handling for PredictPage (uncontrolled inputs, no re-renders per keystroke) | `npm install react-hook-form` |

No new Python packages required.

---

## Decision 1 — PyTorch Model Serving: FastAPI Lifespan

**Use:** `@asynccontextmanager` lifespan + module-level `ml_models` dict. Confidence: HIGH.

```python
from contextlib import asynccontextmanager
import torch, joblib
from fastapi import FastAPI

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["nn"] = load_nn_model("artifacts/nn_model_weights.pt")
    ml_models["xgb"] = joblib.load("artifacts/xgb_model.joblib")
    ml_models["logreg"] = joblib.load("artifacts/logreg_model.joblib")
    ml_models["scaler"] = joblib.load("artifacts/scaler.joblib")
    ml_models["feature_cols"] = joblib.load("artifacts/feature_cols.joblib")
    build_shap_explainers(ml_models)
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)
```

Save PyTorch as `torch.save(model.state_dict(), path)` + importable architecture class — NOT full pickle (breaks across PyTorch versions). Use `joblib.dump/load` for sklearn and XGBoost.

---

## Decision 2 — Pydantic v2 Request/Response Models

Confidence: HIGH. Place in `backend/models/schemas.py`:

```python
from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    category: str
    country: str
    goal: float = Field(gt=0)
    name_len: int = Field(ge=1)
    blurb_len: int = Field(ge=0)
    duration_days: int = Field(ge=1, le=365)
    prep_days: int = Field(ge=0)
    model_name: Literal["nn", "xgb", "logreg"] = "nn"

class SHAPFeature(BaseModel):
    feature: str
    value: float

class PredictResponse(BaseModel):
    probability: float
    prediction: bool
    model_name: str
    shap_values: list[SHAPFeature]
```

---

## Decision 3 — SHAP Integration

Confidence: MEDIUM (GradientExplainer selection from training knowledge; benchmark at implementation time).

```python
import shap, torch, numpy as np

def build_shap_explainers(ml_models: dict):
    bg = np.load("artifacts/background_data.npy").astype(np.float32)
    ml_models["shap_nn"]     = shap.GradientExplainer(ml_models["nn"], torch.tensor(bg))
    ml_models["shap_xgb"]    = shap.TreeExplainer(ml_models["xgb"])
    ml_models["shap_logreg"] = shap.LinearExplainer(ml_models["logreg"], bg)

def get_top_shap(explainer, X_row, feature_names, top_n=10):
    vals = explainer.shap_values(X_row)
    if isinstance(vals, list):
        vals = vals[1]
    pairs = sorted(zip(feature_names, vals[0]), key=lambda x: abs(x[1]), reverse=True)
    return [{"feature": f, "value": float(v)} for f, v in pairs[:top_n]]
```

**Never use `KernelExplainer` per-request** — O(n_features²) breaks the 2–3s budget on 100+ one-hot features.

---

## Decision 4 — Charting: Chart.js 4.5.1 (already installed)

| Page | Chart | Component |
|---|---|---|
| Dashboard — category success rate | Horizontal Bar | `<Bar indexAxis="y">` |
| Dashboard — country success rate | Horizontal Bar | `<Bar indexAxis="y">` |
| Dashboard — goal distribution | Bar | `<Bar>` |
| Performance — model AUC/F1 comparison | Grouped Bar | `<Bar>` |
| Performance — ROC curve | Line | `<Line>` |
| Predict — SHAP top features | Horizontal Bar | `<Bar indexAxis="y">` |
| Predict — probability gauge | Doughnut / horizontal bar | `<Doughnut>` or CSS gradient bar |

---

## Decision 5 — Styling: Tailwind 4.2.2 (already wired)

Tailwind 4 is wired via `@tailwindcss/vite` and actively used in `frontend/src/App.jsx`. No MUI — conflicts with existing Tailwind classes.

---

## Decision 6 — Form Handling: react-hook-form ^7.54

Uncontrolled inputs (refs), no per-keystroke re-renders, React 19 compatible, zero dependencies, integrates with Tailwind input classes via `{...register("field", validationRules)}`.

---

## Decision 7 — Async Strategy

`async def` endpoints + synchronous inference inline. PyTorch CPU inference on a small tabular model is <10ms. SQLAlchemy 2.0 + psycopg2 (sync) works inside `async def` via FastAPI's thread pool. Do NOT switch to asyncpg. If SHAP proves slow (>100ms), wrap only the SHAP call in `run_in_executor`.

---

## Decision 8 — Frontend State

Component-local `useState` / `useEffect` per page. Custom hooks (`usePrediction`, `useEDAStats`) for encapsulation. No Redux, Zustand, or Context.

---

## Alternatives Ruled Out

| Category | Recommended | Rejected | Reason |
|---|---|---|---|
| Charting | Chart.js (keep) | Recharts | Already installed; zero added value |
| Charting | Chart.js (keep) | Plotly.js | 3–4× heavier; overkill |
| Styling | Tailwind (keep) | MUI | Class conflicts; 200KB overhead |
| Forms | react-hook-form | Formik | Controlled inputs; heavier |
| Forms | react-hook-form | Native useState | Verbose for 8+ fields with validation |
| SHAP | GradientExplainer | KernelExplainer | O(n²) — too slow |
| FastAPI startup | `lifespan` | `@app.on_event` | Deprecated |
| Model save | `state_dict` + class | Full pickle | Breaks across PyTorch versions |

---

## Artifact Directory to Create

```
backend/artifacts/
  nn_model_architecture.py   # importable model class
  nn_model_weights.pt        # torch.save(model.state_dict(), ...)
  xgb_model.joblib
  logreg_model.joblib
  scaler.joblib
  feature_cols.joblib
  background_data.npy        # np.save(..., X_train_sample[:200])
```

---

## Sources

- FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/ — HIGH
- FastAPI Pydantic body: https://fastapi.tiangolo.com/tutorial/body/ — HIGH
- Tailwind v4 + Vite: https://tailwindcss.com/docs/installation — HIGH
- Package version audit of `backend/requirements.txt` and `frontend/package.json` — HIGH
- SHAP explainer selection from training knowledge — MEDIUM (verify during implementation)
- react-hook-form current version — MEDIUM (verify before install)

---

## Open Questions

1. `GradientExplainer` vs `DeepExplainer` for the specific NN activation functions — validate during implementation phase.
2. SHAP response latency under real conditions — benchmark with 100-row background before deciding on `run_in_executor`.
3. react-hook-form exact latest patch version — check npm before install.
