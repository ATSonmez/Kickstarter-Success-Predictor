# Codebase Structure

**Analysis Date:** 2026-04-14

## Directory Layout

```
Kickstarter-Success-Predictor/
├── backend/                          # FastAPI backend server
│   ├── models/                       # [EMPTY] ML model files location
│   ├── services/                     # [EMPTY] Business logic services
│   ├── main.py                       # FastAPI app, routes, CORS config
│   ├── database.py                   # SQLAlchemy engine, SessionLocal, dependency
│   ├── db_models.py                  # ORM models: Prediction, ModelMetric
│   ├── requirements.txt              # Python dependencies
│   ├── Dockerfile                    # Backend container definition
│   ├── .env                          # Backend environment config (secrets)
│   └── .env.example                  # Environment template
├── frontend/                         # React Vite frontend
│   ├── src/
│   │   ├── pages/                    # Page components (routed)
│   │   │   ├── PredictPage.jsx       # Prediction form page [STUB]
│   │   │   ├── DashboardPage.jsx     # EDA dashboard page [STUB]
│   │   │   ├── PerformancePage.jsx   # Model performance page [STUB]
│   │   │   └── HistoryPage.jsx       # Prediction history page [STUB]
│   │   ├── components/               # [EMPTY] Reusable UI components
│   │   ├── assets/                   # [EMPTY] Images, fonts, static files
│   │   ├── App.jsx                   # Root router component, navigation
│   │   ├── main.jsx                  # React DOM entry point
│   │   ├── api.js                    # Axios instance, baseURL config
│   │   ├── index.css                 # Tailwind import
│   │   └── App.css                   # [LIKELY EMPTY] App-specific styles
│   ├── public/                       # [EMPTY] Static files served at root
│   ├── package.json                  # Dependencies, dev scripts
│   ├── vite.config.js                # Vite + React + Tailwind config
│   ├── eslint.config.js              # ESLint rules for React
│   └── node_modules/                 # [EXCLUDED] Installed packages
├── docker-compose.yml                # Microservices orchestration
├── .gitignore                        # Git ignore rules
├── .env                              # Root environment config (secrets)
├── .env.example                      # Root environment template
├── .planning/
│   └── codebase/                     # [NEW] Architecture documentation
├── .vscode/                          # VS Code settings
├── kickstarterModel.py               # [ROOT] Model training script
├── kickstarterModel_testing.py       # [ROOT] Model testing script
├── hyperparameter_search.py          # [ROOT] Hyperparameter tuning
├── kickstarter_data_with_features.csv # [ROOT] Training dataset (excluded from git)
└── README.md                         # Project documentation
```

## Directory Purposes

**backend/:**
- Purpose: FastAPI REST API server for predictions and metrics
- Contains: Main application logic, database models, configuration
- Key files: `main.py` (entry point), `database.py` (DB setup), `db_models.py` (schema)

**backend/models/:**
- Purpose: Trained ML models (PyTorch .pt files, joblib .pkl files)
- Contains: Serialized model weights and inference artifacts
- Key files: [To be populated] - Expected: model.pt, scaler.pkl, etc.

**backend/services/:**
- Purpose: Business logic services (prediction, metric calculation, data preprocessing)
- Contains: Service classes wrapping model inference
- Key files: [To be implemented] - Expected: prediction_service.py, metrics_service.py, etc.

**frontend/src/:**
- Purpose: React application source code
- Contains: Components, pages, routing, API client
- Key files: `App.jsx` (router), `main.jsx` (bootstrap), `api.js` (HTTP client)

**frontend/src/pages/:**
- Purpose: Page-level components matching routes
- Contains: Four main pages (all stubs referencing Phase 4)
- Key files: `PredictPage.jsx`, `DashboardPage.jsx`, `PerformancePage.jsx`, `HistoryPage.jsx`

**frontend/src/components/:**
- Purpose: Reusable UI components (forms, charts, cards, etc.)
- Contains: [Empty] - To be populated with shared components
- Key files: [To be created]

**frontend/src/assets/:**
- Purpose: Static assets (images, fonts, SVG icons)
- Contains: [Empty] - To be populated with project media
- Key files: [To be created]

## Key File Locations

**Entry Points:**
- `backend/main.py`: FastAPI application initialization, route mounting
- `frontend/src/main.jsx`: React DOM render target, StrictMode wrapper
- `frontend/src/App.jsx`: Browser Router initialization, navigation structure
- `docker-compose.yml`: Service orchestration startup configuration

**Configuration:**
- `backend/requirements.txt`: Python dependency versions (FastAPI, SQLAlchemy, torch, scikit-learn, xgboost, etc.)
- `frontend/package.json`: Node dependencies (React, Vite, Tailwind, Chart.js, Axios, React Router)
- `docker-compose.yml`: Container images, ports, environment variables, service dependencies
- `vite.config.js`: Frontend build configuration with React and Tailwind plugins
- `eslint.config.js`: Frontend linting rules for React code style

**Core Logic:**
- `backend/main.py`: API endpoint stubs (root, health check)
- `backend/database.py`: SQLAlchemy engine, SessionLocal factory, get_db dependency
- `backend/db_models.py`: Prediction and ModelMetric ORM schemas
- `frontend/src/api.js`: Axios HTTP client with baseURL configuration
- `frontend/src/App.jsx`: React Router with 4-page navigation structure

**Testing:**
- `kickstarterModel_testing.py`: Model evaluation script (root level)
- `hyperparameter_search.py`: Hyperparameter optimization script (root level)
- [No frontend test files present]

## Naming Conventions

**Files:**
- Python: `snake_case.py` (e.g., `main.py`, `db_models.py`, `kickstarterModel.py`)
- JavaScript: `camelCase.js` or `PascalCase.jsx` for components (e.g., `api.js`, `App.jsx`, `PredictPage.jsx`)
- CSS: Match component name or purpose (e.g., `App.css`, `index.css`)
- Config: Kebab-case or dotfiles (e.g., `eslint.config.js`, `vite.config.js`, `.env`)

**Directories:**
- Lowercase, descriptive names (e.g., `backend`, `frontend`, `models`, `services`, `pages`, `components`)

**React Components:**
- PascalCase filenames for components (e.g., `PredictPage.jsx`, `DashboardPage.jsx`)
- Functional components with default export

## Where to Add New Code

**New Backend Route:**
- Primary code: Add handler function in `backend/main.py` with @app.get() or @app.post() decorator
- Database access: Inject `db: Session = Depends(get_db)` parameter
- Models: Define request/response Pydantic models inline or in separate `schemas.py`
- Example location: `backend/main.py` (lines after existing endpoints)

**New Service/Business Logic:**
- Implementation: Create `backend/services/[feature_name]_service.py`
- Import in `backend/main.py` and use within route handlers
- Example: `backend/services/prediction_service.py` for model inference logic

**New Frontend Page:**
- Implementation: Create `frontend/src/pages/[FeatureName]Page.jsx` as functional component
- Route definition: Import in `frontend/src/App.jsx` and add `<Route>` element
- Navigation: Add `<NavLink>` in App.jsx navigation bar

**New Frontend Component:**
- Implementation: Create `frontend/src/components/[ComponentName].jsx`
- Usage: Import in page or other components as needed
- Example: `frontend/src/components/PredictionForm.jsx`

**Model Training Code:**
- Location: Root level scripts (following current pattern of `kickstarterModel.py`, `hyperparameter_search.py`)
- Output: Save trained models to `backend/models/` directory with .pt or .pkl extensions
- Note: Scripts are excluded from production builds (Docker runs inference only)

**Utilities/Helpers:**
- Backend: Create `backend/utils.py` or `backend/utils/[feature].py` for shared functions
- Frontend: Create `frontend/src/utils/[feature].js` for shared utility functions

## Special Directories

**backend/models/:**
- Purpose: Store trained ML models
- Generated: Yes (by `kickstarterModel.py`)
- Committed: No (models/ directory is empty, models are in .gitignore as *.pkl and *.pt)

**backend/services/:**
- Purpose: Service layer for business logic
- Generated: No
- Committed: Yes (Python source code)

**frontend/node_modules/:**
- Purpose: Installed Node.js packages
- Generated: Yes (by npm/yarn install)
- Committed: No (in .gitignore)

**frontend/dist/:**
- Purpose: Frontend production build output
- Generated: Yes (by vite build)
- Committed: No (in .gitignore)

**.planning/codebase/:**
- Purpose: GSD architecture documentation (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: No
- Committed: Yes (reference docs)

**postgres_data/:**
- Purpose: PostgreSQL database volume mount (Docker)
- Generated: Yes (by PostgreSQL container)
- Committed: No (in .gitignore)

---

*Structure analysis: 2026-04-14*
