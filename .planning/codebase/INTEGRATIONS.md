# External Integrations

**Analysis Date:** 2026-04-14

## APIs & External Services

**Internal API:**
- Kickstarter Success Predictor API - Backend REST API for predictions
  - SDK/Client: axios (JavaScript HTTP client)
  - Authentication: None detected
  - Base URL: Environment variable `VITE_API_URL` (frontend)
  - Endpoints: `/` (health check), `/health` (status check) - additional endpoints not yet implemented

**Third-Party Services:**
- Not detected - No external API integrations (Stripe, Supabase, AWS SDK, etc.)

## Data Storage

**Databases:**
- PostgreSQL 16
  - Connection: Environment variable `DATABASE_URL`
  - Client: SQLAlchemy 2.0.40 ORM
  - Connection parameters via psycopg2-binary 2.9.10
  - File: `backend/database.py` manages engine and session factory

**File Storage:**
- Local filesystem only
  - CSV data file: `kickstarter_data_with_features.csv` (98MB)
  - ML models: Trained via PyTorch, persisted via joblib

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- Custom/None - No authentication system implemented
  - Implementation: FastAPI CORS middleware only (allows frontend origin)
  - File: `backend/main.py` configures CORS with hardcoded/env-based frontend URL
  - Current setup: Open API with CORS protection to specific frontend URL

## Monitoring & Observability

**Error Tracking:**
- None detected

**Logs:**
- Python logging via print statements only
  - Training logs: `kickstarterModel.py` uses print()
  - Testing logs: `kickstarterModel_testing.py` uses print()

## CI/CD & Deployment

**Hosting:**
- Docker containers (local or cloud-agnostic)
  - Backend: `backend/Dockerfile` builds Python 3.12 slim image
  - Frontend: `frontend/Dockerfile` builds Node 22 slim image
  - Database: PostgreSQL 16 official Docker image

**CI Pipeline:**
- Not detected - No GitHub Actions, GitLab CI, or other CI/CD configuration files

**Orchestration:**
- Docker Compose (via `docker-compose.yml`)
  - Services: db (PostgreSQL), pgadmin (PgAdmin 4), backend (FastAPI), frontend (React/Vite)
  - Port mappings: 5432 (PostgreSQL), 5050 (PgAdmin), 8000 (Backend API), 5173 (Frontend)

## Environment Configuration

**Required env vars:**

**Docker Compose Level (.env):**
- `POSTGRES_USER` - PostgreSQL username
- `POSTGRES_PASSWORD` - PostgreSQL password
- `POSTGRES_DB` - Database name
- `PGADMIN_EMAIL` - PgAdmin admin email
- `PGADMIN_PASSWORD` - PgAdmin admin password

**Backend (`backend/.env`):**
- `DATABASE_URL` - PostgreSQL connection string (format: postgresql://user:password@host:port/dbname)
- `FRONTEND_URL` - Frontend origin for CORS (default: http://localhost:5173)

**Frontend (`frontend/.env`):**
- `VITE_API_URL` - Backend API base URL (default: http://localhost:8000)

**Secrets location:**
- `.env` files (present but not committed, see `.gitignore`)
- Example configurations provided in `.env.example` files at project root and service directories

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Database Schema

**Tables:**

**predictions:**
- `id` (Integer, Primary Key)
- `model_name` (String)
- `category` (String)
- `country` (String)
- `goal` (Float)
- `name_len` (Integer)
- `blurb_len` (Integer)
- `duration_days` (Integer)
- `prep_days` (Integer)
- `probability` (Float)
- `prediction` (Boolean)
- `created_at` (DateTime, server default)

**model_metrics:**
- `id` (Integer, Primary Key)
- `model_name` (String)
- `accuracy` (Float)
- `precision` (Float)
- `recall` (Float)
- `f1_score` (Float)
- `auc_score` (Float)
- `trained_at` (DateTime, server default)

Location: `backend/db_models.py` defines both SQLAlchemy models

## ML Model Integration

**Framework:**
- PyTorch 2.7.1 - Neural network models
- scikit-learn 1.7.0 - Preprocessing and metrics
- XGBoost 3.0.2 - Gradient boosting models
- SHAP 0.47.2 - Model interpretability

**Training:**
- Script: `kickstarterModel.py` - Main training pipeline
- Data: `kickstarter_data_with_features.csv` (98MB Kickstarter dataset)
- Output: Trained models persisted via joblib

**Testing:**
- Script: `kickstarterModel_testing.py` - Model evaluation

**Hyperparameter Tuning:**
- Script: `hyperparameter_search.py` - GridSearch/RandomSearch for model optimization

---

*Integration audit: 2026-04-14*
