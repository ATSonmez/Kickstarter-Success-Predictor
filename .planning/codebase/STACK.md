# Technology Stack

**Analysis Date:** 2026-04-14

## Languages

**Primary:**
- JavaScript/JSX 19 - Frontend React application and build tooling
- Python 3.12 - Backend API server and ML model training
- SQL - Database schema and queries via SQLAlchemy ORM

**Secondary:**
- CSS (via Tailwind) - Frontend styling

## Runtime

**Environment:**
- Node.js 22 (slim) - Frontend development and build
- Python 3.12 slim - Backend API and ML workloads
- Docker - Containerization and orchestration

**Package Manager:**
- npm - JavaScript/Node.js dependencies
- pip - Python dependencies
- Lockfile: `frontend/package-lock.json` (present), `requirements.txt` with pinned versions (present)

## Frameworks

**Core:**
- FastAPI 0.115.12 - Backend REST API framework
- React 19.2.4 - Frontend UI library
- React Router DOM 7.14.0 - Frontend routing

**UI/Styling:**
- Tailwind CSS 4.2.2 - Utility-first CSS framework
- @tailwindcss/vite 4.2.2 - Vite integration for Tailwind
- Chart.js 4.5.1 - Data visualization library
- react-chartjs-2 5.3.1 - React wrapper for Chart.js

**Testing:**
- Not detected in dependencies

**Build/Dev:**
- Vite 8.0.1 - Frontend bundler and dev server
- ESLint 9.39.4 - JavaScript linting
- @vitejs/plugin-react 6.0.1 - React support for Vite
- Uvicorn 0.34.2 - ASGI server for FastAPI

## Key Dependencies

**Critical:**
- FastAPI 0.115.12 - Core backend framework for REST API
- SQLAlchemy 2.0.40 - ORM for database operations
- torch 2.7.1 - Deep learning library for neural network models
- scikit-learn 1.7.0 - Machine learning utilities and preprocessing
- xgboost 3.0.2 - Gradient boosting for predictive models
- pandas 3.0.2 - Data manipulation and analysis
- numpy 2.4.4 - Numerical computing
- React 19.2.4 - Frontend framework
- axios 1.14.0 - HTTP client for API calls

**Infrastructure:**
- psycopg2-binary 2.9.10 - PostgreSQL database adapter
- python-dotenv 1.1.0 - Environment variable loading
- pydantic 2.11.3 - Data validation and serialization
- joblib 1.5.1 - Serialization and job scheduling
- shap 0.47.2 - Model interpretability
- uvicorn 0.34.2 - ASGI server

## Configuration

**Environment:**
- Backend `.env` file (via `backend/.env.example`):
  - `DATABASE_URL` - PostgreSQL connection string
  - `FRONTEND_URL` - Frontend origin for CORS
- Frontend `.env` file (via `frontend/.env.example`):
  - `VITE_API_URL` - Backend API base URL
- Docker Compose `.env`:
  - `POSTGRES_USER` - Database username
  - `POSTGRES_PASSWORD` - Database password
  - `POSTGRES_DB` - Database name
  - `PGADMIN_EMAIL` - PgAdmin admin email
  - `PGADMIN_PASSWORD` - PgAdmin admin password

**Build:**
- `frontend/vite.config.js` - Vite configuration with React and Tailwind plugins
- `frontend/eslint.config.js` - ESLint configuration
- `backend/Dockerfile` - Python 3.12 slim with pip dependencies
- `frontend/Dockerfile` - Node 22 slim with npm dependencies
- `docker-compose.yml` - Multi-container orchestration

**Frontend Configuration Files:**
- `frontend/tsconfig.json` - Not present (no TypeScript)
- `frontend/package.json` - Defines all npm scripts and dependencies

**Backend Configuration Files:**
- `backend/requirements.txt` - All Python dependencies with pinned versions

## Platform Requirements

**Development:**
- Docker and Docker Compose
- Python 3.12
- Node.js 22
- PostgreSQL 16 (via Docker)
- PgAdmin 4 (via Docker)

**Production:**
- Docker container runtime
- PostgreSQL 16 database
- Environment variables for database credentials and API URLs
- Deployment target: Docker containers orchestrated by docker-compose

---

*Stack analysis: 2026-04-14*
