# Architecture

**Analysis Date:** 2026-04-14

## Pattern Overview

**Overall:** Full-stack web application with multi-tier separation between frontend, backend API, and database layers. Client-server architecture with containerized microservices.

**Key Characteristics:**
- FastAPI backend for REST API endpoints
- React frontend with React Router for client-side navigation
- PostgreSQL relational database for persistence
- Docker containerization for all services
- CORS-enabled API with environment-based configuration
- Model prediction pipeline (PyTorch/scikit-learn/XGBoost) integrated in backend

## Layers

**Presentation Layer (Frontend):**
- Purpose: User interface for model prediction, data visualization, and prediction history
- Location: `frontend/src/`
- Contains: React components, pages, routing, API client
- Depends on: Axios for HTTP communication with backend
- Used by: Web browser clients via React Router navigation

**API Layer (Backend):**
- Purpose: REST API endpoints serving model predictions, metrics, and historical data
- Location: `backend/main.py`
- Contains: FastAPI application with CORS middleware, route definitions
- Depends on: Database layer for persistence, model services (to be implemented)
- Used by: Frontend React application, external clients

**Data Access Layer:**
- Purpose: Database schema definition and connection management
- Location: `backend/database.py`, `backend/db_models.py`
- Contains: SQLAlchemy ORM models (Prediction, ModelMetric), database configuration
- Depends on: PostgreSQL via SQLAlchemy
- Used by: API layer for storing/retrieving predictions and model metrics

**Infrastructure Layer:**
- Purpose: Container orchestration and service coordination
- Location: `docker-compose.yml`, `backend/Dockerfile`
- Contains: Database service (PostgreSQL), backend service, frontend service, pgAdmin
- Depends on: Environment variables for configuration
- Used by: Development and production deployment

## Data Flow

**Prediction Request Flow:**

1. User fills form on `frontend/src/pages/PredictPage.jsx`
2. Form submission calls API client at `frontend/src/api.js` (Axios instance)
3. Request sent to `backend/main.py` endpoint (not yet implemented)
4. Backend processes features, calls model inference (not yet implemented)
5. Result stored in PostgreSQL via `backend/db_models.py` Prediction table
6. API returns prediction JSON to frontend
7. Frontend displays result to user

**Model Metrics Flow:**

1. Model training happens offline (root-level: `kickstarterModel.py`, `hyperparameter_search.py`)
2. Trained model metrics inserted into PostgreSQL ModelMetric table
3. PerformancePage queries metrics endpoint (not yet implemented)
4. Backend returns metrics from database
5. Frontend renders charts using Chart.js

**State Management:**

- No centralized state management (Redux, Context API) currently in place
- React component state for form input (planned in Phase 4)
- API responses cached at component level as needed
- Database is single source of truth for persistent state

## Key Abstractions

**API Client:**
- Purpose: Centralized HTTP communication configuration
- Examples: `frontend/src/api.js`
- Pattern: Axios instance with baseURL from environment variable, allows easy endpoint addition

**Database Models:**
- Purpose: ORM schema definitions for database tables
- Examples: `backend/db_models.py` (Prediction, ModelMetric classes)
- Pattern: SQLAlchemy declarative models with timestamp auto-generation

**Page Components:**
- Purpose: Top-level route components for each feature area
- Examples: `frontend/src/pages/PredictPage.jsx`, `frontend/src/pages/DashboardPage.jsx`, etc.
- Pattern: Simple functional components with placeholder UI (Phase 4 implementation pending)

## Entry Points

**Frontend Application:**
- Location: `frontend/src/main.jsx`
- Triggers: Browser page load (served by Vite dev server or build output)
- Responsibilities: React DOM bootstrap, mounting App component

**Frontend Router:**
- Location: `frontend/src/App.jsx`
- Triggers: Page navigation via NavLink components
- Responsibilities: Navigation bar rendering, route definition, layout wrapper

**Backend API:**
- Location: `backend/main.py`
- Triggers: HTTP requests from frontend or external clients
- Responsibilities: CORS configuration, health check endpoint, root endpoint, route mounting (incomplete)

**Database:**
- Location: PostgreSQL service in `docker-compose.yml`
- Triggers: SQLAlchemy queries from backend
- Responsibilities: Data persistence for predictions and model metrics

## Error Handling

**Strategy:** Layer-specific error handling with minimal implementation currently.

**Patterns:**
- Backend: FastAPI automatic 422 validation errors for request bodies (Pydantic)
- Backend: Database dependency injection with try/finally for session cleanup (`backend/database.py` get_db)
- Frontend: Axios interceptor structure prepared but not yet configured
- Frontend: No error boundaries or error UI components implemented yet

## Cross-Cutting Concerns

**Logging:** Not implemented. Should use Python logging in backend, console logging in frontend.

**Validation:** Pydantic models defined in backend for request validation (to be used in route handlers). Frontend form validation not yet implemented.

**Authentication:** Not implemented. API endpoints currently public with CORS allowing configured frontend URL.

**Environment Configuration:** Managed via .env files using python-dotenv (backend) and Vite import.meta.env (frontend). Key vars: DATABASE_URL, FRONTEND_URL, VITE_API_URL.

---

*Architecture analysis: 2026-04-14*
