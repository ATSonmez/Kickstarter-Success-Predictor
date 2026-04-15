# Coding Conventions

**Analysis Date:** 2026-04-14

## Naming Patterns

**Files:**
- JavaScript/React files: `camelCase.jsx` (e.g., `PredictPage.jsx`, `DashboardPage.jsx`)
- Configuration files: `camelCase.js` (e.g., `eslint.config.js`, `vite.config.js`)
- Python files: `snake_case.py` (e.g., `database.py`, `db_models.py`, `kickstarterModel.py`)
- Test files: `snake_case_testing.py` (e.g., `kickstarterModel_testing.py`)

**Functions:**
- JavaScript/React: `camelCase` for functions and arrow functions
- Python: `snake_case` for functions (e.g., `get_db()` in `database.py`)
- Class methods follow the same naming as their parent language

**Variables:**
- JavaScript: `camelCase` for constants and variables (e.g., `train_loader`, `best_val_loss` in Python; `baseURL`, `isActive` in JS)
- Python: `snake_case` for all variables and parameters (e.g., `X_train`, `y_test`, `num_features`)
- Constants: `UPPERCASE` (e.g., `NUM_EPOCHS`, `PATIENCE`)

**Types/Classes:**
- React components: `PascalCase` (e.g., `PredictPage`, `DashboardPage`, `KickstarterNet`)
- Database models: `PascalCase` (e.g., `Prediction`, `ModelMetric`)
- Python neural network classes: `PascalCase` (e.g., `KickstarterNet`)

## Code Style

**Formatting:**
- No explicit formatter configured in frontend (ESLint handles basic formatting)
- Python files use standard formatting with sections marked by comment separators (e.g., `# ── 1. Load & Clean Data ─────────────────────────────────────────────────────`)
- Line length: No strict limit enforced but Python models aim for readable, logical sections

**Linting:**
- Frontend: ESLint 9.39.4 with React hooks and React Refresh plugins
- Python: No linter configured
- Key ESLint rule: `no-unused-vars` with pattern `^[A-Z_]` to allow uppercase constants

**Key settings from `eslint.config.js`:**
- ECMAScript version: 2020 (latest)
- JSX support enabled
- Module source type: `module`
- Browser globals enabled

## Import Organization

**Order (Frontend):**
1. React and React Router imports (e.g., `import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'`)
2. Page/component imports (e.g., `import PredictPage from './pages/PredictPage'`)
3. Configuration/API imports (e.g., `import api from './api'`)
4. Styling imports (implicit via Tailwind CSS)

**Frontend example from `App.jsx`:**
```javascript
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import PredictPage from './pages/PredictPage'
import DashboardPage from './pages/DashboardPage'
```

**Path Aliases:**
- No path aliases configured in frontend build
- Relative paths used throughout (e.g., `./pages/`, `./api.js`)

**Order (Python):**
1. Standard library imports (e.g., `import time`, `import os`)
2. Third-party imports (e.g., `import pandas as pd`, `import torch`)
3. Local imports (e.g., `from database import Base`)

**Python example from `kickstarterModel.py`:**
```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
```

## Error Handling

**Patterns:**
- Python backend: Try/finally pattern for database sessions (e.g., in `database.py` `get_db()` function)
```python
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```
- Python models: Implicit error handling via exceptions (no explicit try/catch blocks in training loops)
- Frontend: No explicit error handling visible in page components (minimal implementation)
- Logging: Print statements used for progress tracking in Python models

## Logging

**Framework:** `print()` statements for Python, console implicitly available for JavaScript

**Patterns:**
- Python models log epoch progress at intervals:
```python
if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], '
          f'Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f}, '
          f'Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}', flush=True)
```
- Status messages at key points (data loading, training completion)
- No structured logging framework configured

## Comments

**When to Comment:**
- Section headers for major code blocks (e.g., `# ── 1. Load & Clean Data ─────────────────────────────────────────────────────`)
- Inline comments for non-obvious operations (e.g., explaining data leakage columns)
- Docstrings: Not used in current codebase

**JSDoc/TSDoc:**
- Not used in frontend code
- No type annotations in JavaScript files

**Example from `kickstarterModel.py`:**
```python
# Drop data-leakage columns (unknowable before campaign ends)
leakage_cols = ['pledged', 'usd_pledged', 'backers_count', 'spotlight']
df.drop(columns=leakage_cols, inplace=True)
```

## Function Design

**Size:** 
- Python training functions: Large, monolithic (entire training loop in main section, 500+ lines)
- React components: Small, typically under 20 lines
- Utility functions: Minimal (only `get_db()` in backend)

**Parameters:** 
- Python models: Pass data structures as parameters (DataLoader, tensors)
- React components: Accept no parameters or use Tailwind CSS classes
- Functions prefer passing aggregated objects over individual parameters

**Return Values:** 
- Python training: Implicit returns (models saved via `.state_dict()`)
- React components: Return JSX elements
- Database functions: Yield database sessions (generator pattern)

## Module Design

**Exports:**
- JavaScript: Named exports for utility objects (e.g., `export default api` in `api.js`)
- React components: Default exports (e.g., `export default PredictPage`)
- Python: No explicit module-level exports; imports via function/class names

**Barrel Files:**
- Not used in frontend
- No index.js files for aggregating exports

**Example from `api.js`:**
```javascript
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
})

export default api
```

---

*Convention analysis: 2026-04-14*
