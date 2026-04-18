# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-18
**Phase:** 01-foundation
**Areas discussed:** Training script cleanup depth

---

## Training Script Cleanup Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Minimal — import only | Replace inline preprocessing with shared import. Leave matplotlib/EDA blocks, print statements, exploratory plots intact. Scripts stay usable as standalone notebooks. | ✓ |
| Full cleanup | Strip matplotlib/seaborn EDA blocks, dead variables, and debug prints. Cleaner files, training-only focus. More effort. | |

**User's choice:** Minimal — import only  
**Notes:** User wants the exploratory/notebook character of the training scripts preserved.

---

## Script File Locations

| Option | Description | Selected |
|--------|-------------|----------|
| Leave at root | Training scripts at project root, import backend/ via sys.path.insert. Matches common ML project conventions. | ✓ |
| Move into backend/ | All Python in one place, but mixes offline training tools with the served API. | |

**User's choice:** Leave at root

---

## nn.Module Architecture Class Location

| Option | Description | Selected |
|--------|-------------|----------|
| backend/models/nn_model.py | Inside backend package, importable by both training (sys.path) and FastAPI lifespan loader. | ✓ |
| Root-level nn_model.py | Easier import from training scripts, but puts application code at root alongside data/notebooks. | |

**User's choice:** `backend/models/nn_model.py`

---

## Claude's Discretion

- **`static_usd_rate`:** Drop from feature set (user did not select this area for discussion; Claude decided: drop, users cannot supply exchange rates at form time)
- **EDA goal buckets:** `<$1K`, `$1K–$10K`, `$10K–$100K`, `>$100K` (4 ranges meaningful to Kickstarter backers)
- **Background sample size:** 100 rows for SHAP background

## Deferred Ideas

None
