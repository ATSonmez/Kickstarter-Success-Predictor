# Phase 1: Foundation - Context

**Gathered:** 2026-04-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Build the shared `backend/services/preprocessing.py` module, fix the scaler leakage bug, save all model artifacts (`.pt`, `.pkl`, `.json`), and refactor the three training scripts to import the shared module. No API endpoints, no frontend work. Phase complete when `python kickstarterModel.py` produces all required artifacts in `backend/models/`.

</domain>

<decisions>
## Implementation Decisions

### Training Script Refactor Depth
- **D-01:** Minimal refactor only — replace inline preprocessing code with the shared import. Matplotlib/seaborn EDA blocks, print statements, and exploratory plots stay in place. Scripts remain usable as standalone notebooks for future exploratory runs.

### File Locations
- **D-02:** Training scripts stay at project root (`kickstarterModel.py`, `kickstarterModel_testing.py`, `hyperparameter_search.py`). They import backend modules via `sys.path.insert(0, str(Path(__file__).parent / "backend"))`.
- **D-03:** The PyTorch `nn.Module` architecture class lives in `backend/models/nn_model.py` — importable by both the root training scripts (via sys.path) and the FastAPI lifespan loader. Not at root.

### Claude's Discretion
- **`static_usd_rate` handling:** Drop it from the feature set. Users cannot supply a currency exchange rate at form time, and it may represent a post-campaign value in the CSV snapshot. Default to dropping; if the model needs it, hardcode 1.0 in `transform_single()` and document the decision clearly in the preprocessing module.
- **EDA goal buckets:** Define 4 ranges that are meaningful to Kickstarter backers: `<$1,000`, `$1,000–$10,000`, `$10,000–$100,000`, `>$100,000`. Use these consistently in `eda_stats.json` and the Phase 3 dashboard.
- **Background sample size:** 100 rows (balanced sample, 50 success + 50 failure) saved as `background.pt` for SHAP `DeepExplainer`.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase specification
- `.planning/ROADMAP.md` §Phase 1 — Success criteria and requirement IDs
- `.planning/REQUIREMENTS.md` §FND-01 through FND-06 — Acceptance criteria

### Architecture and patterns
- `.planning/research/ARCHITECTURE.md` — Pattern 2: `KickstarterPreprocessor` class design (fit_transform / transform_single / save / load), build order, anti-patterns
- `.planning/research/PITFALLS.md` §Pitfalls 1–3, 7 — Scaler ordering bug, preprocessing duplication, artifact saving

### Existing code to refactor
- `kickstarterModel.py` — Primary training script; scaler bug is on the line `df[continuous_cols] = scaler.fit_transform(df[continuous_cols])` where `continuous_cols` includes `backers_count`, `pledged`, `usd_pledged` before those are dropped
- `kickstarterModel_testing.py` — Secondary training script; already diverges from primary
- `hyperparameter_search.py` — Hyperparameter tuning script; also has inline preprocessing

### Codebase maps
- `.planning/codebase/CONCERNS.md` — Priority debt items for Phase 1
- `.planning/codebase/STACK.md` — Dependency versions (torch 2.7.1, shap 0.47.2, sklearn 1.7.0, joblib)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `kickstarterModel.py` column drop lists — `columns_to_drop`, `continuous_cols`, `pd.get_dummies` call can be directly ported into `KickstarterPreprocessor.fit_transform()` with the leakage columns removed
- `kickstarterModel.py` model architecture (nn.Module) — Extract to `backend/models/nn_model.py` as-is; only the class definition moves, not training logic

### Established Patterns
- `backend/database.py`, `backend/db_models.py` — SQLAlchemy pattern already uses `SessionLocal`; no pattern changes needed for Phase 1
- `backend/main.py` — Bare FastAPI app with no lifespan; Phase 2 will add the lifespan hook

### Integration Points
- `backend/models/` — Empty target directory for all saved artifacts; `.gitkeep` should be added (or confirm it's already there)
- Training scripts → preprocessing: `sys.path.insert(0, str(Path(__file__).parent / "backend"))` then `from services.preprocessing import KickstarterPreprocessor`

</code_context>

<specifics>
## Specific Ideas

No specific visual or UX requirements — Phase 1 is pure backend infrastructure.

Key invariants to preserve:
- Scaler fitted ONLY after leakage columns (`backers_count`, `pledged`, `usd_pledged`) are dropped
- `feature_columns.pkl` saved after `pd.get_dummies` encoding so inference can zero-fill against the exact training column list
- `eda_stats.json` written by the training script (not computed at serve time)
- All artifacts land in `backend/models/` and that directory is gitignored except `.gitkeep`

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-04-18*
