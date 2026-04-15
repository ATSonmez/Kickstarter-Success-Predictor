# Features Research — Kickstarter Success Predictor

**Project:** Kickstarter Success Predictor
**Mode:** Ecosystem / Features dimension
**Confidence:** MEDIUM — web/search access unavailable; findings draw on SHAP library docs, XAI UX research literature, ML web app design patterns, and dataviz best practices from training data (cutoff August 2025).

---

## Key Findings

- **Plain-English SHAP sentences are the highest-value differentiator.** Every other ML demo app either omits explanations entirely or shows raw force/waterfall plots. Neither works for a casual backer. Templated sentences ("Your goal is high for Technology campaigns — this is your biggest risk factor") are the thing that makes this product feel like a tool rather than a demo.
- **The gauge vs. bar question has a clear answer:** Use a horizontal gradient bar (red → green), not a semicircle gauge. Gauges evoke speedometers; casual users misread them as real-time or temperature. The horizontal bar maps directly onto the "funding progress" mental model Kickstarter backers already know.
- **Model comparison is portfolio value, not user value** — and must be designed accordingly. Lead with a threshold slider + plain-language confusion matrix, not ROC curves. ROC is academic unless the cells are labeled in plain English and the user can see the tradeoff move as they drag.
- **Three anti-features will kill trust faster than missing features:** raw waterfall plots as primary output, showing all three model predictions simultaneously on the Predict page (destroys "which one should I trust?"), and spurious decimal precision on probabilities (73.4% sounds like it means more than it does).
- **Preprocessing consolidation is not a feature — it's a prerequisite.** The three-way duplication in the training scripts means inference preprocessing will silently diverge from training unless consolidated first. Wrong preprocessing = wrong predictions = no feature in the product works correctly.

---

## Feature Landscape

### Table Stakes (Users Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Prediction form with plain-English field labels | Entry point of the product; a confusing form destroys trust before a number is shown | LOW | Fields: category, country, funding goal (USD), duration (days), reward tier count, has video (boolean). Labels must be human, not feature-engineering names. |
| Single probability output, clearly worded | "Will this succeed?" is the user's question. A number answers it. | LOW | "73% chance of success" — NOT "probability: 0.73." No decimals. |
| Success/failure verdict badge | Users think in binary | LOW | "Likely to succeed" (green) / "At risk" (red) at clear thresholds. Avoid yellow. |
| Top-3 plain-English SHAP explanations | The "why" is half the product's value | MEDIUM | Templated sentences mapped from feature name + SHAP sign + magnitude. |
| Loading state | No spinner = users assume the app crashed | LOW | Spinner/skeleton during API call. |
| Input validation with helpful messages | Red borders teach nothing | LOW | "Goal must be a number greater than $0" beats a red border. |
| EDA dashboard (≥3 charts) | Backers want prior context | MEDIUM | Success rate by category, by country, by goal range. Via `/eda/stats`. |
| Prediction history list | Users re-evaluate campaigns over time | MEDIUM | Timestamp, name, probability, verdict. Anonymous session ID in localStorage. |
| Mobile-responsive layout | Kickstarter is mobile-heavy | LOW | Tailwind responsive utilities. |

### Differentiators

| Feature | Value Proposition | Complexity |
|---------|-------------------|------------|
| Plain-English SHAP sentence generation | Every competitor omits XAI or only shows it to practitioners | MEDIUM |
| Horizontal gradient probability bar | Intuitive, avoids speedometer misread | LOW |
| Confidence-language tier labels | "73%" is meaningless without calibration context; bucketed language changes user behavior | LOW |
| Collapsible SHAP feature bar chart | Serves curious users and portfolio reviewers without overwhelming casual backers | MEDIUM |
| Category context annotation | "Technology campaigns with goals over $50K succeed 28% of the time historically" | LOW |
| Model comparison with threshold slider | Demonstrates ML rigor; slider makes the abstract concrete | HIGH |
| Per-history-entry SHAP replay | Revisit past predictions with original explanation | MEDIUM (add `shap_values JSONB` column early) |

### Anti-Features (Deliberately NOT Building)

| Feature | Why Problematic | Alternative |
|---------|-----------------|-------------|
| Raw SHAP waterfall/force plot as primary output | Non-ML users see overlap arrows and feel confused; drops trust | Plain-English sentences first; chart behind toggle |
| Semicircle / speedometer gauge | Speedometer misread, low data-ink ratio | Horizontal gradient bar |
| All three model predictions simultaneously on Predict page | Forces "which one do I trust?" — destroys confidence in all three | One primary model (NN) default; comparison on its own page |
| Decimal-precision probabilities | Spurious precision undermines trust | Round to nearest whole percent |
| Confidence intervals shown to casual users | "73% ± 12%" converts a clear signal into anxiety | Confidence-language tier labels instead |
| User accounts / login | Out of scope; no portfolio payoff | Anonymous session ID in localStorage |
| Kickstarter API scraping / autofill | ToS violation, fragile, out of scope | Manual form, honest framing |
| Real-time probability update as user types | Flickering, debounce complexity | Single submit button |
| "Similar campaigns" recommendation engine | Weeks of scope for marginal value | Link prediction → filtered EDA dashboard |
| Animated charts on EDA dashboard | Jank, distraction | Static or very short (150–200ms) entrance only |

---

## Feature Dependencies

```
[Consolidated Preprocessing Module]
    └── required by ─→ everything below

[Model Artifacts Saved (.pt, .pkl, .json)]
    └── required by ─→ [Model Serving Pipeline]

[Model Serving Pipeline — /predict endpoint]
    └── required by ─→ [Prediction Form Output]
    └── required by ─→ [SHAP Computation at Inference]
    └── required by ─→ [Prediction History Storage]

[SHAP Computation at Inference]
    └── required by ─→ [Plain-English SHAP Sentences]
    └── required by ─→ [Collapsible SHAP Bar Chart]
    └── required by ─→ [Per-History SHAP Replay]  (store values as JSONB)

[Prediction Form Output]
    └── enhances ─→ [Confidence-Language Tier Labels] (pure frontend)
    └── enhances ─→ [Gradient Probability Bar]       (pure frontend)
    └── enhances ─→ [Verdict Badge]                  (pure frontend)

[EDA Stats Endpoint (/eda/stats)]
    └── required by ─→ [EDA Dashboard Charts]
    └── required by ─→ [Category Context Annotation]

[All 3 Models Trained + Saved]
    └── required by ─→ [Model Comparison Page]
    └── required by ─→ [ModelMetric Table Populated]

[Prediction History Storage (Postgres)]
    └── required by ─→ [History List Page]
    └── required by ─→ [Per-History SHAP Replay]
```

**Critical ordering implications:**
1. Preprocessing consolidation must happen before any model is wired — silent divergence = wrong predictions with no error.
2. Model artifacts must be saved to `backend/models/` before serving can load them.
3. SHAP JSONB column should be added when the Prediction table is first created.
4. EDA stats endpoint is independent of the predict pipeline — parallelizable.
5. Model comparison is independent of the primary predict flow once all 3 models are saved.

---

## MVP Definition

### Launch With (v1) — predict-and-explain loop

- [ ] Consolidated preprocessing module
- [ ] Model saving (`kickstarterModel.py` → `.pt`, `scaler.pkl`, `feature_columns.json` in `backend/models/`)
- [ ] Model serving pipeline (FastAPI lifespan + `/predict` POST)
- [ ] Prediction form (6–8 fields, plain labels, validation)
- [ ] Probability output (gradient bar + rounded percent + verdict badge)
- [ ] Top-3 plain-English SHAP sentences
- [ ] EDA dashboard (3 charts via `/eda/stats`)
- [ ] Prediction history (Postgres + table UI)
- [ ] `docker compose up` works on fresh clone

### Add After Core Loop Stable (v1.x)

- [ ] Collapsible SHAP bar chart
- [ ] Category context annotation
- [ ] Confidence-language tier labels
- [ ] Per-history SHAP replay
- [ ] Model comparison page

### Future (v2+)

- [ ] Live deployment
- [ ] Automated data refresh

---

## Feature Prioritization Matrix

| Feature | User Value | Cost | Priority |
|---------|------------|------|----------|
| Preprocessing consolidation | HIGH | MED | P1 — hard blocker |
| Model artifact saving | HIGH | LOW | P1 — hard blocker |
| Model serving pipeline | HIGH | MED | P1 — hard blocker |
| Prediction form | HIGH | LOW | P1 |
| Probability bar + verdict badge | HIGH | LOW | P1 |
| Plain-English SHAP sentences | HIGH | MED | P1 |
| EDA dashboard (3 charts) | MED | MED | P1 |
| Prediction history list | MED | LOW | P1 |
| Docker compose end-to-end | HIGH | LOW | P1 |
| Mobile responsive | MED | LOW | P1 |
| Confidence-language tiers | MED | LOW | P2 |
| Category context annotation | MED | LOW | P2 |
| Collapsible SHAP chart | MED | MED | P2 |
| Per-history SHAP replay | LOW user / MED portfolio | MED | P2 |
| Model comparison page | LOW user / HIGH portfolio | HIGH | P2 |

---

## UX Pattern Notes

### Probability: Horizontal Gradient Bar

- Maps to the "funding progress" mental model Kickstarter backers already use
- Avoids speedometer misread
- Show rounded percentage in large bold text above the bar
- Do not show decimals, animate the fill, or show a second "chance of failure" bar

### SHAP: Sentences First, Chart Behind a Toggle

Sentence template (server-side Python):

```python
magnitude_label = (
    "has little impact on"       if abs_shap < 0.05 else
    "slightly affects"           if abs_shap < 0.10 else
    "moderately affects"         if abs_shap < 0.20 else
    "strongly affects"           if abs_shap < 0.35 else
    "is the dominant factor in"
)
direction = "supporting success" if shap_value > 0 else "working against success"
sentence = f"{feature_display_name} {magnitude_label} your prediction, {direction}."
```

Contextual enrichment for the top-1 feature only:
> "Your funding goal ($85,000) is high for Technology campaigns — this is your biggest risk factor."

### Model Comparison: Threshold Slider as Entry Point

1. Model selector tabs (NN / XGBoost / LogReg)
2. Threshold slider (default 0.5)
3. Confusion matrix with plain-English cell labels, live-updating
4. AUC summary with one-sentence explanation
5. ROC curve (secondary, labeled "technical detail")

Do not lead with accuracy — Kickstarter data is imbalanced.

---

## Sources

- SHAP library documentation (shap.readthedocs.io) — visualization audience assumptions. HIGH.
- Molnar, *Interpretable Machine Learning* (2nd ed.) — waterfall/force plot design intent. HIGH.
- Poursabzi-Sangdeh et al. (2021), *Manipulating and Measuring Model Interpretability* (SIGCHI). MEDIUM.
- Tufte / Few dataviz criticism of gauge charts. MEDIUM.
- `.planning/PROJECT.md` and `.planning/codebase/STRUCTURE.md`.

*Researched: 2026-04-14*
