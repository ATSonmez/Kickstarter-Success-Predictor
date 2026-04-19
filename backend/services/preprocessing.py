"""Shared preprocessing for Kickstarter Success Predictor.

Single source of truth for feature transforms. Imported by training scripts
(kickstarterModel.py et al.) and by FastAPI inference path (Phase 2 lifespan).

Leakage fix (FND-02): StandardScaler is fitted AFTER POST_CAMPAIGN_FEATURES
and AMBIGUOUS_DROP columns are removed from the DataFrame. The current bug
in kickstarterModel.py fits the scaler on 9 columns including backers_count,
pledged, usd_pledged — this module fits on the 5 CONTINUOUS_COLS only.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Feature category documentation (FND-03) ──────────────────────────────
# Columns the user CAN know before launching a campaign.
LAUNCH_TIME_FEATURES: list[str] = [
    "goal", "name_len", "blurb_len", "create_to_launch", "launch_to_deadline",
    "country", "category", "deadline_yr", "launched_at_yr",
]

# Columns knowable ONLY after a campaign ends — never used at inference.
POST_CAMPAIGN_FEATURES: list[str] = [
    "backers_count", "pledged", "usd_pledged", "spotlight",
]

# Dropped entirely (Claude's Discretion decision, locked in CONTEXT.md):
# static_usd_rate may be a post-campaign exchange rate and cannot be
# supplied by the user at form time. If the model ever needs it,
# hardcode 1.0 in transform_single and document here.
AMBIGUOUS_DROP: list[str] = ["static_usd_rate"]

# Exactly 5 continuous columns that the scaler fits on (FND-02 invariant).
CONTINUOUS_COLS: list[str] = [
    "goal", "name_len", "blurb_len", "create_to_launch", "launch_to_deadline",
]

METADATA_COLS: list[str] = [
    "Unnamed: 0", "id", "photo", "name", "blurb", "slug",
    "currency", "currency_symbol", "currency_trailing_code",
    "state_changed_at", "created_at", "creator", "location",
    "profile", "urls", "source_url", "friends", "is_starred",
    "is_backing", "permissions",
    "deadline_weekday", "state_changed_at_weekday", "created_at_weekday",
    "launched_at_weekday", "deadline_day", "deadline_hr",
    "state_changed_at_month", "state_changed_at_day", "state_changed_at_yr",
    "state_changed_at_hr", "created_at_month", "created_at_day",
    "created_at_yr", "created_at_hr", "launched_at_day", "launched_at_hr",
    "launched_at_month", "deadline_month",
    "launch_to_state_change",
    "deadline", "launched_at", "name_len_clean", "blurb_len_clean",
]


class KickstarterPreprocessor:
    """Stateful preprocessor owning scaler and feature column list."""

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Training path. Fits the scaler AFTER leakage columns are dropped.

        Returns (X, y) as float32 numpy arrays.
        """
        df = df.copy()
        df.drop(columns=METADATA_COLS, errors="ignore", inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Target
        df["succeeded"] = (df["state"] == "successful").astype(int)
        df.drop(columns="state", inplace=True)

        # Timedelta → days
        df["create_to_launch"] = pd.to_timedelta(df["create_to_launch"]).dt.days
        df["launch_to_deadline"] = pd.to_timedelta(df["launch_to_deadline"]).dt.days

        # One-hot encode categoricals
        df = pd.get_dummies(
            df, columns=["country", "category", "deadline_yr", "launched_at_yr"]
        )
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        # CRITICAL (FND-02): drop leakage + ambiguous BEFORE fitting scaler.
        leakage_and_ambiguous = POST_CAMPAIGN_FEATURES + AMBIGUOUS_DROP
        df.drop(
            columns=[c for c in leakage_and_ambiguous if c in df.columns],
            inplace=True,
        )

        # Fit scaler ONLY on clean continuous columns (should be exactly 5).
        present_continuous = [c for c in CONTINUOUS_COLS if c in df.columns]
        df[present_continuous] = self.scaler.fit_transform(df[present_continuous])

        # Capture post-encoding column list (excludes target).
        self.feature_columns = [c for c in df.columns if c != "succeeded"]

        y = df["succeeded"].values.astype(np.float32)
        X = df[self.feature_columns].values.astype(np.float32)
        return X, y

    def transform_single(self, raw: dict) -> np.ndarray:
        """Inference path. Zero-fills columns not present, scales continuous
        fields via the FITTED scaler. Never calls pd.get_dummies on one row.
        """
        row: dict[str, float] = {col: 0 for col in self.feature_columns}

        cont_vals = np.array([[
            raw.get("goal", 0),
            raw.get("name_len", 0),
            raw.get("blurb_len", 0),
            raw.get("create_to_launch", raw.get("prep_days", 0)),
            raw.get("launch_to_deadline", raw.get("duration_days", 0)),
        ]], dtype=np.float32)
        scaled = self.scaler.transform(cont_vals)[0]
        for i, col in enumerate(CONTINUOUS_COLS):
            if col in row:
                row[col] = float(scaled[i])

        if raw.get("country"):
            col = f"country_{raw['country']}"
            if col in row:
                row[col] = 1
        if raw.get("category"):
            col = f"category_{raw['category']}"
            if col in row:
                row[col] = 1

        return np.array([list(row.values())], dtype=np.float32)

    def save(self, models_dir: Path | str) -> None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, models_dir / "scaler.pkl")
        joblib.dump(self.feature_columns, models_dir / "feature_columns.pkl")

    @classmethod
    def load(cls, models_dir: Path | str) -> "KickstarterPreprocessor":
        models_dir = Path(models_dir)
        pp = cls()
        pp.scaler = joblib.load(models_dir / "scaler.pkl")
        pp.feature_columns = joblib.load(models_dir / "feature_columns.pkl")
        return pp
