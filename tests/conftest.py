"""Shared pytest fixtures for Phase 1 tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking the Kickstarter CSV schema.

    Contains both launch-time and post-campaign columns so tests can
    verify leakage columns are dropped before scaler fit.
    """
    n = 20
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Unnamed: 0": range(n),
        "id": range(1000, 1000 + n),
        "photo": ["p"] * n,
        "name": [f"proj{i}" for i in range(n)],
        "blurb": ["b"] * n,
        "slug": ["s"] * n,
        "goal": rng.uniform(500, 100_000, n).astype(float),
        "pledged": rng.uniform(0, 50_000, n).astype(float),
        "state": (["successful"] * (n // 2)) + (["failed"] * (n - n // 2)),
        "country": rng.choice(["US", "GB", "CA"], n),
        "currency": ["USD"] * n,
        "currency_symbol": ["$"] * n,
        "currency_trailing_code": [True] * n,
        "deadline": pd.date_range("2020-01-01", periods=n, freq="D"),
        "state_changed_at": pd.date_range("2020-01-01", periods=n, freq="D"),
        "created_at": pd.date_range("2019-06-01", periods=n, freq="D"),
        "launched_at": pd.date_range("2019-12-01", periods=n, freq="D"),
        "staff_pick": rng.choice([True, False], n),
        "backers_count": rng.integers(0, 500, n),
        "static_usd_rate": [1.0] * n,
        "usd_pledged": rng.uniform(0, 50_000, n).astype(float),
        "creator": ["c"] * n,
        "location": ["l"] * n,
        "profile": ["p"] * n,
        "urls": ["u"] * n,
        "source_url": ["s"] * n,
        "friends": [None] * n,
        "is_starred": [None] * n,
        "is_backing": [None] * n,
        "permissions": [None] * n,
        "spotlight": rng.choice([True, False], n),
        "category": rng.choice(["art", "music", "tech"], n),
        "name_len": rng.integers(1, 20, n),
        "blurb_len": rng.integers(1, 30, n),
        "name_len_clean": rng.integers(1, 20, n),
        "blurb_len_clean": rng.integers(1, 30, n),
        "deadline_weekday": rng.integers(0, 7, n),
        "state_changed_at_weekday": rng.integers(0, 7, n),
        "created_at_weekday": rng.integers(0, 7, n),
        "launched_at_weekday": rng.integers(0, 7, n),
        "deadline_month": rng.integers(1, 13, n),
        "deadline_day": rng.integers(1, 29, n),
        "deadline_yr": rng.integers(2015, 2021, n),
        "deadline_hr": rng.integers(0, 24, n),
        "state_changed_at_month": rng.integers(1, 13, n),
        "state_changed_at_day": rng.integers(1, 29, n),
        "state_changed_at_yr": rng.integers(2015, 2021, n),
        "state_changed_at_hr": rng.integers(0, 24, n),
        "created_at_month": rng.integers(1, 13, n),
        "created_at_day": rng.integers(1, 29, n),
        "created_at_yr": rng.integers(2015, 2021, n),
        "created_at_hr": rng.integers(0, 24, n),
        "launched_at_month": rng.integers(1, 13, n),
        "launched_at_day": rng.integers(1, 29, n),
        "launched_at_yr": rng.integers(2015, 2021, n),
        "launched_at_hr": rng.integers(0, 24, n),
        "create_to_launch": pd.to_timedelta(rng.integers(1, 60, n), unit="D"),
        "launch_to_deadline": pd.to_timedelta(rng.integers(10, 60, n), unit="D"),
        "launch_to_state_change": pd.to_timedelta(rng.integers(10, 60, n), unit="D"),
    })


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d
