"""Tests for backend/services/preprocessing.py — FND-01, FND-02, FND-03, FND-04 (partial)."""
from pathlib import Path

import numpy as np
import pytest


def test_preprocessor_importable():
    """FND-01: KickstarterPreprocessor importable from backend.services.preprocessing."""
    from backend.services.preprocessing import KickstarterPreprocessor
    assert KickstarterPreprocessor is not None


def test_feature_constants_exist():
    """FND-03: LAUNCH_TIME_FEATURES, POST_CAMPAIGN_FEATURES, AMBIGUOUS_DROP documented."""
    from backend.services import preprocessing as pp
    assert isinstance(pp.LAUNCH_TIME_FEATURES, list) and len(pp.LAUNCH_TIME_FEATURES) > 0
    assert isinstance(pp.POST_CAMPAIGN_FEATURES, list) and len(pp.POST_CAMPAIGN_FEATURES) > 0
    assert "backers_count" in pp.POST_CAMPAIGN_FEATURES
    assert "pledged" in pp.POST_CAMPAIGN_FEATURES
    assert "usd_pledged" in pp.POST_CAMPAIGN_FEATURES
    assert "static_usd_rate" in pp.AMBIGUOUS_DROP
    assert "static_usd_rate" not in pp.LAUNCH_TIME_FEATURES


def test_scaler_excludes_leakage(sample_raw_df):
    """FND-02: scaler fitted on exactly 5 CONTINUOUS_COLS, not on leakage columns."""
    from backend.services.preprocessing import KickstarterPreprocessor, CONTINUOUS_COLS
    pp = KickstarterPreprocessor()
    X, y = pp.fit_transform(sample_raw_df)
    assert pp.scaler.mean_.shape == (5,), (
        f"Scaler mean_ has shape {pp.scaler.mean_.shape}; expected (5,) — "
        "leakage columns may have been included in fit."
    )
    assert len(CONTINUOUS_COLS) == 5


def test_transform_single_zero_fills(sample_raw_df):
    """FND-01 contract: transform_single returns (1, N) matching feature_columns length."""
    from backend.services.preprocessing import KickstarterPreprocessor
    pp = KickstarterPreprocessor()
    pp.fit_transform(sample_raw_df)
    out = pp.transform_single({
        "goal": 5000,
        "name_len": 5,
        "blurb_len": 10,
        "create_to_launch": 30,
        "launch_to_deadline": 30,
        "category": "art",
        "country": "US",
    })
    assert out.shape == (1, len(pp.feature_columns))
    assert out.dtype == np.float32


def test_save_creates_artifacts(sample_raw_df, tmp_models_dir):
    """FND-04 (partial): save() writes scaler.pkl and feature_columns.pkl."""
    from backend.services.preprocessing import KickstarterPreprocessor
    pp = KickstarterPreprocessor()
    pp.fit_transform(sample_raw_df)
    pp.save(tmp_models_dir)
    assert (tmp_models_dir / "scaler.pkl").exists()
    assert (tmp_models_dir / "feature_columns.pkl").exists()


def test_load_roundtrip(sample_raw_df, tmp_models_dir):
    """FND-01 contract: load() restores a functional preprocessor."""
    from backend.services.preprocessing import KickstarterPreprocessor
    pp = KickstarterPreprocessor()
    pp.fit_transform(sample_raw_df)
    pp.save(tmp_models_dir)
    loaded = KickstarterPreprocessor.load(tmp_models_dir)
    assert loaded.feature_columns == pp.feature_columns
    assert np.allclose(loaded.scaler.mean_, pp.scaler.mean_)
