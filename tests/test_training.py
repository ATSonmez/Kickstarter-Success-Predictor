"""Tests for training script refactor — FND-04 (full), FND-05, FND-06."""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "backend" / "models"
TRAINING_SCRIPTS = [
    REPO / "kickstarterModel.py",
    REPO / "kickstarterModel_testing.py",
    REPO / "hyperparameter_search.py",
]
REQUIRED_ARTIFACTS = [
    "kickstarter_nn.pt",
    "scaler.pkl",
    "feature_columns.pkl",
    "eda_stats.json",
    "background.pt",
    "model_metadata.json",
]


@pytest.mark.xfail(reason="FND-04 integration — requires a full training run; verified manually via checkpoint in Plan 03")
def test_all_six_artifacts_exist_after_training():
    """FND-04: all 6 artifacts land in backend/models/ after training."""
    for name in REQUIRED_ARTIFACTS:
        assert (MODELS_DIR / name).exists(), f"Missing artifact: {name}"


def test_no_standardscaler_import_in_root_scripts():
    """FND-05: no duplicated preprocessing — scripts import from shared module instead."""
    for script in TRAINING_SCRIPTS:
        text = script.read_text(encoding="utf-8")
        assert "StandardScaler" not in text, (
            f"{script.name} still imports/uses StandardScaler directly; "
            "should use KickstarterPreprocessor instead."
        )
        assert "from services.preprocessing import KickstarterPreprocessor" in text, (
            f"{script.name} does not import the shared preprocessor."
        )


def test_json_artifacts_gitignored():
    """FND-06: backend/models/*.json is gitignored."""
    probe = MODELS_DIR / "eda_stats.json"
    result = subprocess.run(
        ["git", "check-ignore", "-v", str(probe)],
        capture_output=True, text=True, cwd=REPO,
    )
    assert result.returncode == 0, (
        f"backend/models/eda_stats.json is NOT gitignored. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )


def test_gitkeep_exists():
    """FND-06: backend/models/.gitkeep tracked so empty dir survives clone."""
    assert (MODELS_DIR / ".gitkeep").exists()
