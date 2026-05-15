from pathlib import Path

import pytest

from ticket_classifier.model_service import load_classifier


def test_load_classifier_requires_model_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run train-ticket-classifier"):
        load_classifier(tmp_path / "missing.joblib")
