"""Unit Tests for Model Artifacts and Inference Accuracy."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.predictor import predict_message
from src.train_models import DEFAULT_MODELS_DIR, load_all_models


def test_all_four_models_present():
    """Verify all 4 required algorithms are serialized and loadable."""
    models = load_all_models(DEFAULT_MODELS_DIR)
    expected_algorithms = [
        "Logistic Regression",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbors (KNN)",
        "AdaBoost",
    ]
    for algo in expected_algorithms:
        assert algo in models, f"Model {algo} not loaded"


def test_metrics_summary_content():
    """Verify metrics summary CSV contains all 4 algorithms and high scores."""
    csv_path = DEFAULT_MODELS_DIR / "metrics_summary.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    assert len(df) == 4
    for algo in ["Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "AdaBoost"]:
        assert algo in df["Algorithm"].values

    # Check accuracy threshold across all algorithms
    for _, row in df.iterrows():
        assert row["Accuracy"] >= 0.95, f"Algorithm {row['Algorithm']} accuracy below 95%"
        assert row["Precision (Spam)"] >= 0.90, f"Algorithm {row['Algorithm']} precision below 90%"


def test_known_ham_prediction():
    """Test standard legitimate message classification."""
    msg = "Hey, are we still meeting at the library at 4pm for the project discussion?"
    result = predict_message(msg, model_name="Support Vector Machine (SVM)")
    assert result["is_spam"] is False
    assert result["prediction_label"] == "Ham (Legitimate)"
    assert result["confidence_pct"] is not None
    assert result["confidence_pct"] > 80.0


def test_known_spam_prediction():
    """Test blatant spam message classification."""
    msg = "WINNER! You have won a guaranteed £1000 cash prize or iPad. Call 09061701461 to claim your award today!"
    result = predict_message(msg, model_name="Support Vector Machine (SVM)")
    assert result["is_spam"] is True
    assert result["prediction_label"] == "Spam"
    assert result["confidence_pct"] is not None
    assert result["confidence_pct"] > 80.0
