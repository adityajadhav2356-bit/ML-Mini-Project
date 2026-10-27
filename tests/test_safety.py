"""Unit Tests for Probability Calibration Safety and Phishing Taxonomy."""

import numpy as np
import pytest
from sklearn.svm import LinearSVC

from src.predictor import audit_smishing_indicators, predict_message
from src.preprocessor import load_vectorizer
from src.train_models import DEFAULT_MODELS_DIR


def test_uncalibrated_model_never_returns_confidence_percentage():
    """Verify Requirement 10: Non-probabilistic models NEVER output confidence percentages."""
    vectorizer = load_vectorizer(DEFAULT_MODELS_DIR / "tfidf_vectorizer.joblib")

    # Construct an uncalibrated LinearSVC that only has decision_function
    raw_svm = LinearSVC(C=1.0, random_state=42)
    # Fit on synthetic data
    X_toy = vectorizer.transform(["free cash win prize", "hello friend see you tomorrow"])
    y_toy = np.array([1, 0])
    raw_svm.fit(X_toy, y_toy)

    assert not hasattr(raw_svm, "predict_proba"), "Raw LinearSVC should not have predict_proba"
    assert hasattr(raw_svm, "decision_function"), "Raw LinearSVC has decision_function"

    mock_models = {"Raw LinearSVC": raw_svm}

    result = predict_message(
        "Free cash prize win reward",
        model_name="Raw LinearSVC",
        vectorizer=vectorizer,
        models=mock_models,
    )

    # CRITICAL VERIFICATIONS FOR REQUIREMENT 10:
    assert result["has_probability"] is False
    assert result["confidence_pct"] is None, "Uncalibrated score must NOT be returned as percentage!"
    assert result["decision_score"] is not None
    assert isinstance(result["decision_score"], float)
    assert "Uncalibrated" in result["score_interpretation"]
    assert "NOT a probability percentage" in result["score_interpretation"]


def test_smishing_heuristic_audit_taxonomy():
    """Verify Requirement 9: Phishing risk audit works with explicit scientific disclaimer."""
    # Smishing attack sample with suspicious shortener, urgency, and credential lure
    smishing_msg = "URGENT: Your bank account is locked. Verify immediately at http://bit.ly/bank-security"
    audit = audit_smishing_indicators(smishing_msg)

    assert audit["smishing_risk"] == "HIGH"
    assert audit["indicator_count"] >= 2
    assert any("URL shortener" in ind for ind in audit["detected_indicators"])
    assert any("Urgency" in ind for ind in audit["detected_indicators"])

    # Verify transparent disclaimer
    assert "disclaimer" in audit
    assert "Heuristic Indicator" in audit["disclaimer"]
    assert "The ML classifier strictly evaluates Spam vs Ham." in audit["disclaimer"]


def test_benign_message_has_low_smishing_risk():
    """Verify normal benign chat text does not trigger smishing indicators."""
    normal_msg = "Let's meet at Starbucks tomorrow at 10am for coffee and work on the slides."
    audit = audit_smishing_indicators(normal_msg)

    assert audit["smishing_risk"] == "LOW"
    assert audit["indicator_count"] == 0
    assert len(audit["detected_indicators"]) == 0
