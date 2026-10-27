"""Inference, Prediction Pipeline, and Phishing Heuristic Audit.

Strictly honors:
1. Model probability safety (Requirement 10): Never displays uncalibrated scores as percentages.
2. Spam vs Phishing scientific distinction (Requirement 9): Explicitly acknowledges
   the dataset ground truth (Ham vs Spam) while providing transparent heuristic smishing analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.preprocessor import clean_text, load_vectorizer
from src.train_models import DEFAULT_MODELS_DIR, load_all_models

# Recognized URL shorteners and high-risk domains frequently leveraged in smishing
SUSPICIOUS_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "ow.ly", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc", "goo.gl"
}

# Smishing urgency and credential phishing keyword clusters
URGENCY_KEYWORDS = [
    r"\burgent\b", r"\bimmediately\b", r"\baction required\b", r"\bsuspended\b",
    r"\blocked\b", r"\bexpire[s|d]?\b", r"\b24\s*hours?\b", r"\bfinal notice\b",
    r"\bunauthorized\b", r"\bwarning\b", r"\brestricted\b"
]

CREDENTIAL_KEYWORDS = [
    r"\bverify your\b", r"\bconfirm your\b", r"\bpassword\b", r"\bpin\b",
    r"\botp\b", r"\bcvv\b", r"\bbank account\b", r"\blogin to\b", r"\bsecurity alert\b",
    r"\bupdate details\b", r"\bkyc\b"
]

FINANCIAL_LURE_KEYWORDS = [
    r"\bwon\b", r"\bprize\b", r"\bclaim\b", r"\blottery\b", r"\bfree gift\b",
    r"\bcongratulations\b", r"\bcash reward\b", r"\brefund\b", r"\binheritance\b"
]


def audit_smishing_indicators(raw_text: str) -> Dict[str, Any]:
    """Inspect text for heuristic smishing (SMS phishing) threat patterns.

    DISCLAIMER (Requirement 9):
    This rule-based analysis is distinct from the machine learning model.
    The ML model is trained on the SMS Spam Collection to identify Spam vs Ham.
    This heuristic audit inspects targeted social-engineering hallmarks commonly
    seen in Smishing (SMS Phishing) attacks.

    Parameters
    ----------
    raw_text : str
        Original uncleaned SMS text.

    Returns
    -------
    Dict[str, Any]
        Dictionary with detected indicators and smishing risk score.
    """
    text_lower = raw_text.lower()
    indicators: List[str] = []

    # 1. URL Shortener & Link Detection
    found_urls = re.findall(r"(https?://\S+|www\.\S+)", raw_text, re.IGNORECASE)
    has_shortener = False
    for url in found_urls:
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0].lower()
        if any(shortener in domain for shortener in SUSPICIOUS_SHORTENERS):
            has_shortener = True
            indicators.append(f"Suspicious URL shortener detected: '{domain}' (masks real destination)")

    if found_urls and not has_shortener:
        indicators.append(f"Contains web link ({len(found_urls)} link found)")

    # 2. Urgency and Coercion Keywords
    urgency_matches = [kw for kw in URGENCY_KEYWORDS if re.search(kw, text_lower)]
    if urgency_matches:
        indicators.append("Urgency/Intimidation language detected (common in social-engineering attacks)")

    # 3. Credential/Personal Data Harvesting
    credential_matches = [kw for kw in CREDENTIAL_KEYWORDS if re.search(kw, text_lower)]
    if credential_matches:
        indicators.append("Credential/Account verification prompt detected (high phishing risk)")

    # 4. Financial/Lottery Lure
    lure_matches = [kw for kw in FINANCIAL_LURE_KEYWORDS if re.search(kw, text_lower)]
    if lure_matches:
        indicators.append("Unsolicited prize or financial reward lure detected")

    # 5. Currency Symbols Combined With Numbers
    if re.search(r"[£$€₹]\s*\d+", raw_text) or re.search(r"\d+\s*(pounds|dollars|rupees|euros)", text_lower):
        indicators.append("Monetary amount claim highlighted")

    # Risk assessment tier based on heuristics
    indicator_count = len(indicators)
    if indicator_count >= 3 or (has_shortener and credential_matches):
        smishing_risk = "HIGH"
        risk_color = "#ef4444"
    elif indicator_count >= 1:
        smishing_risk = "MODERATE"
        risk_color = "#f59e0b"
    else:
        smishing_risk = "LOW"
        risk_color = "#10b981"

    return {
        "smishing_risk": smishing_risk,
        "risk_color": risk_color,
        "indicator_count": indicator_count,
        "detected_indicators": indicators,
        "url_count": len(found_urls),
        "disclaimer": (
            "Heuristic Indicator: Phishing detection is rule-based and educational. "
            "The ML classifier strictly evaluates Spam vs Ham."
        ),
    }


def predict_message(
    message: str,
    model_name: str = "Logistic Regression",
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    vectorizer: Optional[Any] = None,
    models: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end inference on a single message with probability safety.

    Parameters
    ----------
    message : str
        Input text.
    model_name : str
        Algorithm to utilize.
    models_dir : Path | str
        Path where artifacts reside.
    vectorizer : Optional
        Pre-loaded TF-IDF vectorizer (avoids re-reading disk).
    models : Optional
        Pre-loaded models dictionary (avoids re-reading disk).

    Returns
    -------
    Dict[str, Any]
        Structured prediction output with confidence and calibration details.
    """
    if not message or not message.strip():
        return {
            "error": "Empty message provided. Please enter text to classify.",
            "prediction": "Unknown",
        }

    # 1. Load artifacts if not passed in memory
    if vectorizer is None:
        vectorizer = load_vectorizer(Path(models_dir) / "tfidf_vectorizer.joblib")
    if models is None:
        models = load_all_models(models_dir)

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not available. Choose from: {list(models.keys())}")

    model = models[model_name]

    # 2. Text preprocessing and vectorization
    normalized_text = clean_text(message)
    X_vec = vectorizer.transform([normalized_text])

    # 3. Model Prediction
    prediction_num = int(model.predict(X_vec)[0])
    prediction_label = "Spam" if prediction_num == 1 else "Ham (Legitimate)"

    # 4. Strict Probability / Score Calibration Handling (Requirement 10)
    has_proba = hasattr(model, "predict_proba")
    confidence_pct: Optional[float] = None
    calibrated_probabilities: Optional[Dict[str, float]] = None
    decision_score: Optional[float] = None
    score_interpretation: str = ""

    if has_proba:
        probs = model.predict_proba(X_vec)[0]
        prob_ham = float(probs[0])
        prob_spam = float(probs[1])
        calibrated_probabilities = {
            "Ham": round(prob_ham * 100, 2),
            "Spam": round(prob_spam * 100, 2),
        }
        confidence_pct = round((prob_spam if prediction_num == 1 else prob_ham) * 100, 2)
        score_interpretation = "Statistically calibrated probability percentage."
    elif hasattr(model, "decision_function"):
        # Model does NOT support probabilities (e.g., raw LinearSVC).
        # NEVER claim decision margin is a percentage!
        raw_dist = float(model.decision_function(X_vec)[0])
        decision_score = round(raw_dist, 4)
        confidence_pct = None  # None ensures UI does not display false percentage
        score_interpretation = (
            f"Uncalibrated hyperplane margin distance: {decision_score}. "
            "Warning: This is an uncalibrated geometric distance, NOT a probability percentage."
        )
    else:
        score_interpretation = "Model does not output probabilities or continuous decision scores."

    # 5. Extract top influential TF-IDF terms in the message
    feature_names = np.array(vectorizer.get_feature_names_out())
    row_vec = X_vec.toarray()[0]
    non_zero_indices = np.where(row_vec > 0)[0]
    term_weights = []
    for idx in non_zero_indices:
        term_weights.append({
            "term": feature_names[idx],
            "tfidf_weight": round(float(row_vec[idx]), 4),
        })
    term_weights.sort(key=lambda item: item["tfidf_weight"], reverse=True)

    # 6. Heuristic Smishing Audit (Requirement 9)
    smishing_audit = audit_smishing_indicators(message)

    return {
        "raw_message": message,
        "cleaned_message": normalized_text,
        "model_used": model_name,
        "prediction_num": prediction_num,
        "prediction_label": prediction_label,
        "is_spam": bool(prediction_num == 1),
        "has_probability": has_proba,
        "confidence_pct": confidence_pct,
        "probabilities": calibrated_probabilities,
        "decision_score": decision_score,
        "score_interpretation": score_interpretation,
        "top_features": term_weights[:8],
        "smishing_audit": smishing_audit,
    }
