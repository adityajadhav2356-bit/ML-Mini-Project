"""Text Preprocessing and Feature Extraction Pipeline.

Ensures strict separation between training and test sets to guarantee
zero data leakage during TF-IDF vectorization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VECTORIZER_PATH = DEFAULT_MODELS_DIR / "tfidf_vectorizer.joblib"


def clean_text(text: str) -> str:
    """Normalize and clean a single raw SMS message string.

    Steps:
    1. Lowercase text.
    2. Replace URLs with 'httpaddr'.
    3. Replace email addresses with 'emailaddr'.
    4. Replace currency symbols (£, $, €, ¥, ₹) with 'moneysymb'.
    5. Replace telephone/numeric strings with 'numbr'.
    6. Strip punctuation and excessive whitespace.

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    str
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower().strip()

    # Normalize URLs
    text = re.sub(r"(https?://\S+|www\.\S+)", " httpaddr ", text)

    # Normalize Email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " emailaddr ", text)

    # Normalize currency symbols (£, $, €, ¥, etc.)
    text = re.sub(r"[£$€¥₹]", " moneysymb ", text)

    # Normalize phone numbers and digit sequences (>= 3 digits)
    text = re.sub(r"\b\d{3,}\b", " numbr ", text)

    # Remove non-alphanumeric characters, keeping word characters and whitespace
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 3000,
    save_vectorizer: bool = True,
    vectorizer_path: Path | str = VECTORIZER_PATH,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, TfidfVectorizer]:
    """Split dataset without data leakage and fit TF-IDF solely on training set.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'message' and 'label_num'.
    test_size : float
        Proportion of dataset reserved for testing (default 0.2).
    random_state : int
        Seed for reproducibility.
    max_features : int
        Maximum number of TF-IDF vocabulary features.
    save_vectorizer : bool
        Whether to serialize the fitted vectorizer with joblib.
    vectorizer_path : Path | str
        Destination path for serialized vectorizer.

    Returns
    -------
    Tuple
        (X_train_vec, X_test_vec, y_train, y_test, vectorizer)
    """
    # Apply text cleaning
    clean_messages = df["message"].apply(clean_text)
    labels = df["label_num"]

    # CRITICAL: Train/test split performed BEFORE fitting TF-IDF to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        clean_messages,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),       # Capture informative unigrams and bigrams
        stop_words="english",     # Standard English stop words
        sublinear_tf=True,        # Sublinear scaling: 1 + log(tf)
        min_df=2                  # Ignore terms that appear in fewer than 2 documents
    )

    # FIT ONLY ON TRAINING DATA
    X_train_vec = vectorizer.fit_transform(X_train_raw)

    # TRANSFORM TEST DATA USING FITTED VOCABULARY ONLY
    X_test_vec = vectorizer.transform(X_test_raw)

    if save_vectorizer:
        vec_path = Path(vectorizer_path)
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, vec_path)
        print(f"[Preprocessor] TF-IDF Vectorizer serialized to: {vec_path}")

    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


def load_vectorizer(vectorizer_path: Path | str = VECTORIZER_PATH) -> TfidfVectorizer:
    """Load the serialized TF-IDF vectorizer.

    Parameters
    ----------
    vectorizer_path : Path | str
        Path to joblib file.

    Returns
    -------
    TfidfVectorizer
        Fitted vectorizer instance.
    """
    path = Path(vectorizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {path}. Run training pipeline first.")
    return joblib.load(path)
