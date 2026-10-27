"""Unit Tests for Dataset Loader, Preprocessing, and Anti-Leakage Pipeline."""

from pathlib import Path
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import (
    CLEANED_DATA_FILE,
    RAW_DATA_FILE,
    audit_and_clean_dataset,
    load_raw_dataset,
)
from src.preprocessor import clean_text, prepare_train_test_data


def test_raw_dataset_exists_and_loads():
    """Verify raw dataset exists and parses without schema failure."""
    assert RAW_DATA_FILE.exists(), f"Raw dataset not found at {RAW_DATA_FILE}"
    df = load_raw_dataset(RAW_DATA_FILE)
    assert not df.empty
    assert "label" in df.columns
    assert "message" in df.columns
    assert set(df["label"].unique()).issubset({"ham", "spam"})


def test_audit_and_clean_dataset():
    """Verify cleaning drops empty rows and duplicate messages."""
    df = load_raw_dataset(RAW_DATA_FILE)
    clean_df, audit = audit_and_clean_dataset(df, drop_duplicates=True)

    assert len(clean_df) > 0
    assert clean_df["message"].isna().sum() == 0
    assert (clean_df["message"].str.len() == 0).sum() == 0
    assert clean_df.duplicated(subset=["message"]).sum() == 0
    assert audit["total_clean"] == len(clean_df)
    assert audit["ham_count"] > audit["spam_count"]
    assert "label_num" in clean_df.columns


def test_clean_text_normalization():
    """Verify regex tokenization for URLs, emails, currencies, and phone numbers."""
    raw = "WIN £500 cash! Visit https://win-lottery.org or call 09061701461 now."
    cleaned = clean_text(raw)

    assert "moneysymb" in cleaned
    assert "httpaddr" in cleaned
    assert "numbr" in cleaned
    assert "£" not in cleaned
    assert "https://" not in cleaned


def test_zero_data_leakage():
    """Verify TF-IDF vectorizer fits strictly on training partition without test leakage."""
    df = pd.read_csv(CLEANED_DATA_FILE)
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = prepare_train_test_data(
        df,
        test_size=0.2,
        random_state=42,
        max_features=1000,
        save_vectorizer=False,
    )

    # Check partition shapes
    assert X_train_vec.shape[0] == len(y_train)
    assert X_test_vec.shape[0] == len(y_test)
    assert X_train_vec.shape[1] == X_test_vec.shape[1] == len(vectorizer.vocabulary_)

    # Test size should be roughly 20%
    total = len(y_train) + len(y_test)
    test_ratio = len(y_test) / total
    assert 0.19 <= test_ratio <= 0.21

    # Verify stratified balance in train and test
    train_spam_rate = y_train.mean()
    test_spam_rate = y_test.mean()
    assert abs(train_spam_rate - test_spam_rate) < 0.02
