"""End-to-End Training and Evaluation Pipeline.

Executes dataset acquisition, leakage-free preprocessing, TF-IDF vectorization,
model training for 4 algorithms, quantitative evaluation, and figure generation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    CLEANED_DATA_FILE,
    DEFAULT_DATA_DIR,
    RAW_DATA_FILE,
    audit_and_clean_dataset,
    download_uci_dataset,
    load_raw_dataset,
)
from src.evaluate import evaluate_all_models
from src.preprocessor import prepare_train_test_data
from src.train_models import DEFAULT_MODELS_DIR, train_and_save_models


def run_pipeline() -> None:
    """Run full pipeline end to end."""
    start_time = time.time()
    print("=" * 70)
    print("   INTELLIGENT SPAM & PHISHING MESSAGE DETECTOR - TRAINING PIPELINE")
    print("=" * 70)

    # 1. Dataset Acquisition
    print("\n[Step 1/5] Acquiring and validating UCI SMS Spam Collection dataset...")
    raw_path = download_uci_dataset(data_dir=DEFAULT_DATA_DIR)
    raw_df = load_raw_dataset(raw_path)
    print(f"  -> Raw records loaded: {len(raw_df)}")

    # 2. Data Cleaning & Hygiene
    print("\n[Step 2/5] Cleaning dataset and removing duplicate records...")
    clean_df, audit = audit_and_clean_dataset(raw_df, drop_duplicates=True)
    clean_df.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"  -> Total clean messages: {audit['total_clean']}")
    print(f"  -> Ham: {audit['ham_count']} | Spam: {audit['spam_count']} ({audit['spam_ratio_pct']}%)")
    print(f"  -> Cleaned dataset saved to: {CLEANED_DATA_FILE}")

    # 3. Preprocessing & Leakage-Free Splitting
    print("\n[Step 3/5] Preprocessing text and performing leakage-free TF-IDF extraction...")
    print("  -> Performing train_test_split (80% train, 20% test, stratified)...")
    print("  -> Fitting TF-IDF Vectorizer ONLY on training partition...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer = prepare_train_test_data(
        clean_df,
        test_size=0.20,
        random_state=42,
        max_features=3000,
        save_vectorizer=True,
    )
    print(f"  -> Vocabulary size: {len(vectorizer.vocabulary_)} features")
    print(f"  -> Training feature matrix: {X_train_vec.shape}")
    print(f"  -> Test feature matrix:     {X_test_vec.shape}")

    # 4. Model Training (4 Classifiers)
    print("\n[Step 4/5] Training 4 Machine Learning Classifiers...")
    trained_models = train_and_save_models(
        X_train_vec,
        y_train,
        models_dir=DEFAULT_MODELS_DIR,
        random_state=42
    )

    # 5. Model Evaluation and Visualization Generation
    print("\n[Step 5/5] Evaluating models on unseen test partition...")
    summary_df, detailed_metrics = evaluate_all_models(
        trained_models,
        X_test_vec,
        y_test,
        figures_dir=PROJECT_ROOT / "reports" / "figures",
        models_dir=DEFAULT_MODELS_DIR
    )

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("                  BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70)
    print(f"Pipeline finished successfully in {elapsed:.2f} seconds.")
    print("All models, metrics, and visualization figures saved in models/ and reports/figures/")


if __name__ == "__main__":
    run_pipeline()
