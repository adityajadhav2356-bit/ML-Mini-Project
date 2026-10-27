"""Model Training Pipeline for Intelligent Spam & Phishing Message Detector.

Trains and serializes 4 core machine learning classifiers:
1. Logistic Regression
2. Support Vector Machine (SVM) with calibrated decision probabilities
3. K-Nearest Neighbors (KNN) with cosine metric
4. AdaBoost Classifier
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_model_definitions(random_state: int = 42) -> Dict[str, object]:
    """Define the 4 target classification algorithms with tuned hyperparameters.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, object]
        Dictionary mapping model names to scikit-learn estimator instances.
    """
    # LinearSVC does not natively output calibrated probabilities.
    # Wrapping with CalibratedClassifierCV implements Platt scaling / sigmoid calibration,
    # ensuring mathematically valid probabilities without fabricating confidence.
    base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=random_state)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=5)

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs"
        ),
        "Support Vector Machine (SVM)": calibrated_svm,
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",  # Cosine similarity is optimal for high-dimensional sparse TF-IDF
            weights="distance"
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=random_state
        ),
    }

    return models


def train_and_save_models(
    X_train_vec: np.ndarray,
    y_train: pd.Series | np.ndarray,
    models_dir: Path | str = DEFAULT_MODELS_DIR,
    random_state: int = 42
) -> Dict[str, object]:
    """Train all 4 classifiers on training features and save with joblib.

    Parameters
    ----------
    X_train_vec : np.ndarray
        TF-IDF feature matrix for training set.
    y_train : pd.Series | np.ndarray
        Training labels (0=ham, 1=spam).
    models_dir : Path | str
        Target directory to save model artifacts.
    random_state : int
        Random seed.

    Returns
    -------
    Dict[str, object]
        Dictionary of trained model instances.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    estimators = get_model_definitions(random_state=random_state)
    trained_models = {}

    file_slug_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Support Vector Machine (SVM)": "svm_model.joblib",
        "K-Nearest Neighbors (KNN)": "knn_model.joblib",
        "AdaBoost": "adaboost_model.joblib",
    }

    print("[ModelTraining] Beginning model training on training partition...")
    for name, model in estimators.items():
        print(f"  -> Fitting {name}...")
        model.fit(X_train_vec, y_train)
        trained_models[name] = model

        # Serialize individual model
        save_file = models_path / file_slug_map[name]
        joblib.dump(model, save_file)
        print(f"     Saved artifact to: {save_file}")

    print("[ModelTraining] All 4 algorithms successfully trained and serialized.")
    return trained_models


def load_all_models(models_dir: Path | str = DEFAULT_MODELS_DIR) -> Dict[str, object]:
    """Load all 4 trained model artifacts from disk.

    Parameters
    ----------
    models_dir : Path | str
        Directory containing model artifacts.

    Returns
    -------
    Dict[str, object]
        Dictionary mapping model names to loaded estimator instances.
    """
    models_path = Path(models_dir)
    file_slug_map = {
        "Logistic Regression": "logistic_regression.joblib",
        "Support Vector Machine (SVM)": "svm_model.joblib",
        "K-Nearest Neighbors (KNN)": "knn_model.joblib",
        "AdaBoost": "adaboost_model.joblib",
    }

    loaded = {}
    for name, filename in file_slug_map.items():
        filepath = models_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model file {filepath} not found. Run training first.")
        loaded[name] = joblib.load(filepath)

    return loaded
