# Intelligent Spam & Phishing Message Detector 🛡️

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Tests Passing](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg?logo=pytest)](https://pytest.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **College Machine Learning Mini-Project**  
> An end-to-end, reproducible natural language processing (NLP) and machine learning framework designed to detect unsolicited spam and analyze heuristic phishing indicators in SMS communication.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Format Handling](#-dataset--format-handling)
3. [Methodology & Zero Data Leakage](#-methodology--zero-data-leakage)
4. [Machine Learning Algorithms Evaluated](#-machine-learning-algorithms-evaluated)
5. [Benchmark Results & Evaluation Metrics](#-benchmark-results--evaluation-metrics)
6. [Spam vs Phishing Distinction (Taxonomy)](#-spam-vs-phishing-distinction-taxonomy)
7. [Probability Safety & Score Calibration](#-probability-safety--score-calibration)
8. [Project Structure](#-project-structure)
9. [Installation & Setup](#-installation--setup)
10. [Running the Application](#-running-the-application)
11. [Running Automated Tests](#-running-automated-tests)
12. [Limitations & Future Scope](#-limitations--future-scope)

---

## 📖 Project Overview

SMS messaging remains a high-engagement communication channel with an estimated 98% open rate. Consequently, it has become a pervasive attack vector for commercial spam and fraudulent social-engineering schemes. 

This project implements a rigorous, beginner-friendly machine learning pipeline that:
- Ingests and cleans the publicly available, gold-standard **UCI SMS Spam Collection dataset**.
- Enforces strict isolation between training and testing sets to guarantee **zero data leakage**.
- Transforms raw text into high-dimensional numerical features using sublinear **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Trains and benchmarks **four distinct machine learning algorithms**:
  1. **Logistic Regression**
  2. **Support Vector Machine (SVM)** (with Platt-calibrated decision probabilities)
  3. **K-Nearest Neighbors (KNN)** (with cosine distance)
  4. **AdaBoost Classifier**
- Evaluates models using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves.
- Deploys an interactive, responsive **Streamlit Web Application** for single-message inspection, batch CSV prediction, and interactive model comparison.

---

## 📂 Dataset & Format Handling

### Dataset Provenance
We utilize the legitimate, publicly available **SMS Spam Collection dataset** curated by Tiago A. Almeida and José María Gómez Hidalgo from the **UCI Machine Learning Repository**:
- **Source URL:** `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- **Citation:** Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng'11.

### File Format & Parsing Hygiene
The raw dataset file (`SMSSpamCollection`) is a tab-separated text file without headers:
- **Format:** `<label>\t<message_text>`
- **Labels:** `ham` (legitimate) or `spam` (unsolicited)
- **Character Encoding:** Ingested using `latin-1` / `utf-8` with `quoting=3` (`QUOTE_NONE`) to correctly preserve non-ASCII currency symbols (`£`, `$`, `€`) and prevent text truncation when SMS messages contain quotes.

### Data Cleaning and Deduplication Audit
| Metric | Count / Percentage |
| :--- | :--- |
| **Total Raw Messages** | 5,574 |
| **Missing / Null Messages** | 0 |
| **Identified Duplicate Messages** | 414 |
| **Final Cleaned Unique Messages** | 5,160 |
| **Clean Legitimate (`ham`) Messages** | 4,518 (87.56%) |
| **Clean Spam (`spam`) Messages** | 642 (12.44%) |

---

## 🧪 Methodology & Zero Data Leakage

### 1. Text Normalization
Raw SMS texts undergo systematic cleaning:
- Conversion to lowercase.
- URL replacement with token `httpaddr`.
- Email address replacement with token `emailaddr`.
- Currency symbol conversion (`£`, `$`, `€`, `¥`, `₹`) to `moneysymb`.
- Long numeric sequences and telephone numbers converted to `numbr`.
- Punctuation removal and whitespace normalization.

### 2. Preventing Data Leakage
A critical flaw in many student projects is fitting the vectorizer on the entire dataset prior to splitting, which leaks test set vocabulary and frequency statistics into the training pipeline.

Our project enforces strict mathematical isolation:
1. Messages are first partitioned into **80% Training (4,128 messages)** and **20% Test (1,032 messages)** using **stratified sampling** (`stratify=y`) to maintain the 12.44% spam class proportion.
2. The `TfidfVectorizer` is fitted **solely on `X_train`**.
3. `X_test` is transformed exclusively using the frozen vocabulary learned from `X_train`.

### 3. TF-IDF Hyperparameters
- `max_features`: 3,000 top n-grams
- `ngram_range`: `(1, 2)` (captures both keywords and 2-word collocations like `claim prize`, `urgent call`)
- `sublinear_tf`: `True` ($TF_{scaled} = 1 + \log(TF)$) to dampen the influence of repeated words
- `stop_words`: Standard English stop words
- `min_df`: 2 (discards single-occurrence typos)

---

## 🤖 Machine Learning Algorithms Evaluated

| Algorithm | Theoretical Formulation & Configuration | Why Selected |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear classifier using sigmoid link function; L2 regularization ($C=1.0$), `solver='lbfgs'`. | Fast, highly interpretable baseline for high-dimensional sparse text. |
| **Support Vector Machine (SVM)** | Linear kernel maximizing hyperplane margin ($C=1.0$); wrapped with 5-fold `CalibratedClassifierCV` (Platt scaling). | Industry benchmark for sparse text classification with exceptional generalization. |
| **K-Nearest Neighbors (KNN)** | Non-parametric instance-based learner ($k=5$, distance weighting, cosine similarity metric). | Evaluates local neighborhood similarity in TF-IDF cosine vector space. |
| **AdaBoost** | Sequential boosting ensemble of 100 decision stumps with adaptive sample re-weighting (`learning_rate=0.8`). | Assesses if boosting weak classifiers improves recall on minority spam class. |

---

## 📊 Benchmark Results & Evaluation Metrics

The models were evaluated on the **completely unseen test set of 1,032 messages** (904 Ham, 128 Spam):

### Comprehensive Comparison Table

| Algorithm | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | ROC-AUC | TN (Ham) | FP | FN | TP (Spam) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** 🏆 | **99.13%** | **100.00%** | **92.97%** | **96.36%** | **0.9985** | **904** | **0** | **9** | **119** |
| **K-Nearest Neighbors (KNN)** | 97.97% | 98.20% | 85.16% | 91.21% | 0.9758 | 902 | 2 | 19 | 109 |
| **Logistic Regression** | 97.58% | 99.05% | 81.25% | 89.27% | 0.9985 | 903 | 1 | 24 | 104 |
| **AdaBoost** | 97.48% | 96.36% | 82.81% | 89.08% | 0.9934 | 900 | 4 | 22 | 106 |

### Key Findings
1. **Support Vector Machine (SVM)** achieved the best overall performance with **99.13% Accuracy** and **zero False Positives** ($FP=0$), meaning no legitimate messages were erroneously flagged as spam.
2. In spam detection, **Precision is paramount** because falsely marking a critical personal or banking message as spam carries severe consequences. SVM and Logistic Regression demonstrated near-perfect precision ($\ge 99\%$).
3. **KNN with Cosine Metric** demonstrated strong precision (98.20%) and out-performed Logistic Regression in recall (85.16% vs 81.25%).

All generated publication-grade figures are stored in `reports/figures/`:
- `model_comparison_bar.png`: Side-by-side grouped bar plot.
- `confusion_matrices.png`: 2x2 grid of confusion matrices.
- `roc_curves.png`: Combined ROC curves with AUC scores.

---

## 🛡️ Spam vs Phishing Distinction (Taxonomy)

> **Academic Rigor Notice (Requirement 9):**  
> Unsolicited commercial spam and malicious phishing (smishing) are distinct phenomena. The **UCI SMS Spam Collection** provides ground truth exclusively for **Ham vs Spam**. Claiming the machine learning model is an "AI Phishing Classifier" without phishing ground-truth labels is scientifically inaccurate.

### Threat Vector Taxonomy
1. **Commercial Spam:** Unsolicited marketing sent indiscriminately (ringtone promotions, bulk advertising). Intent: lead generation or annoyance. Threat: Low/Moderate.
2. **Smishing (SMS Phishing):** Targeted social-engineering messages engineered to deceive victims into surrendering sensitive credentials, OTPs, or banking details. Intent: financial theft, account takeover. Threat: Critical.

### How Our Framework Implements Both Rigorously:
- **ML Classifier (Tier 1):** Accurately determines if the message exhibits the linguistic traits of Spam vs Ham based on empirical training data.
- **Heuristic Smishing Auditor (Tier 2):** Scans the message for actionable social-engineering heuristics:
  - Masked destination links (e.g. `bit.ly`, `tinyurl.com`, `is.gd`, IP addresses).
  - High-urgency/intimidation phrases (`urgent`, `account suspended`, `verify immediately`).
  - Sensitive credential harvesting terms (`otp`, `cvv`, `password`, `login to restore`).
  - Deceptive monetary lures combined with external links.
- **Clear Disclaimers:** The UI transparently presents smishing ratings as rule-based risk indicators, preserving academic integrity.

---

## 📐 Probability Safety & Score Calibration

> **Requirement 10 Compliance:**  
> Non-probabilistic machine learning algorithms (such as raw linear SVMs) compute a signed hyperplane distance via `decision_function`, NOT a posterior probability. Presenting an uncalibrated geometric distance as a "percentage confidence" is mathematically incorrect.

### How We Handle This:
1. **Calibrated Models:** Our primary SVM estimator utilizes `CalibratedClassifierCV` with Platt scaling (5-fold cross-validation fitting a sigmoid function mapping hyperplane distance to calibrated probability $P(y=1|x)$).
2. **Defensive Prediction Guardrail:** The `src.predictor.predict_message()` engine inspects `hasattr(model, 'predict_proba')`:
   - If `True`: Returns calibrated confidence percentage (e.g. `98.4%`).
   - If `False`: Captures the raw geometric margin, returns `confidence_pct = None`, and displays an explicit educational warning: *"Uncalibrated geometric margin distance; not a statistical percentage."*
3. **Automated Verification:** Verified via `tests/test_safety.py::test_uncalibrated_model_never_returns_confidence_percentage`.

---

## 📁 Project Structure

```
Space03/
├── data/
│   ├── SMSSpamCollection            # Official UCI dataset (raw tab-separated text)
│   └── cleaned_sms_dataset.csv      # Cleaned and deduplicated dataset
├── models/
│   ├── tfidf_vectorizer.joblib      # TF-IDF vectorizer fitted ONLY on X_train
│   ├── logistic_regression.joblib   # Serialized Logistic Regression
│   ├── svm_model.joblib             # Serialized Calibrated SVM
│   ├── knn_model.joblib             # Serialized KNN
│   ├── adaboost_model.joblib        # Serialized AdaBoost
│   ├── evaluation_metrics.json      # Complete quantitative JSON metrics
│   └── metrics_summary.csv          # Comparative metrics table
├── reports/
│   └── figures/
│       ├── model_comparison_bar.png # Side-by-side grouped metrics plot
│       ├── confusion_matrices.png   # 2x2 grid of confusion matrices
│       └── roc_curves.png           # Overlaid ROC curves with AUC values
├── src/
│   ├── __init__.py                  # Source package initialization
│   ├── data_loader.py               # Dataset acquisition, download, and parsing
│   ├── preprocessor.py              # Text cleaning and leakage-free TF-IDF pipeline
│   ├── train_models.py              # Model instantiation and training
│   ├── evaluate.py                  # Metric computation and figure generation
│   └── predictor.py                 # Safe inference pipeline & smishing heuristic audit
├── tests/
│   ├── __init__.py
│   ├── test_data_pipeline.py        # Data loading, cleaning, and anti-leakage tests
│   ├── test_models.py               # Model serialization and benchmark threshold tests
│   └── test_safety.py               # Probability safety and taxonomy tests
├── app.py                           # Responsive, interactive Streamlit Web Dashboard
├── train.py                         # Single-command end-to-end training pipeline
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Internet connection (for initial automatic UCI dataset download)

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "c:\Users\NITIN\Build With AI\Space03"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Run Full Training and Benchmark Pipeline
To re-train the models from scratch and regenerate all evaluation tables and plots:
```powershell
python train.py
```
*Execution takes ~8-10 seconds and outputs the full benchmark table in your terminal.*

### Option B: Launch the Interactive Streamlit Web Dashboard
```powershell
streamlit run app.py
```
or via the Python module runner:
```powershell
python -m streamlit run app.py
```
Once launched, open your web browser at:
`http://localhost:8501`

---

## 🧪 Running Automated Tests

Run the comprehensive pytest suite to verify dataset integrity, zero data leakage, model predictions, and probability calibration safety:
```powershell
python -m pytest tests/ -v
```

All 11 unit tests execute in ~5 seconds with 100% pass rate:
```text
tests/test_data_pipeline.py::test_raw_dataset_exists_and_loads PASSED
tests/test_data_pipeline.py::test_audit_and_clean_dataset PASSED
tests/test_data_pipeline.py::test_clean_text_normalization PASSED
tests/test_data_pipeline.py::test_zero_data_leakage PASSED
tests/test_models.py::test_all_four_models_present PASSED
tests/test_models.py::test_metrics_summary_content PASSED
tests/test_models.py::test_known_ham_prediction PASSED
tests/test_models.py::test_known_spam_prediction PASSED
tests/test_safety.py::test_uncalibrated_model_never_returns_confidence_percentage PASSED
tests/test_safety.py::test_smishing_heuristic_audit_taxonomy PASSED
tests/test_safety.py::test_benign_message_has_low_smishing_risk PASSED
```

---

## 🔮 Limitations & Future Scope

### Limitations
1. **Dataset Era:** The UCI SMS dataset reflects communication patterns predominantly from the early 2010s. Modern smishing attacks frequently mimic two-factor authentication (2FA) prompts and banking apps.
2. **Language Scope:** The current vectorizer is English-centric and does not account for multilingual SMS messages (e.g. Hindi, Spanish, Hinglish).
3. **URL Static Inspection:** The smishing heuristic analyzer performs static regex detection on URLs; it does not resolve live redirects or query DNS/threat-intelligence APIs.

### Future Scope
1. **Deep Learning & Transformers:** Benchmark DistilBERT or RoBERTa against the TF-IDF + SVM baseline.
2. **Real-time Threat Intelligence API:** Integrate Google Safe Browsing or VirusTotal API to inspect real URL redirection targets.
3. **Multilingual Embeddings:** Incorporate multilingual embeddings (such as XLM-RoBERTa or LaBSE) to support SMS detection in regional dialects.
4. **Mobile Deployment:** Package the inference model into an Android / iOS on-device TFLite / ONNX runtime for client-side spam filtering without transmitting private SMS messages to the cloud.

---

## 📜 License
Distributed under the MIT License. Developed for academic demonstration and machine learning mini-project coursework.
