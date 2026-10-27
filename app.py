"""Intelligent Spam & Phishing Message Detector - Streamlit Dashboard.

An interactive, responsive college ML mini-project application featuring:
- Real-time message classification across 4 ML models
- Calibrated probability visualization with strict probability safety (Req 10)
- Heuristic Smishing / Phishing Risk Audit with transparent scientific taxonomy (Req 9)
- Benchmark comparison matrices, ROC curves, and batch processing
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.predictor import audit_smishing_indicators, predict_message
from src.preprocessor import load_vectorizer
from src.train_models import DEFAULT_MODELS_DIR, load_all_models

# Page Configuration
st.set_page_config(
    page_title="Intelligent Spam & Phishing Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Responsive Glassmorphism & Modern Typography)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e3a8a 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
        background: linear-gradient(to right, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #e0e7ff;
        margin-top: 0.5rem;
        margin-bottom: 0;
        opacity: 0.95;
    }

    .metric-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    .badge-ham {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }

    .badge-spam {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    .badge-info {
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
    }

    .result-card {
        background-color: #ffffff;
        padding: 1.8rem;
        border-radius: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.07), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .smishing-box {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }

    .disclaimer-pill {
        background-color: #f1f5f9;
        color: #475569;
        font-size: 0.8rem;
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        border-left: 3px solid #64748b;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading machine learning artifacts...")
def get_cached_pipeline():
    """Load vectorizer and all 4 models once and cache in memory."""
    vec_path = DEFAULT_MODELS_DIR / "tfidf_vectorizer.joblib"
    vectorizer = load_vectorizer(vec_path)
    models = load_all_models(DEFAULT_MODELS_DIR)
    summary_path = DEFAULT_MODELS_DIR / "metrics_summary.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    return vectorizer, models, summary_df


# Load models safely
try:
    vectorizer, models, summary_df = get_cached_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)


# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Intelligent Spam & Phishing Message Detector</h1>
        <p>A college ML mini-project comparing 4 scikit-learn algorithms with TF-IDF, zero data leakage, and calibrated probability estimation.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if not pipeline_ready:
    st.error(f"⚠️ Model artifacts not found. Please run the training pipeline first: `python train.py`\n\nError: {pipeline_error}")
    st.stop()


# Sidebar Navigation & Model Settings
with st.sidebar:
    st.subheader("⚙️ System Control")
    selected_model_name = st.selectbox(
        "Select Classification Model:",
        options=list(models.keys()),
        index=1,  # Default to SVM (highest benchmark performance)
        help="Select which of the 4 trained machine learning algorithms to evaluate."
    )

    st.markdown("---")
    st.markdown("### 🏆 Algorithm Leaderboard")
    if not summary_df.empty:
        best_model = summary_df.sort_values(by="F1-Score (Spam)", ascending=False).iloc[0]
        st.success(f"**Top Model:** {best_model['Algorithm']}")
        st.write(f"🎯 **Accuracy:** `{best_model['Accuracy'] * 100:.2f}%`")
        st.write(f"🔬 **Precision:** `{best_model['Precision (Spam)'] * 100:.2f}%`")
        st.write(f"⚡ **F1-Score:** `{best_model['F1-Score (Spam)'] * 100:.2f}%`")
    st.markdown("---")
    st.info(
        "💡 **Academic Note:**\n"
        "Trained strictly on the **UCI SMS Spam Collection** (5,160 clean messages). "
        "TF-IDF was fitted solely on training splits to prevent data leakage."
    )


# Tabs
tab_live, tab_benchmark, tab_batch, tab_pipeline, tab_taxonomy = st.tabs([
    "🔍 Live Message Inspector",
    "📊 Model Benchmark & Comparison",
    "📁 Batch File Predictor",
    "🧪 Anti-Leakage Pipeline & Data",
    "🛡️ Spam vs Phishing Taxonomy"
])


# ==============================================================================
# TAB 1: LIVE MESSAGE INSPECTOR
# ==============================================================================
with tab_live:
    st.markdown("### Test Any Message in Real-Time")
    st.caption("Type or paste an SMS message below, or choose a preset test message to evaluate the detector.")

    # Preset message buttons
    col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
    preset_choice = ""
    if col_preset1.button("💬 Benign Chat (Ham)"):
        preset_choice = "Hey Sarah, are we still meeting in the computer lab at 3pm for the mini-project discussion?"
    if col_preset2.button("🎁 Prize / Lottery Spam"):
        preset_choice = "WINNER!! As a valued customer you have won a £900 cash reward or iPad! Call 09061701461 now to claim your prize."
    if col_preset3.button("⚠️ Urgent Smishing Lure"):
        preset_choice = "URGENT: Your bank account is suspended due to suspicious activity. Verify immediately at http://bit.ly/bank-security"
    if col_preset4.button("💳 Billing / Refund Scam"):
        preset_choice = "FINAL NOTICE: You have an unclaimed tax refund of $480.90 pending. Click http://tinyurl.com/irs-verify to confirm OTP."

    user_text = st.text_area(
        "Enter SMS text below:",
        value=preset_choice if preset_choice else "",
        placeholder="Type or paste any SMS text here (e.g., 'Hey are you free tomorrow?' or 'Congratulations, you won a free ticket!')...",
        height=110
    )

    col_btn, col_clear = st.columns([1, 5])
    predict_clicked = col_btn.button("⚡ Inspect Message", type="primary", use_container_width=True)

    if (predict_clicked or preset_choice) and user_text.strip():
        with st.spinner("Analyzing message with TF-IDF and selected classifier..."):
            res = predict_message(
                user_text,
                model_name=selected_model_name,
                vectorizer=vectorizer,
                models=models
            )

        st.markdown("---")

        # Result Top Banner
        res_col1, res_col2 = st.columns([1.5, 2.5])

        with res_col1:
            st.markdown("#### Primary ML Classification")
            if res["is_spam"]:
                st.markdown(
                    f"""
                    <div style="background-color: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 1.2rem; text-align: center;">
                        <span style="font-size: 2.2rem;">🚨</span>
                        <h2 style="color: #991b1b; margin: 0.3rem 0;">SPAM MESSAGE</h2>
                        <span class="metric-badge badge-spam">Model: {res['model_used']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style="background-color: #ecfdf5; border: 2px solid #10b981; border-radius: 12px; padding: 1.2rem; text-align: center;">
                        <span style="font-size: 2.2rem;">✅</span>
                        <h2 style="color: #065f46; margin: 0.3rem 0;">HAM (LEGITIMATE)</h2>
                        <span class="metric-badge badge-ham">Model: {res['model_used']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("#### Confidence & Probability Metrics (Req 10 Safety)")
            if res["has_probability"] and res["confidence_pct"] is not None:
                st.write(f"**Calibrated Confidence:** `{res['confidence_pct']:.2f}%`")
                st.progress(res["confidence_pct"] / 100.0)

                col_p1, col_p2 = st.columns(2)
                col_p1.metric("Ham Probability", f"{res['probabilities']['Ham']:.2f}%")
                col_p2.metric("Spam Probability", f"{res['probabilities']['Spam']:.2f}%")
                st.caption(f"ℹ️ {res['score_interpretation']}")
            else:
                # Requirement 10: Pure uncalibrated decision distance
                st.warning("⚠️ **Uncalibrated Model Output Detected**")
                st.write(f"**Geometric Hyperplane Margin:** `{res['decision_score']}`")
                st.caption(f"🛡️ **Safety Guardrail:** {res['score_interpretation']}")

        # TF-IDF Token Importance & Smishing Heuristics Side by Side
        st.markdown("---")
        feat_col, smish_col = st.columns(2)

        with feat_col:
            st.markdown("#### 🔤 Influential TF-IDF N-Grams")
            if res["top_features"]:
                feat_df = pd.DataFrame(res["top_features"])
                feat_df.columns = ["Term / N-Gram", "TF-IDF Weight"]
                st.dataframe(feat_df, use_container_width=True, hide_index=True)
            else:
                st.info("No vocabulary terms from training lexicon found in this message.")

        with smish_col:
            st.markdown("#### 🛡️ Heuristic Smishing / Phishing Risk Audit")
            audit = res["smishing_audit"]
            risk_badge = f"<span style='color: white; background-color: {audit['risk_color']}; padding: 0.3rem 0.8rem; border-radius: 6px; font-weight: bold;'>{audit['smishing_risk']} RISK</span>"
            st.markdown(f"**Smishing Threat Level:** {risk_badge}", unsafe_allow_html=True)

            if audit["detected_indicators"]:
                st.markdown("**Detected Threat Red Flags:**")
                for ind in audit["detected_indicators"]:
                    st.write(f"• 🚩 {ind}")
            else:
                st.write("• ✅ No common smishing patterns (urgency, masking shorteners, credential hooks) detected.")

            st.markdown(
                f"<div class='disclaimer-pill' style='margin-top: 0.8rem;'>{audit['disclaimer']}</div>",
                unsafe_allow_html=True
            )


# ==============================================================================
# TAB 2: MODEL BENCHMARK & COMPARISON
# ==============================================================================
with tab_benchmark:
    st.markdown("### Quantitative Model Evaluation & Benchmarking")
    st.write(
        "Four machine learning algorithms were trained on the training set (80%) and evaluated "
        "on the completely unseen test partition (20% = 1,032 messages: 904 Ham, 128 Spam)."
    )

    if not summary_df.empty:
        # KPI Metric Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        best_acc_row = summary_df.sort_values(by="Accuracy", ascending=False).iloc[0]
        best_prec_row = summary_df.sort_values(by="Precision (Spam)", ascending=False).iloc[0]
        best_rec_row = summary_df.sort_values(by="Recall (Spam)", ascending=False).iloc[0]
        best_f1_row = summary_df.sort_values(by="F1-Score (Spam)", ascending=False).iloc[0]

        kpi1.metric("Highest Accuracy", f"{best_acc_row['Accuracy']*100:.2f}%", best_acc_row['Algorithm'])
        kpi2.metric("Highest Precision", f"{best_prec_row['Precision (Spam)']*100:.2f}%", best_prec_row['Algorithm'])
        kpi3.metric("Highest Recall", f"{best_rec_row['Recall (Spam)']*100:.2f}%", best_rec_row['Algorithm'])
        kpi4.metric("Highest F1-Score", f"{best_f1_row['F1-Score (Spam)']*100:.2f}%", best_f1_row['Algorithm'])

        st.markdown("#### Algorithm Performance Comparison Table")
        st.dataframe(
            summary_df.style.highlight_max(
                subset=["Accuracy", "Precision (Spam)", "Recall (Spam)", "F1-Score (Spam)"],
                color="#bbf7d0"
            ),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("#### Publication-Grade Visualizations")

    figures_dir = Path(__file__).resolve().parent / "reports" / "figures"
    bar_img = figures_dir / "model_comparison_bar.png"
    cm_img = figures_dir / "confusion_matrices.png"
    roc_img = figures_dir / "roc_curves.png"

    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📊 Grouped Metrics Comparison",
        "🔲 Confusion Matrices (2x2 Grid)",
        "📈 ROC Curves & AUC Analysis"
    ])

    with viz_tab1:
        if bar_img.exists():
            st.image(str(bar_img), caption="Side-by-Side Evaluation Metrics across all 4 Classifiers", use_container_width=True)
        else:
            st.info("Run `python train.py` to regenerate comparison bar plot.")

    with viz_tab2:
        if cm_img.exists():
            st.image(str(cm_img), caption="Confusion Matrices for Logistic Regression, SVM, KNN, and AdaBoost", use_container_width=True)
        else:
            st.info("Run `python train.py` to regenerate confusion matrix plot.")

    with viz_tab3:
        if roc_img.exists():
            st.image(str(roc_img), caption="Receiver Operating Characteristic (ROC) Comparison Curves", use_container_width=True)
        else:
            st.info("Run `python train.py` to regenerate ROC curves plot.")


# ==============================================================================
# TAB 3: BATCH FILE PREDICTOR
# ==============================================================================
with tab_batch:
    st.markdown("### Batch Message Classification")
    st.write("Upload a CSV file containing a column of SMS messages to process predictions at scale.")

    sample_template = pd.DataFrame({
        "message": [
            "Are we having lunch at noon today?",
            "URGENT: Claim your £500 supermarket gift voucher today! Text CLAIM to 8007",
            "Hey buddy, call me back when you get a chance.",
            "Your mobile account has been credited with 200 free minutes. Valid until Friday."
        ]
    })
    csv_sample_buffer = sample_template.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample CSV Template",
        data=csv_sample_buffer,
        file_name="sample_messages_input.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV file:", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded file with {len(batch_df)} rows.")

            # Identify message column
            text_col = None
            for candidate in ["message", "text", "sms", "content", "msg"]:
                if candidate in [c.lower() for c in batch_df.columns]:
                    text_col = [c for c in batch_df.columns if c.lower() == candidate][0]
                    break

            if text_col is None:
                text_col = st.selectbox("Select message column:", options=batch_df.columns)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                progress_bar = st.progress(0.0)
                predictions = []
                confidences = []
                smishing_risks = []

                total_rows = len(batch_df)
                for i, row_text in enumerate(batch_df[text_col]):
                    res = predict_message(str(row_text), model_name=selected_model_name, vectorizer=vectorizer, models=models)
                    predictions.append(res["prediction_label"])
                    confidences.append(res["confidence_pct"] if res["confidence_pct"] is not None else 0.0)
                    smishing_risks.append(res["smishing_audit"]["smishing_risk"])
                    progress_bar.progress((i + 1) / total_rows)

                batch_df["Predicted_Category"] = predictions
                batch_df["Confidence_Pct"] = confidences
                batch_df["Smishing_Risk_Level"] = smishing_risks

                st.markdown("#### Batch Results Preview")
                st.dataframe(batch_df.head(20), use_container_width=True)

                # Summary chart
                spam_count = int((batch_df["Predicted_Category"] == "Spam").sum())
                ham_count = int((batch_df["Predicted_Category"] == "Ham (Legitimate)").sum())

                chart_col1, chart_col2 = st.columns(2)
                chart_col1.metric("Spam Detected", f"{spam_count} ({spam_count/total_rows*100:.1f}%)")
                chart_col2.metric("Ham Detected", f"{ham_count} ({ham_count/total_rows*100:.1f}%)")

                # Export
                out_csv = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Annotated Results CSV",
                    data=out_csv,
                    file_name="classified_messages_results.csv",
                    mime="text/csv"
                )
        except Exception as batch_err:
            st.error(f"Error processing batch file: {batch_err}")


# ==============================================================================
# TAB 4: ANTI-LEAKAGE PIPELINE & DATA
# ==============================================================================
with tab_pipeline:
    st.markdown("### Methodology: Zero Data Leakage & Feature Engineering")

    st.write(
        "A common flaw in academic ML projects is **data leakage**, which occurs when test partition "
        "information bleeds into the training phase (e.g., fitting TF-IDF on the full dataset before splitting). "
        "Our pipeline enforces strict isolation."
    )

    st.markdown(
        """
        ```
        Raw UCI SMS Spam Dataset (5,574 rows)
                      │
                      ▼
        [1] Data Cleaning & Deduplication (414 duplicates dropped -> 5,160 clean rows)
                      │
                      ▼
        [2] Stratified Split (80% Train: 4,128 rows | 20% Test: 1,032 rows)
              │                                      │
              ▼                                      ▼
        [3] Fit TF-IDF Vectorizer                 [4] Transform ONLY (Using
            Strictly on X_train (Vocabulary: 3,000)   frozen vocabulary from X_train)
              │                                      │
              ▼                                      ▼
        [5] Train 4 ML Classifiers                [6] Unseen Evaluation & Scoring
        ```
        """
    )

    st.markdown("#### UCI SMS Spam Collection Dataset Statistics")
    data_stat1, data_stat2, data_stat3, data_stat4 = st.columns(4)
    data_stat1.metric("Raw Messages", "5,574")
    data_stat2.metric("Duplicates Dropped", "414")
    data_stat3.metric("Clean Ham Messages", "4,518 (87.6%)")
    data_stat4.metric("Clean Spam Messages", "642 (12.4%)")

    st.markdown("#### TF-IDF Vectorizer Configuration")
    st.markdown(
        """
        - **Vocabulary Size (`max_features`):** 3,000 top n-grams
        - **N-Gram Range:** `(1, 2)` (captures both individual keywords and 2-word phrases like 'free prize', 'urgent call')
        - **Sublinear TF Scaling:** $TF_{scaled} = 1 + \\log(TF)$ to dampen the influence of repeated words
        - **Stop Words:** Standard English stop words
        - **Minimum Document Frequency (`min_df`):** 2 (eliminates rare typos)
        """
    )


# ==============================================================================
# TAB 5: SPAM VS PHISHING TAXONOMY (ACADEMIC GUIDE)
# ==============================================================================
with tab_taxonomy:
    st.markdown("### Scientific Taxonomy: Spam vs Phishing (Smishing)")
    st.info(
        "**Academic Integrity Statement (Requirement 9):**\n"
        "Many beginner projects carelessly claim that a simple spam detector is an AI phishing detector. "
        "In rigorous computer science and cybersecurity, these are distinct threat categories."
    )

    col_tax1, col_tax2 = st.columns(2)

    with col_tax1:
        st.markdown("#### 📢 Commercial & Bulk Spam")
        st.write(
            """
            - **Definition:** Unsolicited bulk messaging sent indiscriminately to thousands of recipients.
            - **Primary Intent:** Commercial marketing, ringtone subscriptions, dating service promotion, affiliate traffic generation.
            - **Threat Level:** Low to moderate (annoyance, unwanted marketing).
            - **Dataset Ground Truth:** The **UCI SMS Spam Collection** specifically labels whether an SMS is legitimate personal communication (`ham`) or unsolicited bulk messaging (`spam`).
            """
        )

    with col_tax2:
        st.markdown("#### 🎣 Smishing (SMS Phishing)")
        st.write(
            """
            - **Definition:** Malicious social-engineering attack conducted over SMS aiming to harvest sensitive personal or financial credentials.
            - **Primary Intent:** Stealing bank credentials, identity theft, OTP interception, malware distribution.
            - **Threat Level:** Critical / Cybercrime.
            - **Detection Mechanism:** Requires specialized phishing ground-truth datasets, domain reputation lookups, and heuristic urgency analysis.
            """
        )

    st.markdown("---")
    st.markdown("#### How Our System Implements Multi-Tiered Analysis:")
    st.markdown(
        """
        1. **Tier 1 (Statistical ML Model):** Evaluates whether message text matches the statistical linguistic patterns of unsolicited Spam vs Ham (TF-IDF + SVM/LR/KNN/AdaBoost).
        2. **Tier 2 (Heuristic Risk Auditor):** Scans for structural social-engineering indicators (URL masking shorteners, urgency triggers, credential harvesting language).
        3. **Tier 3 (Scientific Transparency):** Unambiguously labels statistical outputs as Spam vs Ham and flags phishing risks as rule-based educational heuristics without fabricating false ML ground truth.
        """
    )
