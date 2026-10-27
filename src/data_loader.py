"""Data Loader Module for Intelligent Spam & Phishing Message Detector.

Handles acquisition, download, unzipping, and parsing of the official
UCI SMS Spam Collection dataset with strict format verification.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import pandas as pd

UCI_SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_FILE = DEFAULT_DATA_DIR / "SMSSpamCollection"
CLEANED_DATA_FILE = DEFAULT_DATA_DIR / "cleaned_sms_dataset.csv"


def download_uci_dataset(data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download and extract the UCI SMS Spam Collection dataset.

    Parameters
    ----------
    data_dir : Path | str
        Destination directory to store the dataset files.
    force : bool
        If True, re-download and overwrite existing files.

    Returns
    -------
    Path
        Path to the extracted SMSSpamCollection file.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    raw_file = data_path / "SMSSpamCollection"
    zip_target = data_path / "sms_spam_collection.zip"

    if raw_file.exists() and not force:
        return raw_file

    print(f"[DataLoader] Downloading UCI dataset from: {UCI_SMS_URL}")
    req = urllib.request.Request(
        UCI_SMS_URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamDetector/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_target, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

        print("[DataLoader] Extracting archive...")
        with zipfile.ZipFile(zip_target, "r") as zip_ref:
            zip_ref.extractall(data_path)

        # Cleanup zip file to keep workspace clean
        if zip_target.exists():
            zip_target.unlink()

        if not raw_file.exists():
            # Check if extracted inside subdirectory or differently cased
            candidates = list(data_path.glob("*SMSSpamCollection*"))
            if candidates:
                candidates[0].rename(raw_file)

        print(f"[DataLoader] Dataset extracted successfully to: {raw_file}")
    except Exception as exc:
        print(f"[DataLoader] Warning during network download: {exc}")
        if not raw_file.exists():
            raise RuntimeError(
                f"Failed to acquire dataset from UCI URL. Error: {exc}. "
                f"Please ensure network access or place 'SMSSpamCollection' in {data_path}."
            ) from exc

    return raw_file


def load_raw_dataset(file_path: Path | str = RAW_DATA_FILE) -> pd.DataFrame:
    """Parse the raw SMS Spam Collection file.

    The UCI file is tab-delimited, contains no header, and uses Latin-1/UTF-8
    encoding. Rows have format: `<label>\t<message_text>`.

    Parameters
    ----------
    file_path : Path | str
        Path to the raw dataset file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'label' ('ham'/'spam') and 'message'.
    """
    path = Path(file_path)
    if not path.exists():
        path = download_uci_dataset(path.parent)

    # UCI dataset contains currency symbols (£, €, etc.) requiring latin-1 or utf-8 fallback
    encodings = ["utf-8", "latin-1", "cp1252"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                header=None,
                names=["label", "message"],
                usecols=[0, 1],
                encoding=enc,
                quoting=3,  # QUOTE_NONE prevents premature truncation of text containing quotes
                on_bad_lines="skip"
            )
            break
        except (UnicodeDecodeError, Exception):
            continue

    if df is None or df.empty:
        raise ValueError(f"Unable to parse dataset at {path} with tested encodings.")

    # Validate schema
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    # Filter to only valid ham and spam classes
    df = df[df["label"].isin(["ham", "spam"])].copy()

    return df


def audit_and_clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, dict]:
    """Audit missing values, empty strings, duplicates, and prepare clean dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    drop_duplicates : bool
        Whether to deduplicate identical messages.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Cleaned DataFrame and audit report dictionary.
    """
    total_raw = len(df)
    missing_count = df["message"].isna().sum() + (df["message"].str.len() == 0).sum()

    # Clean empty rows
    cleaned_df = df.dropna(subset=["message"]).copy()
    cleaned_df = cleaned_df[cleaned_df["message"].str.len() > 0].copy()

    duplicate_count = cleaned_df.duplicated(subset=["message"]).sum()
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    # Encode label: ham -> 0, spam -> 1
    cleaned_df["label_num"] = cleaned_df["label"].map({"ham": 0, "spam": 1}).astype(int)

    # Compute length features for analysis
    cleaned_df["char_length"] = cleaned_df["message"].str.len()
    cleaned_df["word_count"] = cleaned_df["message"].apply(lambda txt: len(txt.split()))

    ham_count = int((cleaned_df["label"] == "ham").sum())
    spam_count = int((cleaned_df["label"] == "spam").sum())

    report = {
        "total_raw": total_raw,
        "missing_cleaned": int(missing_count),
        "duplicates_identified": int(duplicate_count),
        "duplicates_dropped": drop_duplicates,
        "total_clean": len(cleaned_df),
        "ham_count": ham_count,
        "spam_count": spam_count,
        "spam_ratio_pct": round((spam_count / len(cleaned_df)) * 100, 2) if len(cleaned_df) else 0.0,
    }

    return cleaned_df, report


if __name__ == "__main__":
    raw_path = download_uci_dataset()
    raw_data = load_raw_dataset(raw_path)
    clean_data, audit = audit_and_clean_dataset(raw_data)
    clean_data.to_csv(CLEANED_DATA_FILE, index=False)
    print(f"[DataLoader] Successfully processed {len(clean_data)} messages.")
    print(f"[DataLoader] Audit Report: {audit}")
