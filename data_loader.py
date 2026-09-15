# data_loader.py
# Reads raw .eml files from data/raw/<dataset>/{spam,ham}/
# Extracts ALL_HEADER_FEATURES (Branch B) + raw body text (Branch A input)
# Saves one CSV per dataset: header features + body_text + label

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from email import message_from_bytes, message_from_string

from config import DATASETS, ALL_HEADER_FEATURES, DATA_DIR
from branch_b_headers import extract_header_features, extract_body_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Per-email processing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _process_file(filepath: str, label: int) -> dict | None:
    """Extract header features + body text from one email file."""
    try:
        header_feats = extract_header_features_from_file(filepath)
        body_text    = extract_body_text(filepath)
    except Exception as e:
        print(f"    [WARN] Could not process {os.path.basename(filepath)}: {e}")
        return None

    row = {"label": label, "body_text": body_text}
    row.update(header_feats)
    return row


def extract_header_features_from_file(filepath: str) -> dict:
    """Parse raw email and return header feature dict."""
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        msg = message_from_bytes(raw)
    except Exception:
        try:
            with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                msg = message_from_string(f.read())
        except Exception:
            return {name: 0.0 for name in ALL_HEADER_FEATURES}
    from branch_b_headers import extract_header_features
    return extract_header_features(msg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dataset builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_dataset(dataset_name: str, cfg: dict,
                  force_rebuild: bool = False) -> pd.DataFrame:
    """
    Build (or load from cache) the feature CSV for one dataset.
    Returns a DataFrame with columns:
        label, body_text, <ALL_HEADER_FEATURES...>
    """
    csv_path = cfg["csv"]

    if os.path.exists(csv_path) and not force_rebuild:
        print(f"  [{dataset_name}] Loading cached CSV: {csv_path}")
        return pd.read_csv(csv_path)

    spam_dir = cfg["spam_dir"]
    ham_dir  = cfg["ham_dir"]

    rows = []
    for label_val, folder, label_name in [
        (1, spam_dir, "spam"),
        (0, ham_dir,  "ham"),
    ]:
        if not os.path.isdir(folder):
            print(f"  [{dataset_name}] WARNING: {folder} not found — skipping {label_name}.")
            continue

        files = sorted([
            f for f in os.listdir(folder)
            if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))
        ])
        print(f"  [{dataset_name}] Processing {len(files):,} {label_name} emails…")

        for i, fname in enumerate(files):
            if i % 500 == 0 and i > 0:
                print(f"    … {i}/{len(files)}")
            row = _process_file(os.path.join(folder, fname), label_val)
            if row is not None:
                rows.append(row)

    if not rows:
        print(f"  [{dataset_name}] ERROR: No emails found. "
              f"Place files in {spam_dir}/ and {ham_dir}/")
        return pd.DataFrame()

    cols = ["label", "body_text"] + ALL_HEADER_FEATURES
    df   = pd.DataFrame(rows)[cols]
    df["body_text"] = df["body_text"].fillna("").astype(str)
    for feat in ALL_HEADER_FEATURES:
        df[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(0.0)

    df.to_csv(csv_path, index=False)
    spam_n = (df["label"] == 1).sum()
    ham_n  = (df["label"] == 0).sum()
    print(f"  [{dataset_name}] Saved {len(df):,} rows → {csv_path}")
    print(f"    Spam: {spam_n:,}  |  Ham: {ham_n:,}")
    return df


def load_all_datasets(force_rebuild: bool = False) -> dict:
    """Load or build all configured datasets. Returns {name: DataFrame}."""
    results = {}
    for name, cfg in DATASETS.items():
        print(f"\n── {name} ──")
        df = build_dataset(name, cfg, force_rebuild=force_rebuild)
        if not df.empty:
            results[name] = df
        else:
            print(f"  [{name}] Skipped (empty).")
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Synthetic data generator — for smoke-testing without real emails
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import numpy as np

_RNG = np.random.default_rng(42)

_SPAM_BODIES = [
    "URGENT your account has been suspended click here to verify immediately",
    "Congratulations you have won prize click to claim your reward now",
    "Your paypal account requires immediate verification confirm identity",
    "Security alert unusual login detected update your password now",
    "Invoice attached please confirm payment details immediately urgent",
    "Your apple id has been locked verify your account to restore access",
    "Final notice your account will be terminated unless you act now",
    "Click here to receive your tax refund claim before deadline expires",
    "Your subscription has expired renew now to avoid service interruption",
    "Bank alert unusual activity on your account please verify now",
]
_HAM_BODIES = [
    "Hi please find the meeting notes attached let me know if you have questions",
    "The quarterly report is ready for review please check before Thursday",
    "Following up on our discussion last week do you have time to chat",
    "Thanks for your email I will get back to you by end of week",
    "Team lunch is scheduled for Friday at noon let me know if you can make it",
    "Please review the attached proposal and share your feedback",
    "The project deadline has been moved to next month let us discuss",
    "I wanted to check in on the status of the deliverables",
    "Could you send me the latest version of the document when you get a chance",
    "Reminder that the training session is scheduled for next Tuesday morning",
]

def make_synthetic_dataset(n_spam: int = 300, n_ham: int = 300,
                            seed: int = 42) -> pd.DataFrame:
    """
    Generate a labelled DataFrame with realistic synthetic emails.
    Header features are drawn from distributions that approximate
    real spam/ham header statistics (based on Kulkarni et al. Table II).
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Spam header feature distributions (prob of feature=1)
    spam_probs = {
        "To_empty": 0.70, "BCC_notempty_To_empty": 0.40,
        "Total_Recp": 0.55, "Received_count": 0.50,
        "Span_time": 0.60, "From_symbol": 0.35,
        "Subject_symbol": 0.65, "Received_SPF": 0.55,
        "Authentication_Results": 0.60, "MessageID_domainname": 0.70,
        "MessageID_dollar": 0.30, "Reply_To_empty_symbol": 0.65,
        "Reply_To_domain": 0.60, "X_Mailer_exist": 0.75,
        "Content_Transfer_Encoding": 0.50, "Return_Path_empty": 0.55,
        "Return_Path_From_domain": 0.65,
        # extended
        "from_username_length": 0.45, "from_username_digit_ratio": 0.40,
        "from_username_special_ratio": 0.30, "from_username_suspicious_keyword": 0.55,
        "from_domain_typosquat_score": 0.35, "from_domain_subdomain_depth": 0.25,
        "from_tld_suspicious": 0.40, "replyto_username_length": 0.35,
        "replyto_domain_typosquat_score": 0.30, "replyto_tld_suspicious": 0.35,
        "returnpath_username_length": 0.30, "returnpath_domain_typosquat_score": 0.25,
        "returnpath_tld_suspicious": 0.30, "from_vs_replyto_edit_distance": 0.60,
        "from_vs_returnpath_edit_distance": 0.55, "from_vs_msgid_edit_distance": 0.65,
    }
    # Ham: lower probability of anomalous header features
    ham_probs = {k: max(0.02, v * 0.12) for k, v in spam_probs.items()}

    for _ in range(n_spam):
        row = {"label": 1,
               "body_text": _SPAM_BODIES[rng.integers(len(_SPAM_BODIES))]}
        for feat in ALL_HEADER_FEATURES:
            p = spam_probs.get(feat, 0.3)
            row[feat] = float(rng.random() < p)
        rows.append(row)

    for _ in range(n_ham):
        row = {"label": 0,
               "body_text": _HAM_BODIES[rng.integers(len(_HAM_BODIES))]}
        for feat in ALL_HEADER_FEATURES:
            p = ham_probs.get(feat, 0.05)
            row[feat] = float(rng.random() < p)
        rows.append(row)

    cols = ["label", "body_text"] + ALL_HEADER_FEATURES
    df   = pd.DataFrame(rows)[cols]
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    print("Data loader — synthetic smoke test")
    df = make_synthetic_dataset(200, 200)
    print(f"  Shape: {df.shape}")
    print(f"  Spam: {(df.label==1).sum()}  Ham: {(df.label==0).sum()}")
    print(f"  Columns: {list(df.columns[:6])} … (+{len(df.columns)-6} more)")
    print(f"\n  Sample header features (first spam row):")
    spam_row = df[df.label == 1].iloc[0]
    for feat in ALL_HEADER_FEATURES[:8]:
        print(f"    {feat:<40} {spam_row[feat]}")
