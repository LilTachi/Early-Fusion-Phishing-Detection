#!/usr/bin/env python3
# train_eval_infer.py
# ─────────────────────────────────────────────────────────────────────────────
# 70 / 15 / 15  Train → Validate → Test (Inference) Pipeline
# Abundo, Custodio, Pastoral — Mapua University 2026
#
# Replaces the pure cross-validation flow in run_pipeline.py with a proper
# held-out split so the test set is NEVER seen during training or tuning.
#
# Split sizes (10, 000 emails):
#   Train      70 %  7 000   fit assembler + classifiers
#   Validate   15 %  1 500   tune / pick best classifier per condition
#   Test       15 %  1 500   final one-shot evaluation (inference)
#
# Usage:
#   # Real data (emails in data/raw/<Dataset>/{spam,ham}/)
#   python train_eval_infer.py
#   python train_eval_infer.py --datasets SpamAssassin Nazario
#   python train_eval_infer.py --grid-search
#
#   # Smoke test (no real data needed)
#   python train_eval_infer.py --smoke
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, roc_curve,
)

# ── Project modules (unchanged) ──────────────────────────────────────────────
from config      import DATASETS, OUTPUTS_DIR, MODELS_DIR, RANDOM_SEED
from data_loader import load_all_datasets, make_synthetic_dataset
from early_fusion import (
    FeatureAssembler, get_classifiers,
    _best_rf, _best_svm,          # grid-search helpers
    compute_metrics, fpr_at_tpr,
    CONDITIONS, save_results,
)

SPLIT_TRAIN = 0.70
SPLIT_VAL   = 0.15   # of total; test gets the remainder (also 0.15)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1.  DATA SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def split_dataset(df: pd.DataFrame,
                  train_frac: float = SPLIT_TRAIN,
                  val_frac:   float = SPLIT_VAL,
                  seed:       int   = RANDOM_SEED):
    """
    Stratified 70 / 15 / 15 split.
    Stratification keeps the spam:ham ratio equal in all three subsets.

    Returns
    -------
    df_train, df_val, df_test  — three DataFrames
    """
    # Step 1: carve out test set (15 %)
    val_test_frac = 1.0 - train_frac          # 0.30 of total
    df_train, df_temp = train_test_split(
        df,
        test_size=val_test_frac,
        stratify=df["label"],
        random_state=seed,
    )

    # Step 2: split the remaining 30 % evenly into val / test (each 15 %)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,                        # 50 % of 30 % = 15 % of total
        stratify=df_temp["label"],
        random_state=seed,
    )

    print(f"\n  Split summary (n={len(df):,}):")
    for name, subset in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        spam = (subset["label"] == 1).sum()
        ham  = (subset["label"] == 0).sum()
        print(f"    {name:6s}  {len(subset):5,}  "
              f"({len(subset)/len(df)*100:.1f}%)  "
              f"spam={spam}  ham={ham}")

    return df_train, df_val, df_test


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.  TRAINING PHASE
#     Fit FeatureAssembler + all classifiers on df_train ONLY.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(df_train: pd.DataFrame,
          dataset_name: str,
          grid_search: bool = False):
    """
    1. Fit FeatureAssembler (TF-IDF vocab + scalers) on training data.
    2. Fit one classifier per condition × classifier combination.
    3. Persist assembler + models to disk.

    Returns
    -------
    asm       : fitted FeatureAssembler
    models    : dict  { condition: { clf_name: fitted_clf } }
    """
    print(f"\n{'━'*65}")
    print(f"  [TRAIN]  {dataset_name}  (n={len(df_train):,})")
    print(f"{'━'*65}")

    # ── Assemble features (fit on train only) ────────────────────────────────
    asm = FeatureAssembler(use_discriminative_vocab=True)
    asm.fit_transform(df_train)

    X = {
        "header_only": asm.X_header,
        "body_only":   asm.X_body,
        "fusion":      asm.X_fusion,
    }
    y_train = asm.y

    # ── Classifiers ──────────────────────────────────────────────────────────
    base_clfs = get_classifiers()

    if grid_search:
        print("\n  Grid-searching RF and SVM on fusion features …")
        base_clfs["RF"]  = _best_rf( asm.X_fusion, y_train)
        base_clfs["SVM"] = _best_svm(asm.X_fusion, y_train)

    # ── Fit every condition × classifier ─────────────────────────────────────
    models = {cond: {} for cond in CONDITIONS}

    for condition in CONDITIONS:
        X_cond = X[condition]
        for clf_name, clf in base_clfs.items():
            print(f"  Fitting  {condition:12s} | {clf_name} …", end=" ")
            clf_clone = _clone_clf(clf)
            clf_clone.fit(X_cond, y_train)
            models[condition][clf_name] = clf_clone
            print("done")

    # ── Persist ──────────────────────────────────────────────────────────────
    asm_path = os.path.join(MODELS_DIR, f"assembler_{dataset_name}.pkl")
    asm.save(asm_path)

    mdl_path = os.path.join(MODELS_DIR, f"models_{dataset_name}.pkl")
    with open(mdl_path, "wb") as f:
        pickle.dump(models, f)

    print(f"\n  Assembler saved → {asm_path}")
    print(f"  Models    saved → {mdl_path}")
    return asm, models


def _clone_clf(clf):
    """Return a fresh unfitted copy of a classifier (avoids shared state)."""
    from sklearn.base import clone
    return clone(clf)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3.  VALIDATION PHASE
#     Evaluate on df_val to pick the best classifier per condition.
#     No re-fitting happens here — only predict().
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate(asm, models: dict,
             df_val: pd.DataFrame,
             dataset_name: str) -> pd.DataFrame:
    """
    Run inference on the validation set and score all
    condition × classifier combinations.

    Returns
    -------
    val_results : DataFrame with one row per (condition, classifier)
    """
    print(f"\n{'━'*65}")
    print(f"  [VALIDATE]  {dataset_name}  (n={len(df_val):,})")
    print(f"{'━'*65}")

    X_val = asm.transform(df_val)
    y_val = df_val["label"].values.astype(int)

    rows = []
    for condition in CONDITIONS:
        X_cond = X_val[condition]
        for clf_name, clf in models[condition].items():
            y_pred = clf.predict(X_cond)
            y_prob = clf.predict_proba(X_cond)[:, 1]
            m = compute_metrics(y_val, y_pred, y_prob)
            m.update({
                "dataset":    dataset_name,
                "condition":  condition,
                "classifier": clf_name,
                "phase":      "val",
            })
            rows.append(m)
            print(f"  {condition:12s} | {clf_name:4s} | "
                  f"F1={m['f1']*100:.2f}%  "
                  f"Prec={m['precision']*100:.2f}%  "
                  f"Rec={m['recall']*100:.2f}%  "
                  f"FPR@95={m['fpr_at_95tpr']:.4f}")

    val_results = pd.DataFrame(rows)

    # ── Show best per condition ───────────────────────────────────────────────
    print("\n  Best per condition (by F1):")
    for condition in CONDITIONS:
        sub  = val_results[val_results["condition"] == condition]
        best = sub.loc[sub["f1"].idxmax()]
        print(f"    {condition:12s}  →  {best['classifier']}  "
              f"F1={best['f1']*100:.2f}%")

    return val_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4.  TEST / INFERENCE PHASE
#     One-shot evaluation on the held-out test set.
#     The test set is NEVER touched before this step.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test(asm, models: dict,
         df_test: pd.DataFrame,
         dataset_name: str) -> pd.DataFrame:
    """
    Final evaluation on the held-out test set.
    Mirrors validate() but marks phase='test'.

    Returns
    -------
    test_results : DataFrame with one row per (condition, classifier)
    """
    print(f"\n{'━'*65}")
    print(f"  [TEST / INFERENCE]  {dataset_name}  (n={len(df_test):,})")
    print(f"{'━'*65}")

    X_test = asm.transform(df_test)
    y_test = df_test["label"].values.astype(int)

    rows = []
    for condition in CONDITIONS:
        X_cond = X_test[condition]
        for clf_name, clf in models[condition].items():
            y_pred = clf.predict(X_cond)
            y_prob = clf.predict_proba(X_cond)[:, 1]
            m = compute_metrics(y_test, y_pred, y_prob)
            m.update({
                "dataset":    dataset_name,
                "condition":  condition,
                "classifier": clf_name,
                "phase":      "test",
            })
            rows.append(m)
            print(f"  {condition:12s} | {clf_name:4s} | "
                  f"F1={m['f1']*100:.2f}%  "
                  f"Prec={m['precision']*100:.2f}%  "
                  f"Rec={m['recall']*100:.2f}%  "
                  f"AUC={m['auc_roc']:.4f}  "
                  f"FPR@95={m['fpr_at_95tpr']:.4f}")

    test_results = pd.DataFrame(rows)
    _print_test_summary(test_results, dataset_name)
    return test_results


def _print_test_summary(test_results: pd.DataFrame, dataset_name: str):
    """Print the final fusion vs single-source comparison table."""
    print(f"\n  ── Final results: {dataset_name} ──")
    print(f"  {'Condition':<14} {'Clf':<5} "
          f"{'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'AUC':>7} {'PR-AUC':>8} {'FPR@95':>8}")
    print(f"  {'─'*80}")

    for condition in CONDITIONS:
        sub = test_results[test_results["condition"] == condition]
        for _, row in sub.iterrows():
            marker = "◆ " if condition == "fusion" else "  "
            print(f"  {marker}{condition:<12} {row['classifier']:<5} "
                  f"{row['accuracy']*100:>6.2f}%"
                  f"{row['precision']*100:>7.2f}%"
                  f"{row['recall']*100:>7.2f}%"
                  f"{row['f1']*100:>7.2f}%"
                  f"{row['auc_roc']:>7.4f}"
                  f"{row['pr_auc']:>8.4f}"
                  f"{row['fpr_at_95tpr']:>8.4f}")
    print(f"\n  (◆ = Early Fusion condition)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5.  SINGLE-EMAIL INFERENCE
#     Use a trained model to classify one raw email at inference time.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def predict_email(raw_email_path: str,
                  dataset_name:   str,
                  condition:      str = "fusion",
                  clf_name:       str = "RF") -> dict:
    """
    Classify a single raw .eml file using a previously trained model.

    Parameters
    ----------
    raw_email_path : path to a single .eml file
    dataset_name   : which trained model to load (e.g. "SpamAssassin")
    condition      : "fusion" | "header_only" | "body_only"
    clf_name       : "RF" | "NB" | "SVM"

    Returns
    -------
    dict with keys: label (0/1), confidence (float), condition, classifier
    """
    from data_loader import extract_header_features_from_file
    from branch_b_headers import extract_body_text

    asm_path = os.path.join(MODELS_DIR, f"assembler_{dataset_name}.pkl")
    mdl_path = os.path.join(MODELS_DIR, f"models_{dataset_name}.pkl")

    if not os.path.exists(asm_path) or not os.path.exists(mdl_path):
        raise FileNotFoundError(
            f"No trained model found for '{dataset_name}'. "
            f"Run train_eval_infer.py first."
        )

    asm = FeatureAssembler.load(asm_path)
    with open(mdl_path, "rb") as f:
        models = pickle.load(f)

    # Build a one-row DataFrame matching the training format
    header_feats = extract_header_features_from_file(raw_email_path)
    body_text    = extract_body_text(raw_email_path)

    from config import ALL_HEADER_FEATURES
    row = {"label": -1, "body_text": body_text}
    row.update(header_feats)
    df_single = pd.DataFrame([row])

    X_dict = asm.transform(df_single)
    clf    = models[condition][clf_name]
    pred   = clf.predict(X_dict[condition])[0]
    prob   = clf.predict_proba(X_dict[condition])[0][1]

    result = {
        "label":      int(pred),
        "verdict":    "PHISHING" if pred == 1 else "LEGITIMATE",
        "confidence": round(float(prob), 4),
        "condition":  condition,
        "classifier": clf_name,
    }
    print(f"\n  Email: {os.path.basename(raw_email_path)}")
    print(f"  Verdict    : {result['verdict']}")
    print(f"  Confidence : {result['confidence']:.2%}")
    print(f"  Model      : {clf_name} / {condition}")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6.  FULL PIPELINE RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_split_pipeline(df: pd.DataFrame,
                       dataset_name: str,
                       grid_search: bool = False) -> dict:
    """
    End-to-end: split → train → validate → test.

    Returns
    -------
    dict with keys: val_results, test_results, asm, models
    """
    # ── 1. Split ─────────────────────────────────────────────────────────────
    df_train, df_val, df_test = split_dataset(df, dataset_name=dataset_name)

    # ── 2. Train ─────────────────────────────────────────────────────────────
    t0 = time.time()
    asm, models = train(df_train, dataset_name, grid_search=grid_search)
    print(f"\n  Training completed in {time.time()-t0:.1f}s")

    # ── 3. Validate ──────────────────────────────────────────────────────────
    val_results = validate(asm, models, df_val, dataset_name)

    # ── 4. Test ──────────────────────────────────────────────────────────────
    test_results = test(asm, models, df_test, dataset_name)

    # ── 5. Save results ──────────────────────────────────────────────────────
    val_path  = os.path.join(OUTPUTS_DIR, f"val_results_{dataset_name}.csv")
    test_path = os.path.join(OUTPUTS_DIR, f"test_results_{dataset_name}.csv")
    val_results.to_csv(val_path,   index=False)
    test_results.to_csv(test_path, index=False)
    print(f"\n  Val  results → {val_path}")
    print(f"  Test results → {test_path}")

    return {
        "val_results":  val_results,
        "test_results": test_results,
        "asm":          asm,
        "models":       models,
    }


# ── fix: split_dataset didn't accept dataset_name kwarg, add it ──────────────
_orig_split = split_dataset

def split_dataset(df, train_frac=SPLIT_TRAIN, val_frac=SPLIT_VAL,
                  seed=RANDOM_SEED, dataset_name=""):
    return _orig_split(df, train_frac=train_frac, val_frac=val_frac, seed=seed)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(
        description="70/15/15 Train → Validate → Test Pipeline"
    )
    p.add_argument("--smoke",       action="store_true",
                   help="Run on 1000 synthetic emails (no real data needed).")
    p.add_argument("--datasets",    nargs="+",
                   choices=list(DATASETS.keys()),
                   default=list(DATASETS.keys()))
    p.add_argument("--grid-search", action="store_true",
                   help="Grid-search RF and SVM hyperparameters.")
    p.add_argument("--rebuild",     action="store_true",
                   help="Rebuild feature CSVs from raw .eml files.")
    p.add_argument("--infer",       metavar="EML_PATH",
                   help="Classify a single .eml file using a saved model.")
    p.add_argument("--infer-dataset", default="SpamAssassin",
                   help="Which saved model to use for --infer (default: SpamAssassin).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "═"*65)
    print("  Early Fusion — 70/15/15 Split Pipeline")
    print("  Abundo · Custodio · Pastoral — Mapua University 2026")
    print("="*65)

    # ── Single-email inference mode ───────────────────────────────────────────
    if args.infer:
        predict_email(
            raw_email_path=args.infer,
            dataset_name=args.infer_dataset,
            condition="fusion",
            clf_name="RF",
        )
        sys.exit(0)

    # ── Smoke test ────────────────────────────────────────────────────────────
    if args.smoke:
        print("\n  SMOKE TEST — 1 000 synthetic emails")
        df_smoke = make_synthetic_dataset(n_spam=500, n_ham=500, seed=RANDOM_SEED)
        run_split_pipeline(df_smoke, "Synthetic", grid_search=False)
        sys.exit(0)

    # ── Real datasets ─────────────────────────────────────────────────────────
    all_dfs = load_all_datasets(force_rebuild=args.rebuild)

    for ds_name in args.datasets:
        if ds_name not in all_dfs or all_dfs[ds_name].empty:
            print(f"\n  [{ds_name}] Skipped — no data found.")
            continue
        run_split_pipeline(
            all_dfs[ds_name],
            dataset_name=ds_name,
            grid_search=args.grid_search,
        )
