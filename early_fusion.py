# early_fusion.py
# ─────────────────────────────────────────────────────────────────────────────
# Early Fusion Pipeline — Thesis Core
# Abundo, Custodio, Pastoral — Mapua University 2026
#
# Three experimental CONDITIONS per classifier:
#   1. header_only  — Branch B features only (~33 dims)
#   2. body_only    — Branch A TF-IDF only (5 000 dims)
#   3. fusion       — Branch A ⊕ Branch B concatenated (~5 033 dims)
#
# Five CLASSIFIERS:
#   NB  — Gaussian Naive Bayes
#   RF  — Random Forest (grid-searched hyperparams)
#   SVM — Support Vector Machine (class-balanced, Platt-calibrated)
#   
# Evaluation: 10-fold stratified CV, per-dataset, all 6 precision-focused
#   metrics: accuracy, precision, recall, F1, AUC-ROC, PR-AUC, FPR@95TPR
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
import pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from config import (
    ALL_HEADER_FEATURES, CV_FOLDS, RANDOM_SEED, OUTPUTS_DIR, MODELS_DIR
)
from branch_a_body import BodyFeaturizer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metric helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fpr_at_tpr(y_true, y_prob, target_tpr: float = 0.95) -> float:
    """
    False positive rate when true positive rate is held at target_tpr.
    Primary operational metric per thesis methodology (Section 3, Table 7).
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    # find the threshold index where TPR first reaches target_tpr
    idx = np.searchsorted(tpr_arr, target_tpr)
    idx = min(idx, len(fpr_arr) - 1)
    return float(fpr_arr[idx])


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute all 7 evaluation metrics for one CV fold."""
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),                          4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0),        4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0),        4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0),        4),
        "auc_roc":   round(roc_auc_score(y_true, y_prob),                           4),
        "pr_auc":    round(average_precision_score(y_true, y_prob),                 4),
        "fpr_at_95tpr": round(fpr_at_tpr(y_true, y_prob, 0.95),                    4),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Classifier factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Grid-search param grids (kept small for speed; expand for final thesis run)
_RF_PARAM_GRID = {
    "n_estimators":   [100, 200],
    "max_depth":      [None, 20],
    "min_samples_split": [2, 5],
    "class_weight":   ["balanced"],
}

_SVM_PARAM_GRID = {
    "C":      [0.1, 1.0, 10.0],
    "kernel": ["rbf", "linear"],
    "gamma":  ["scale"],
    "class_weight": ["balanced"],
}


def _best_rf(X, y) -> RandomForestClassifier:
    """Grid-search RF hyperparams; returns best fitted estimator."""
    base = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    gs   = GridSearchCV(base, _RF_PARAM_GRID, cv=3,
                        scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"      RF best params: {gs.best_params_}")
    return gs.best_estimator_


def _best_svm(X, y):
    """Grid-search SVM hyperparams with Platt calibration; returns Pipeline."""
    svc  = SVC(random_state=RANDOM_SEED, probability=True)
    gs   = GridSearchCV(svc, _SVM_PARAM_GRID, cv=3,
                        scoring="f1", n_jobs=-1, refit=True)
    gs.fit(X, y)
    print(f"      SVM best params: {gs.best_params_}")
    return gs.best_estimator_


def get_classifiers() -> dict:
    """Return classifier instances (unfitted) for NB, RF stub, SVM stub."""
    return {
        "NB":  GaussianNB(),
        # RF and SVM get grid-searched inside evaluate_condition()
        "RF":  RandomForestClassifier(
                    n_estimators=200, max_depth=None,
                    class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale",
                   class_weight="balanced", probability=True,
                   random_state=RANDOM_SEED),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature assembly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeatureAssembler:
    """
    Builds the three feature matrices from a labelled DataFrame.

    The DataFrame must have columns:
        label      : int  (1=phishing/spam, 0=ham)
        body_text  : str  (raw email body)
        <ALL_HEADER_FEATURES> : float

    After calling fit_transform(df) the assembler exposes:
        X_header  : np.ndarray  shape (n, n_header_features)
        X_body    : sparse csr  shape (n, TFIDF_MAX_FEATURES)
        X_fusion  : np.ndarray  shape (n, n_header + n_body)
        y         : np.ndarray  shape (n,)
    """

    def __init__(self, use_discriminative_vocab: bool = True):
        self.body_featurizer = BodyFeaturizer(
            use_discriminative_vocab=use_discriminative_vocab
        )
        self._scaler_header = StandardScaler()
        self._scaler_body   = StandardScaler(with_mean=False)  # sparse-safe
        self._fitted        = False

    def fit_transform(self, df: pd.DataFrame):
        y          = df["label"].values.astype(int)
        texts      = df["body_text"].fillna("").tolist()
        X_hdr_raw  = df[ALL_HEADER_FEATURES].values.astype(float)

        print("    [Assembler] Fitting body TF-IDF …")
        X_body_sp  = self.body_featurizer.fit_transform(texts, y.tolist())

        print("    [Assembler] Scaling header features …")
        X_hdr_sc   = self._scaler_header.fit_transform(X_hdr_raw)

        print("    [Assembler] Scaling body features …")
        X_body_sc  = self._scaler_body.fit_transform(X_body_sp)

        X_body_dense = X_body_sc.toarray()
        X_fusion     = np.hstack([X_hdr_sc, X_body_dense])

        self._fitted   = True
        self.X_header  = X_hdr_sc
        self.X_body    = X_body_dense
        self.X_fusion  = X_fusion
        self.y         = y

        print(f"    [Assembler] Header shape : {X_hdr_sc.shape}")
        print(f"    [Assembler] Body shape   : {X_body_dense.shape}")
        print(f"    [Assembler] Fusion shape : {X_fusion.shape}")
        return self

    def transform(self, df: pd.DataFrame):
        """Transform new (unseen) data with fitted scalers/vectorizer."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        texts     = df["body_text"].fillna("").tolist()
        X_hdr_raw = df[ALL_HEADER_FEATURES].values.astype(float)
        X_body_sp = self.body_featurizer.transform(texts)
        X_hdr_sc  = self._scaler_header.transform(X_hdr_raw)
        X_body_sc = self._scaler_body.transform(X_body_sp).toarray()
        return {
            "header_only": X_hdr_sc,
            "body_only":   X_body_sc,
            "fusion":      np.hstack([X_hdr_sc, X_body_sc]),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core evaluation: 10-fold CV with all metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evaluate_condition(
        X: np.ndarray,
        y: np.ndarray,
        clf_name: str,
        clf,
        condition: str,
        dataset_name: str,
        n_folds: int = CV_FOLDS,
) -> dict:
    """
    Run n-fold stratified CV and return averaged metrics dict.
    Uses cross_val_predict to accumulate out-of-fold probabilities for
    threshold-based metrics (FPR@95TPR, PR-AUC).
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                         random_state=RANDOM_SEED)

    # For NB, dense input is needed
    X_in = np.array(X) if not isinstance(X, np.ndarray) else X

    print(f"      [{dataset_name}] {condition:12s} | {clf_name:6s} | "
          f"shape={X_in.shape} | folds={n_folds}")

    # Accumulate out-of-fold predictions
    y_pred = cross_val_predict(clf, X_in, y, cv=cv, method="predict",
                               n_jobs=-1)
    y_prob = cross_val_predict(clf, X_in, y, cv=cv, method="predict_proba",
                               n_jobs=-1)[:, 1]

    m = compute_metrics(y, y_pred, y_prob)
    m.update({
        "dataset":   dataset_name,
        "condition": condition,
        "classifier": clf_name,
        "n_features": X_in.shape[1],
        "n_samples":  X_in.shape[0],
    })
    return m


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main experiment runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONDITIONS = ["header_only", "body_only", "fusion"]
# CNN and BiLSTM conditions are handled in neural_fusion.py (extended)


def run_experiment(
        dataset_name: str,
        df: pd.DataFrame,
        grid_search: bool = False,
        n_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """
    Full experiment for one dataset:
      - Assemble features (3 conditions)
      - Evaluate NB / RF / SVM under each condition
      - Return results DataFrame

    Args:
        dataset_name : human-readable label for the dataset
        df           : DataFrame with label, body_text, + header features
        grid_search  : if True, grid-search RF and SVM hyperparams
        n_folds      : number of CV folds (default 10)
    """
    print(f"\n{'━'*65}")
    print(f"  EXPERIMENT: {dataset_name}  "
          f"(n={len(df):,}, spam={( df.label==1).sum():,}, "
          f"ham={(df.label==0).sum():,})")
    print(f"{'━'*65}")

    # ── Step 1: assemble all three feature matrices ──
    print("\n  [Step 1] Assembling features …")
    asm = FeatureAssembler(use_discriminative_vocab=True)
    asm.fit_transform(df)

    feature_matrices = {
        "header_only": asm.X_header,
        "body_only":   asm.X_body,
        "fusion":      asm.X_fusion,
    }
    y = asm.y

    # ── Step 2: get classifiers ──
    classifiers = get_classifiers()
    if grid_search:
        print("\n  [Step 2] Grid-searching RF and SVM …")
        classifiers["RF"]  = _best_rf(asm.X_fusion, y)
        classifiers["SVM"] = _best_svm(asm.X_fusion, y)
    else:
        print("\n  [Step 2] Using default classifier hyperparams "
              "(set grid_search=True for tuned params)")

    # ── Step 3: evaluate all condition × classifier combinations ──
    print("\n  [Step 3] Running 10-fold cross-validation …\n")
    rows = []
    for condition in CONDITIONS:
        X = feature_matrices[condition]
        for clf_name, clf in classifiers.items():
            row = evaluate_condition(
                X, y, clf_name, clf,
                condition, dataset_name, n_folds
            )
            rows.append(row)
            _print_row(row)

    results_df = pd.DataFrame(rows)

    # ── Step 4: save assembler for later predict/deploy ──
    asm_path = os.path.join(MODELS_DIR, f"assembler_{dataset_name}.pkl")
    asm.save(asm_path)
    print(f"\n  Assembler saved → {asm_path}")

    return results_df


def _print_row(row: dict):
    print(f"        acc={row['accuracy']:.4f}  "
          f"prec={row['precision']:.4f}  "
          f"rec={row['recall']:.4f}  "
          f"f1={row['f1']:.4f}  "
          f"auc={row['auc_roc']:.4f}  "
          f"pr_auc={row['pr_auc']:.4f}  "
          f"fpr@95tpr={row['fpr_at_95tpr']:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience: save + load results
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_results(results_df: pd.DataFrame, dataset_name: str):
    path = os.path.join(OUTPUTS_DIR, f"results_{dataset_name}.csv")
    results_df.to_csv(path, index=False)
    print(f"  Results saved → {path}")
    return path


def load_results(dataset_name: str) -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, f"results_{dataset_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results for {dataset_name} at {path}")
    return pd.read_csv(path)
