# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# Evaluation, Visualization & Benchmark Comparison
# Abundo, Custodio, Pastoral — Mapua University 2026
#
# Produces:
#   outputs/results_summary.csv        — all metrics across all experiments
#   outputs/benchmark_comparison.csv   — vs Kulkarni et al. & Baccouche et al.
#   outputs/heatmap_<dataset>.png      — accuracy heatmap per dataset
#   outputs/pr_curve_<dataset>.png     — precision-recall curves
#   outputs/confusion_<dataset>_<clf>_<condition>.png
#   outputs/feature_importance_<dataset>.png
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier

from config import (
    ALL_HEADER_FEATURES, KULKARNI_FEATURES, EXTENDED_HEADER_FEATURES,
    CV_FOLDS, RANDOM_SEED, OUTPUTS_DIR,
)
from early_fusion import (
    FeatureAssembler, get_classifiers, fpr_at_tpr, compute_metrics,
    CONDITIONS,
)

# Published baselines from the two thesis reference papers
BASELINES = {
    "Kulkarni_et_al_2020": {
        "SpamAssassin": {"accuracy": 93.56, "method": "RF + Relief (header-only)"},
        "TREC7":        {"accuracy": None,  "method": "RF + Relief (header-only)"},
        "Nazario":      {"accuracy": None,  "method": "RF + Relief (header-only)"},
    },
    "Baccouche_et_al_2020": {
        "SpamAssassin": {"accuracy": 92.70, "method": "LSTM (body-only)"},
        "TREC7":        {"accuracy": 92.70, "method": "LSTM (body-only)"},
        "Nazario":      {"accuracy": 92.70, "method": "LSTM (body-only)"},
    },
}

CLF_DISPLAY  = {"NB": "Naive Bayes", "RF": "Random Forest", "SVM": "SVM"}
COND_DISPLAY = {
    "header_only": "Header-only",
    "body_only":   "Body-only",
    "fusion":      "Early Fusion",
}
METRIC_COLS  = ["accuracy", "precision", "recall", "f1",
                "auc_roc", "pr_auc", "fpr_at_95tpr"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Summary table
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_results_table(results_df: pd.DataFrame, dataset_name: str):
    """Pretty-print the results table for one dataset."""
    print(f"\n{'═'*85}")
    print(f"  RESULTS — {dataset_name}")
    print(f"  {'Condition':<14} {'Classifier':<13} "
          f"{'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} "
          f"{'AUC':>6} {'PR-AUC':>7} {'FPR@95':>7}")
    print(f"  {'─'*80}")
    order = {"header_only": 0, "body_only": 1, "fusion": 2}
    for _, row in results_df.sort_values(
            ["condition", "classifier"],
            key=lambda s: s.map(order) if s.name == "condition" else s
    ).iterrows():
        cond_lbl = COND_DISPLAY.get(row["condition"], row["condition"])
        clf_lbl  = CLF_DISPLAY.get(row["classifier"], row["classifier"])
        # Bold the fusion rows in the console output
        prefix = "* " if row["condition"] == "fusion" else "  "
        print(f"{prefix}{cond_lbl:<14} {clf_lbl:<13} "
              f"{row['accuracy']*100:>5.2f}% "
              f"{row['precision']*100:>5.2f}% "
              f"{row['recall']*100:>5.2f}% "
              f"{row['f1']*100:>5.2f}% "
              f"{row['auc_roc']:>6.4f} "
              f"{row['pr_auc']:>7.4f} "
              f"{row['fpr_at_95tpr']:>7.4f}")
    print(f"  (* = fusion condition)")


def save_summary(all_results: dict) -> pd.DataFrame:
    """Concat all dataset results into one summary CSV."""
    combined = pd.concat(list(all_results.values()), ignore_index=True)
    path = os.path.join(OUTPUTS_DIR, "results_summary.csv")
    combined.to_csv(path, index=False)
    print(f"\n  Full summary saved → {path}")
    return combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Benchmark comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def benchmark_comparison(all_results: dict) -> pd.DataFrame:
    """
    Build and print a table comparing our best fusion result against
    Kulkarni et al. (2020) and Baccouche et al. (2020) baselines.
    """
    rows = []
    for ds_name, results_df in all_results.items():
        # Our best fusion result (highest F1 in fusion condition)
        fusion_df = results_df[results_df["condition"] == "fusion"]
        if fusion_df.empty:
            continue
        best_idx = fusion_df["f1"].idxmax()
        best     = fusion_df.loc[best_idx]

        # Header-only best (direct Kulkarni replication)
        hdr_df   = results_df[results_df["condition"] == "header_only"]
        hdr_best = hdr_df.loc[hdr_df["f1"].idxmax()] if not hdr_df.empty else None

        # Body-only best (direct Baccouche replication)
        body_df   = results_df[results_df["condition"] == "body_only"]
        body_best = body_df.loc[body_df["f1"].idxmax()] if not body_df.empty else None

        rows.append({
            "Dataset":          ds_name,
            "Method":           "Our Fusion (Best)",
            "Classifier":       best["classifier"],
            "Accuracy%":        round(best["accuracy"]  * 100, 2),
            "Precision%":       round(best["precision"] * 100, 2),
            "Recall%":          round(best["recall"]    * 100, 2),
            "F1%":              round(best["f1"]        * 100, 2),
            "PR_AUC":           round(best["pr_auc"],          4),
            "FPR@95TPR":        round(best["fpr_at_95tpr"],    4),
        })
        if hdr_best is not None:
            rows.append({
                "Dataset":    ds_name,
                "Method":     "Header-only (ours)",
                "Classifier": hdr_best["classifier"],
                "Accuracy%":  round(hdr_best["accuracy"]  * 100, 2),
                "Precision%": round(hdr_best["precision"] * 100, 2),
                "Recall%":    round(hdr_best["recall"]    * 100, 2),
                "F1%":        round(hdr_best["f1"]        * 100, 2),
                "PR_AUC":     round(hdr_best["pr_auc"],          4),
                "FPR@95TPR":  round(hdr_best["fpr_at_95tpr"],    4),
            })
        if body_best is not None:
            rows.append({
                "Dataset":    ds_name,
                "Method":     "Body-only (ours)",
                "Classifier": body_best["classifier"],
                "Accuracy%":  round(body_best["accuracy"]  * 100, 2),
                "Precision%": round(body_best["precision"] * 100, 2),
                "Recall%":    round(body_best["recall"]    * 100, 2),
                "F1%":        round(body_best["f1"]        * 100, 2),
                "PR_AUC":     round(body_best["pr_auc"],          4),
                "FPR@95TPR":  round(body_best["fpr_at_95tpr"],    4),
            })
        # Published baselines
        for bl_name, bl_data in BASELINES.items():
            bl_acc = bl_data.get(ds_name, {}).get("accuracy")
            bl_method = bl_data.get(ds_name, {}).get("method", bl_name)
            if bl_acc is not None:
                rows.append({
                    "Dataset":    ds_name,
                    "Method":     f"{bl_name} [{bl_method}]",
                    "Classifier": "—",
                    "Accuracy%":  bl_acc,
                    "Precision%": "—", "Recall%": "—",
                    "F1%":        "—", "PR_AUC":  "—",
                    "FPR@95TPR":  "—",
                })

    comp_df = pd.DataFrame(rows)
    path    = os.path.join(OUTPUTS_DIR, "benchmark_comparison.csv")
    comp_df.to_csv(path, index=False)

    print(f"\n{'═'*85}")
    print("  BENCHMARK COMPARISON")
    print(f"{'═'*85}")
    for ds_name in comp_df["Dataset"].unique():
        print(f"\n  {ds_name}:")
        sub = comp_df[comp_df["Dataset"] == ds_name]
        print(f"  {'Method':<45} {'Clf':<12} {'Acc%':>6} {'F1%':>6} {'PR-AUC':>7}")
        print(f"  {'─'*80}")
        for _, r in sub.iterrows():
            marker = "◆ " if "Fusion" in str(r["Method"]) else "  "
            print(f"  {marker}{str(r['Method']):<43} "
                  f"{str(r['Classifier']):<12} "
                  f"{str(r['Accuracy%']):>6} "
                  f"{str(r['F1%']):>6} "
                  f"{str(r['PR_AUC']):>7}")
    print(f"\n  Saved → {path}")
    return comp_df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Heatmap: accuracy by condition × classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_accuracy_heatmap(results_df: pd.DataFrame, dataset_name: str):
    """
    Heat-map of F1 score (%) for each condition × classifier.
    Mirrors Tables III/IV/V in Kulkarni et al.
    """
    pivot = (
        results_df
        .assign(f1_pct=lambda d: d["f1"] * 100)
        .pivot(index="condition", columns="classifier", values="f1_pct")
    )
    # Reorder rows
    row_order = [c for c in CONDITIONS if c in pivot.index]
    pivot     = pivot.loc[row_order]
    pivot.index = [COND_DISPLAY.get(i, i) for i in pivot.index]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.heatmap(
        pivot, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, vmin=50, vmax=100,
        cbar_kws={"label": "F1 Score (%)"},
    )
    ax.set_title(f"F1 Score (%) — {dataset_name}\n"
                 f"[◆ = Early Fusion condition]", fontsize=11)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Condition")
    # Mark fusion row with a rectangle
    fusion_idx = list(pivot.index).index("Early Fusion") if "Early Fusion" in pivot.index else -1
    if fusion_idx >= 0:
        ax.add_patch(plt.Rectangle(
            (0, fusion_idx), len(pivot.columns), 1,
            fill=False, edgecolor="black", lw=2, clip_on=False,
        ))
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"heatmap_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Heatmap saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Precision–Recall curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_pr_curves(df: pd.DataFrame, dataset_name: str):
    """
    Precision-Recall curves for each classifier under fusion condition.
    Compares all three conditions on the same plot.
    """
    asm = FeatureAssembler(use_discriminative_vocab=True)
    asm.fit_transform(df)
    y = asm.y

    feature_matrices = {
        "header_only": asm.X_header,
        "body_only":   asm.X_body,
        "fusion":      asm.X_fusion,
    }

    clfs = get_classifiers()
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                           random_state=RANDOM_SEED)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors    = {"header_only": "#2196F3", "body_only": "#FF9800", "fusion": "#4CAF50"}
    ls_map    = {"header_only": "--", "body_only": "-.", "fusion": "-"}

    for ax, (clf_name, clf) in zip(axes, clfs.items()):
        for condition, X in feature_matrices.items():
            y_prob = cross_val_predict(
                clf, X, y, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            prec, rec, _ = precision_recall_curve(y, y_prob)
            pr_auc = np.trapz(prec[::-1], rec[::-1])
            ax.plot(rec, prec,
                    color=colors[condition],
                    linestyle=ls_map[condition],
                    linewidth=1.8,
                    label=f"{COND_DISPLAY[condition]} (AP={pr_auc:.3f})")

        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.set_title(f"{CLF_DISPLAY.get(clf_name, clf_name)}", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.axhline(y=0.9, color="grey", linestyle=":", alpha=0.4)

    fig.suptitle(f"Precision–Recall Curves — {dataset_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"pr_curves_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  PR curves saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Confusion matrices for fusion + best classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_confusion_matrices(df: pd.DataFrame, dataset_name: str,
                             results_df: pd.DataFrame):
    """Plot confusion matrices for all classifiers under the fusion condition."""
    asm = FeatureAssembler(use_discriminative_vocab=True)
    asm.fit_transform(df)
    y  = asm.y
    X  = asm.X_fusion
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_SEED)

    clfs = get_classifiers()
    fig, axes = plt.subplots(1, len(clfs), figsize=(5 * len(clfs), 4))

    for ax, (clf_name, clf) in zip(axes, clfs.items()):
        y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict",
                                   n_jobs=-1)
        cm = confusion_matrix(y, y_pred)
        # Normalize
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(
            cm_norm, display_labels=["Ham/Legit", "Phishing"]
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues",
                  values_format=".3f")
        ax.set_title(f"{CLF_DISPLAY.get(clf_name, clf_name)}\n"
                     f"(Early Fusion)", fontsize=10)

    fig.suptitle(f"Confusion Matrices — {dataset_name}", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"confusion_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrices saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Feature importance (RF fusion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_feature_importance(df: pd.DataFrame, dataset_name: str):
    """
    Train RF on fusion features; plot top-30 feature importances.
    Labels header features by name, body features as 'body: <ngram>'.
    """
    asm = FeatureAssembler(use_discriminative_vocab=True)
    asm.fit_transform(df)

    # Feature names: header first, then body
    header_names = ALL_HEADER_FEATURES
    body_names   = ["body: " + n for n in asm.body_featurizer.get_feature_names()]
    all_names    = header_names + body_names
    # Pad if lengths differ
    n_cols = asm.X_fusion.shape[1]
    all_names = (all_names + [f"feat_{i}" for i in range(n_cols)])[:n_cols]

    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(asm.X_fusion, asm.y)
    importances = rf.feature_importances_

    top_k = 30
    top_idx  = np.argsort(importances)[-top_k:][::-1]
    top_imp  = importances[top_idx]
    top_names = [all_names[i] if i < len(all_names) else f"feat_{i}"
                 for i in top_idx]

    # Color code: header vs body
    colors = [
        "#2196F3" if not n.startswith("body:") else "#FF9800"
        for n in top_names
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_k), top_imp[::-1],
                   color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Feature Importance (Gini)", fontsize=10)
    ax.set_title(f"Top {top_k} Features — RF Fusion Model — {dataset_name}",
                 fontsize=11)

    # Legend
    p1 = mpatches.Patch(color="#2196F3", label="Header feature")
    p2 = mpatches.Patch(color="#FF9800", label="Body (TF-IDF) feature")
    ax.legend(handles=[p1, p2], fontsize=9, loc="lower right")

    # Mark Kulkarni vs extended features
    for i, name in enumerate(top_names[::-1]):
        if name in KULKARNI_FEATURES:
            ax.get_yticklabels()[i].set_color("#0D47A1")
        elif name.lstrip() in EXTENDED_HEADER_FEATURES:
            ax.get_yticklabels()[i].set_color("#E65100")

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"feature_importance_{dataset_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Condition gain chart — does fusion actually beat single-source?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_condition_comparison(all_results: dict):
    """
    Grouped bar chart: F1 for header_only / body_only / fusion
    per classifier, for each dataset.  Directly answers RQ1.
    """
    n_datasets = len(all_results)
    if n_datasets == 0:
        return

    clfs = ["NB", "RF", "SVM"]
    fig, axes = plt.subplots(1, n_datasets,
                             figsize=(5 * n_datasets, 5), sharey=True)
    if n_datasets == 1:
        axes = [axes]

    bar_colors = {
        "header_only": "#2196F3",
        "body_only":   "#FF9800",
        "fusion":      "#4CAF50",
    }
    width = 0.25

    for ax, (ds_name, res_df) in zip(axes, all_results.items()):
        x = np.arange(len(clfs))
        for i, cond in enumerate(CONDITIONS):
            vals = []
            for clf in clfs:
                row = res_df[(res_df["condition"] == cond) &
                             (res_df["classifier"] == clf)]
                vals.append(row["f1"].values[0] * 100 if len(row) else 0)
            bars = ax.bar(x + (i - 1) * width, vals, width,
                          label=COND_DISPLAY[cond],
                          color=bar_colors[cond], alpha=0.85,
                          edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + 0.3,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([CLF_DISPLAY.get(c, c) for c in clfs], fontsize=9)
        ax.set_ylabel("F1 Score (%)", fontsize=10)
        ax.set_title(ds_name, fontsize=11)
        ax.set_ylim(0, 108)
        ax.legend(fontsize=8)
        ax.axhline(93.56, color="#0D47A1", linewidth=1, linestyle="--",
                   alpha=0.6, label="Kulkarni baseline")
        ax.axhline(92.70, color="#BF360C", linewidth=1, linestyle="-.",
                   alpha=0.6, label="Baccouche baseline")

    fig.suptitle("F1 Score by Condition and Classifier\n"
                 "(-- Kulkarni baseline  -·- Baccouche baseline)",
                 fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "condition_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Condition comparison chart saved → {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Master evaluate function (called from main runner)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def full_evaluation(all_results: dict, all_dfs: dict,
                    skip_plots: bool = False):
    """
    Run all evaluation and visualization steps.

    Args:
        all_results : {dataset_name: results_df}
        all_dfs     : {dataset_name: raw_feature_df}
        skip_plots  : set True to skip the re-fit plots (faster for CI/CD)
    """
    print(f"\n{'━'*65}")
    print("  FULL EVALUATION & VISUALIZATION")
    print(f"{'━'*65}")

    # Per-dataset tables and plots
    for ds_name, results_df in all_results.items():
        print_results_table(results_df, ds_name)

        if not skip_plots and ds_name in all_dfs:
            df = all_dfs[ds_name]
            plot_accuracy_heatmap(results_df, ds_name)
            plot_confusion_matrices(df, ds_name, results_df)
            plot_feature_importance(df, ds_name)
            # PR curves require re-fitting — skip in fast mode
            # plot_pr_curves(df, ds_name)   # uncomment for full run

    # Cross-dataset summary
    summary_df = save_summary(all_results)
    comp_df    = benchmark_comparison(all_results)

    if not skip_plots:
        plot_condition_comparison(all_results)

    return summary_df, comp_df
