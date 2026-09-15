#!/usr/bin/env python3
# run_pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# FULL EARLY FUSION PIPELINE — Smoke Test & Real Dataset Runner
# Abundo, Custodio, Pastoral — Mapua University 2026
#
# Usage (smoke test no real data needed):
#   python run_pipeline.py --smoke
#
# Usage (real data place emails in data/raw/<Dataset>/{spam,ham}/):
#   python run_pipeline.py
#   python run_pipeline.py --datasets SpamAssassin Nazario
#   python run_pipeline.py --grid-search          
#   python run_pipeline.py --rebuild              
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


from config import DATASETS, OUTPUTS_DIR, MODELS_DIR, RANDOM_SEED
from data_loader import load_all_datasets, make_synthetic_dataset
from early_fusion import run_experiment, save_results, CONDITIONS
from evaluate import full_evaluation


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(
        description="Early Fusion Phishing Detection Pipeline"
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Run on 600 synthetic emails (no real data required). "
             "Perfect for verifying the pipeline end-to-end."
    )
    p.add_argument(
        "--datasets", nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which real datasets to run (default: all configured)."
    )
    p.add_argument(
        "--grid-search", action="store_true",
        help="Grid-search RF and SVM hyperparams (slower, better results)."
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Rebuild feature CSVs from raw email files (ignore cache)."
    )
    p.add_argument(
        "--folds", type=int, default=10,
        help="Number of cross-validation folds (default 10; use 3 for speed)."
    )
    p.add_argument(
        "--skip-plots", action="store_true",
        help="Skip matplotlib plots (useful in headless CI environments)."
    )
    return p.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# testings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_smoke_test(n_folds: int = 5,
                   grid_search: bool = False,
                   skip_plots: bool = False):
    """
    End-to-end pipeline on synthetic data.
    Proves every module works correctly without needing real email datasets.
    Uses 3-fold CV and 600 emails to finish quickly.
    """
    print("\n" + "═" * 65)
    print("  SMOKE TEST — Synthetic Email Dataset")
    print("  600 emails (300 spam + 300 ham) | 3-fold CV")
    print("═" * 65)

    # ── Build synthetic datasets representing our three real datasets ──
    smoke_datasets = {
        "Synthetic_SpamAssassin": make_synthetic_dataset(
            n_spam=200, n_ham=200, seed=RANDOM_SEED),
        "Synthetic_Nazario":      make_synthetic_dataset(
            n_spam=150, n_ham=150, seed=RANDOM_SEED + 1),
        "Synthetic_TREC7":        make_synthetic_dataset(
            n_spam=300, n_ham=300, seed=RANDOM_SEED + 2),
    }

    all_results = {}
    all_dfs     = {}

    for ds_name, df in smoke_datasets.items():
        t0 = time.time()
        results_df = run_experiment(
            dataset_name=ds_name,
            df=df,
            grid_search=grid_search,
            n_folds=n_folds,
        )
        elapsed = time.time() - t0
        print(f"\n  [{ds_name}] ✓ Completed in {elapsed:.1f}s")

        save_results(results_df, ds_name)
        all_results[ds_name] = results_df
        all_dfs[ds_name]     = df

    # ── Evaluate ──
    summary_df, comp_df = full_evaluation(
        all_results, all_dfs, skip_plots=skip_plots
    )

    _print_final_summary(all_results, mode="smoke")
    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Real data run
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_real(dataset_names: list,
             n_folds: int = 10,
             grid_search: bool = False,
             force_rebuild: bool = False,
             skip_plots: bool = False):
    """
    Full pipeline on real email datasets.
    Place raw .eml files in data/raw/<DatasetName>/{spam,ham}/.
    """
    print("\n" + "═" * 65)
    print(f"  FULL PIPELINE — Real Datasets: {dataset_names}")
    print(f"  CV folds={n_folds}  grid_search={grid_search}")
    print("═" * 65)

    # ── Load / build feature CSVs ──
    print("\n[Phase 1] Loading datasets …")
    all_dfs_loaded = load_all_datasets(force_rebuild=force_rebuild)

    # Filter to requested datasets
    all_dfs = {k: v for k, v in all_dfs_loaded.items()
               if k in dataset_names and not v.empty}

    if not all_dfs:
        print("\n  ERROR: No datasets loaded.")
        print("  Place raw emails in data/raw/<DatasetName>/{spam,ham}/")
        print("  Or run with --smoke to test on synthetic data.\n")
        sys.exit(1)

    # ── Run experiments ──
    print(f"\n[Phase 2] Running experiments ({len(all_dfs)} datasets) …")
    all_results = {}

    for ds_name, df in all_dfs.items():
        t0 = time.time()
        results_df = run_experiment(
            dataset_name=ds_name,
            df=df,
            grid_search=grid_search,
            n_folds=n_folds,
        )
        elapsed = time.time() - t0
        print(f"\n  [{ds_name}] ✓ Completed in {elapsed:.1f}s")
        save_results(results_df, ds_name)
        all_results[ds_name] = results_df

    # ── Evaluate ──
    print("\n[Phase 3] Evaluation & visualization …")
    summary_df, comp_df = full_evaluation(
        all_results, all_dfs, skip_plots=skip_plots
    )

    _print_final_summary(all_results, mode="real")
    return all_results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Final summary printout
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _print_final_summary(all_results: dict, mode: str = "real"):
    print(f"\n{'═'*65}")
    print("  PIPELINE COMPLETE — KEY FINDINGS")
    print(f"{'═'*65}\n")

    for ds_name, res_df in all_results.items():
        # Best fusion result
        fusion_df = res_df[res_df["condition"] == "fusion"]
        if fusion_df.empty:
            continue
        best      = fusion_df.loc[fusion_df["f1"].idxmax()]
        hdr_df    = res_df[res_df["condition"] == "header_only"]
        body_df   = res_df[res_df["condition"] == "body_only"]
        hdr_best  = hdr_df.loc[hdr_df["f1"].idxmax()]  if not hdr_df.empty  else None
        body_best = body_df.loc[body_df["f1"].idxmax()] if not body_df.empty else None

        print(f"  {ds_name}:")
        print(f"    Best fusion ({best['classifier']}):     "
              f"F1={best['f1']*100:.2f}%  "
              f"Prec={best['precision']*100:.2f}%  "
              f"Rec={best['recall']*100:.2f}%  "
              f"FPR@95TPR={best['fpr_at_95tpr']:.4f}")
        if hdr_best is not None:
            gain_hdr = (best["f1"] - hdr_best["f1"]) * 100
            print(f"    Header-only ({hdr_best['classifier']}):  "
                  f"F1={hdr_best['f1']*100:.2f}%  "
                  f"[Fusion gain: {gain_hdr:+.2f}pp]")
        if body_best is not None:
            gain_body = (best["f1"] - body_best["f1"]) * 100
            print(f"    Body-only ({body_best['classifier']}):    "
                  f"F1={body_best['f1']*100:.2f}%  "
                  f"[Fusion gain: {gain_body:+.2f}pp]")
        # Kulkarni comparison (SpamAssassin only)
        if "SpamAssassin" in ds_name:
            kulkarni_acc = 93.56
            our_acc = best["accuracy"] * 100
            diff = our_acc - kulkarni_acc
            print(f"    Kulkarni baseline: Acc=93.56%  "
                  f"[Our fusion: Acc={our_acc:.2f}%  diff={diff:+.2f}pp]")
        print()

    print(f"  Outputs saved to: {OUTPUTS_DIR}/")
    print(f"  Models saved to:  {MODELS_DIR}/\n")
    if mode == "smoke":
        print("  NOTE: Smoke test used SYNTHETIC data.")
        print("  Results demonstrate pipeline correctness, not real accuracy.")
        print("  Run without --smoke flag on real email datasets for thesis.\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "═" * 65)
    print("  Early Fusion Phishing Detection")
    print("  Abundo · Custodio · Pastoral — Mapua University 2026")
    print("  Kulkarni et al. (2020) + Baccouche et al. (2020)")
    print("=" * 65)

    if args.smoke:
        run_smoke_test(
            n_folds=3 if args.folds == 10 else args.folds,
            grid_search=args.grid_search,
            skip_plots=args.skip_plots,
        )
    else:
        run_real(
            dataset_names=args.datasets,
            n_folds=args.folds,
            grid_search=args.grid_search,
            force_rebuild=args.rebuild,
            skip_plots=args.skip_plots,
        )
