# Early Fusion Phishing Email Detection
**Abundo · Custodio · Pastoral — Mapua University 2026**

This project detects phishing and spam emails by combining email header features with body text features and comparing the results against using either source alone.

| Mode | What it uses | Based on |
|---|---|---|
| **Header-only** | 33 email header features (SPF, Reply-To, Message-ID anomalies, etc.) | Kulkarni et al. (2020) |
| **Body-only** | TF-IDF features from email body text | Baccouche et al. (2020) |
| **Early Fusion** | Header and body features combined into one vector | This thesis |

Each mode is tested with three classifiers (Naive Bayes, Random Forest, SVM) across three datasets (SpamAssassin, Nazario, TREC7) using 10-fold stratified cross-validation. Metrics reported: Accuracy, Precision, Recall, F1, AUC-ROC, PR-AUC, and FPR@95TPR.

## How It Works

```
Raw .eml file
     |
     |---> Branch A (Body)    -> Text cleaning -> TF-IDF (up to 5,000 n-gram features)
     |                                                         |
     +---> Branch B (Headers) -> 33 header anomaly features --+
                                                              |
                                         +--------------------+
                                         |
                             Early Fusion Feature Vector
                                         |
                             +-----------+-----------+
                             |           |           |
                       Naive Bayes  Random Forest   SVM
```

## Project Structure

```
|- config.py               # Dataset paths, feature names, hyperparameters
|- data_loader.py          # Reads raw .eml files, builds feature CSVs
|- branch_a_body.py        # Branch A: body text preprocessing and TF-IDF
|- branch_b_headers.py     # Branch B: email header feature extraction
|- early_fusion.py         # Feature assembly, CV evaluation, experiment runner
|- evaluate.py             # Metrics, plots, benchmark comparison
|- run_pipeline.py         # Step 1: cross-validation pipeline
|- train_eval_infer.py     # Step 2: train/validate/test split and inference
|- data/
|   +- raw/                # Place raw .eml files here
|- outputs/                # Generated CSVs and charts (auto-created)
+- models/                 # Saved model files (auto-created)
```

## Datasets

Download the datasets and place them in the correct folder structure.

> 📁 **[Download Datasets from Google Drive](#)** <- Link to be added

```
data/raw/
|- SpamAssassin/
|   |- spam/
|   +- ham/
|- Nazario/
|   |- spam/
|   +- ham/
+- TREC7/
    |- spam/
    +- ham/
```

## Requirements

- Python 3.8 or higher

Install all required packages:

```bash
pip install scikit-learn pandas numpy scipy matplotlib seaborn
```

If you get a permissions error:

```bash
pip install --user scikit-learn pandas numpy scipy matplotlib seaborn
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies (see Requirements above).

3. Download the datasets and place them in `data/raw/` following the folder structure above.

## Usage

There are two scripts. Run them in order.

### Step 1: Cross-Validation (run_pipeline.py)

Evaluates all three modes side by side using 10-fold cross-validation. Run this first.

Run on all datasets:

```bash
python run_pipeline.py
```

Run on specific datasets:

```bash
python run_pipeline.py --datasets SpamAssassin Nazario
```

Run with hyperparameter tuning (slower, recommended for final thesis run):

```bash
python run_pipeline.py --grid-search
```

Reprocess raw emails instead of using cached CSVs:

```bash
python run_pipeline.py --rebuild
```

Available flags:

| Flag | Description |
|---|---|
| `--datasets` | Datasets to run (e.g. `SpamAssassin Nazario TREC7`) |
| `--grid-search` | Tune RF and SVM hyperparameters |
| `--rebuild` | Ignore cached CSVs and reprocess raw email files |
| `--folds N` | Number of CV folds (default: 10) |
| `--skip-plots` | Skip chart generation |

At the end of the run the terminal prints a summary showing F1, Precision, and Recall for each mode and classifier per dataset.

```
SpamAssassin:
  Best fusion (RF):   F1=97.45%  Prec=96.80%  Rec=98.10%
  Header-only (RF):   F1=93.56%
  Body-only (SVM):    F1=94.20%
  Kulkarni baseline:  Acc=93.56%  Our fusion: Acc=97.45%
```

### Step 2: Train, Validate, and Test (train_eval_infer.py)

Trains final models using a 70/15/15 split. The test set is never seen during training. Use this after Step 1 to produce the final numbers for the thesis and to save models for inference.

| Split | Size | Purpose |
|---|---|---|
| Train | 70% | Fit the assembler and classifiers |
| Validate | 15% | Pick the best classifier per mode |
| Test | 15% | Final evaluation |

Run on all datasets:

```bash
python train_eval_infer.py
```

Run on specific datasets:

```bash
python train_eval_infer.py --datasets SpamAssassin Nazario
```

Run with hyperparameter tuning:

```bash
python train_eval_infer.py --grid-search
```

Reprocess raw emails instead of using cached CSVs:

```bash
python train_eval_infer.py --rebuild
```

Classify a single email using a saved model:

```bash
python train_eval_infer.py --infer path/to/email.eml
```

To use a model trained on a specific dataset:

```bash
python train_eval_infer.py --infer path/to/email.eml --infer-dataset Nazario
```

The default model used for inference is SpamAssassin + RF + fusion. Output looks like:

```
  Email:      suspicious_email.eml
  Verdict   : PHISHING
  Confidence: 94.72%
  Model     : RF / fusion
```

Available flags:

| Flag | Description |
|---|---|
| `--datasets` | Datasets to train on (e.g. `SpamAssassin Nazario TREC7`) |
| `--grid-search` | Tune RF and SVM hyperparameters before training |
| `--rebuild` | Rebuild feature CSVs from raw email files |
| `--infer <path>` | Classify a single .eml file using a saved model |
| `--infer-dataset` | Which saved model to use for --infer (default: SpamAssassin) |

## Outputs

### From run_pipeline.py (Step 1)

| Location | Contents |
|---|---|
| `outputs/results_summary.csv` | All metrics across every mode, classifier, and dataset |
| `outputs/results_<dataset>.csv` | Metrics for one dataset across all modes and classifiers |
| `outputs/benchmark_comparison.csv` | Results compared against Kulkarni et al. and Baccouche et al. baselines |
| `outputs/condition_comparison.png` | Bar chart of F1 scores: header-only vs body-only vs early fusion |
| `outputs/heatmap_<dataset>.png` | Accuracy heatmap per dataset |
| `outputs/confusion_<dataset>.png` | Confusion matrices for the fusion condition |
| `outputs/feature_importance_<dataset>.png` | Top 30 feature importances from the RF model |

### From train_eval_infer.py (Step 2)

| Location | Contents |
|---|---|
| `outputs/val_results_<dataset>.csv` | Validation set metrics for all modes and classifiers |
| `outputs/test_results_<dataset>.csv` | Final held-out test results |
| `models/assembler_<dataset>.pkl` | Fitted feature assembler |
| `models/models_<dataset>.pkl` | Fitted classifiers for all modes (used by --infer) |

## References

- Kulkarni, A. et al. (2020). Email header analysis for spam detection using Relief and Random Forest.
- Baccouche, A. et al. (2020). Spam detection with LSTM-based deep learning on email body text.
