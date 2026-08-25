# A Human-in-the-Loop Validation for AI-Driven Intrusion Detection Threat Classification

MSc Artificial Intelligence thesis project by Sehrish Shahid (12011153), Faculty of IT and Creative Media, Bahrain Polytechnic, 2026.

This repository contains the implementation and experimental evidence for a human-in-the-loop (HITL) intrusion-detection pipeline evaluated on the corrected CSE-CIC-IDS2018 dataset. The pipeline trains an XGBoost multiclass classifier, calibrates its probabilities, routes the least-confident 5% of validation predictions for simulated analyst review and retrains the classifier using the reviewed records.

SHAP is used for explanation and diagnostic analysis within the confidence-routed subset. In the reported implementation, SHAP concentration does not independently add high-confidence predictions to the review queue.

## Repository contents

```text
.
├── HITL_IDS_Pipeline_v3_realistic_reproduction(1).ipynb
├── test_pipeline.py
├── requirements.txt
├── README.md
└── ARTEFACTS.md
```

The notebook contains 22 numbered sections and 27 code cells. In the uploaded executed version, all 27 code cells were run in sequence without recorded execution errors.

Large datasets, caches, trained models and generated results are not stored in the repository. They are created locally by the notebook.

## Dataset

The project uses the corrected CSE-CIC-IDS2018 release described by Liu et al. The corrected release is required because the original distribution contains documented labelling problems.

Download the corrected files from:

<https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html>

Place the ten daily CSV files in:

```text
data/CSECICIDS2018_improved/
```

The notebook's population scan reports 63,195,145 records across 25 original classes. Four classes containing fewer than 50 records are excluded, leaving 21 classes and a retained population of 63,195,060 records.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The executed notebook explicitly reports XGBoost 3.2.0 and SHAP 0.51.0. It also reports that `FrozenEstimator` is available. The embedded figure metadata indicates Matplotlib 3.11.0. For exact environment reproduction, the Matplotlib version in `requirements.txt` should be aligned with the executed environment before final archival.

## Primary experimental configuration

The uploaded executed notebook represents the primary realistic run with the following settings:

| Parameter | Value |
|---|---:|
| Sampling profile | Unbalanced |
| Sampling fraction | 0.55% per class |
| Maximum per class | None |
| Minimum per retained class | 50 |
| Minimum population class size | 50 |
| Attempted-label policy | Keep |
| Train / validation / test proportions | 60% / 20% / 20% |
| Analyst review budget | 5% |
| SHAP concentration threshold | 0.15 |
| Analyst error rate | 10% |
| Primary analyst model | Realistic |
| Noise-experiment analyst model | Realistic |
| Cross-validation folds | 5 |
| Replication seeds per noise condition | 5 |
| Random seed | 42 |

The XGBoost model uses 300 estimators, maximum depth 8, learning rate 0.1, subsample 0.8, column subsample 0.8, histogram tree construction and balanced sample weights.

## Running the primary experiment

1. Open `HITL_IDS_Pipeline_v3_realistic_reproduction(1).ipynb`.
2. Select the Python environment containing the dependencies above.
3. Confirm the following settings in Section 2:

```python
DATA_DIR = "./data/CSECICIDS2018_improved"
RESULTS_ROOT = "./results_reproduction"
SAMPLING_PROFILE = "unbalanced"
ANALYST_ERROR_MODEL = "realistic"
NOISE_ANALYST_MODEL = "realistic"
ROUTE_BUDGET = 0.05
CONCENTRATION_THRESHOLD = 0.15
ANALYST_ERROR_RATE = 0.10
SEED = 42
```

4. Restart the kernel.
5. Run all cells from the beginning.

The population counts and sampled dataset are cached. The first execution is therefore slower than later executions. The multi-seed degradation analysis in Section 19 is the most computationally expensive section because it trains 80 models.

## Output structure

The notebook writes shared caches and timestamped run directories under `results_reproduction/`:

```text
results_reproduction/
├── population_class_counts.csv
├── sampled_unbalanced.csv
└── run_unbalanced_<timestamp>/
    ├── baseline_xgb.json
    ├── calibrated_model.pkl
    ├── comparison_ai_vs_hitl.csv
    ├── control_augmentation.json
    ├── fig1_confusion_baseline.png
    ├── fig2_calibration.png
    ├── fig3_shap_global.png
    ├── fig4_budget_tradeoff.png
    ├── fig5_noise_multiseed.png
    ├── hitl_xgb.json
    ├── noise_multiseed.csv
    ├── preprocessing.pkl
    ├── results_summary.json
    ├── results_table.md
    ├── sample_representativeness.csv
    ├── shap_stability.csv
    └── threshold_sweep.csv
```

The executed notebook saved the reported run as:

```text
run_unbalanced_2026-08-23_21-03-47
```

## Reported primary results

After deduplication, the experimental dataset contained 320,320 records, 80 features and 21 classes. The split contained 192,192 training records, 64,064 validation records and 64,064 test records.

### Decision gate and analyst review

| Measure | Executed result |
|---|---:|
| Validation predictions routed | 3,203 of 64,064 |
| Routed proportion | 5.00% |
| Low-confidence predictions | 3,203 |
| Weak-explanation flags within that subset | 1,384 |
| Accuracy among auto-accepted predictions | 1.0000 |
| Accuracy among routed predictions | 0.9953 |
| Analyst approvals | 3,188 |
| Analyst corrections | 9 |
| Analyst rejections | 5 |
| Missed analyst correction | 1 |
| Labels correctly changed | 14 |

The realistic analyst simulation left one existing model error uncorrected. It did not corrupt an originally correct prediction.

### Test-set comparison

| Metric | AI-only | HITL | Difference |
|---|---:|---:|---:|
| Accuracy | 0.9994 | 0.9995 | +0.0001 |
| Macro precision | 0.9292 | 0.9367 | +0.0076 |
| Macro recall | 0.9524 | 0.9619 | +0.0095 |
| Macro F1 | 0.9319 | 0.9421 | +0.0102 |
| Weighted F1 | 0.9995 | 0.9995 | +0.0001 |
| Matthews correlation coefficient | 0.9953 | 0.9960 | +0.0007 |

The augmentation control decomposed the macro-F1 change as follows:

| Configuration | Macro F1 |
|---|---:|
| AI-only | 0.9319 |
| Added routed records without corrected labels | 0.9389 |
| Added routed records with analyst-reviewed labels | 0.9421 |

The total increase was +0.0102. Added training data accounted for +0.0070, while the simulated human-label contribution accounted for +0.0032.

### Cross-validation and explanation stability

Five-fold cross-validation produced an AI-only macro F1 of 0.9480 +/- 0.0106 and a HITL macro F1 of 0.9476 +/- 0.0128. These fold means are descriptive and do not show a cross-validation improvement for HITL.

SHAP stability across folds was:

| Measure | Value |
|---|---:|
| Mean Spearman correlation | 0.9523 |
| Minimum Spearman correlation | 0.9349 |
| Mean Kendall correlation | 0.8407 |
| Mean top-10 Jaccard overlap | 0.5678 |

## Label-noise experiments

Section 19 evaluates symmetric and attack-to-benign label noise at 0%, 7.5%, 20% and 40%, using five seeds per condition. Both the AI-only and HITL models are evaluated on the unchanged clean test partition.

The executed notebook uses `NOISE_ANALYST_MODEL = "realistic"`. Mean macro-F1 results are stored in `noise_multiseed.csv` and plotted in `fig5_noise_multiseed.png`.

## Verification tests

The repository contains 23 synthetic-data tests:

```bash
pytest test_pipeline.py -v
```

The tests cover sampling targets, deduplication, partition leakage, stratification, training-only scaling, review-budget routing, SHAP concentration, analyst decisions, label-noise injection and deterministic model training.

The present test suite mirrors functions from the notebook rather than importing a shared implementation. Consequently, the notebook and tests should be reviewed together whenever either implementation changes.

## Scope and limitations

The executed evidence in this notebook is limited to:

- One corrected intrusion-detection benchmark.
- One executed unbalanced sampling profile.
- One classifier family.
- One simulated analyst-review cycle.
- Simulated rather than human-provided labels.
- A realistic analyst-error model for the primary and noise experiments.

The results should therefore be interpreted as evidence for this experimental configuration, not as proof that the same improvements will generalise to other datasets, time periods, organisations, analyst populations or classifier families.

The notebook contains a configurable pessimistic analyst mode, but the uploaded realistic notebook is not an executed pessimistic reproduction. Pessimistic sensitivity results should only be reported from a separately validated implementation and complete clean execution.

## Reproducibility notes

- Stochastic operations use `SEED = 42` unless a multi-seed experiment explicitly supplies seeds 42 to 46.
- The population scan and sampled dataset are cached under `results_reproduction/`.
- Each ordinary execution creates a timestamped output directory.
- `results_summary.json` stores the principal dataset, sampling, routing, seed and evaluation settings with the recorded metrics.
- The exact analyst-model settings and full model hyperparameters should also be archived with the final submitted run.

## Citation

```bibtex
@mastersthesis{shahid2026hitl,
  title  = {A Human-in-the-Loop Validation for AI-Driven Intrusion
            Detection Threat Classification},
  author = {Shahid, Sehrish},
  school = {Bahrain Polytechnic},
  year   = {2026},
  type   = {{MSc} thesis}
}
```

## Acknowledgement of AI tool use

Generative AI tools were used as productivity aids for code drafting, debugging assistance and literature-search support in accordance with the programme requirements. All generated content was reviewed, tested and adapted by the author. The thesis contains the complete declaration of AI-assisted work.
