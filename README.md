# A Human-in-the-Loop Validation for AI-Driven Intrusion Detection Threat Classification

MSc Artificial Intelligence thesis project by Sehrish Shahid (12011153), Faculty of IT and Creative Media, Bahrain Polytechnic, 2026.

## Project overview

This repository contains the implementation and experimental evidence for a human-in-the-loop (HITL) intrusion-detection framework evaluated on the corrected CSE-CIC-IDS2018 dataset.

The pipeline:

1. Samples the corrected intrusion-detection dataset.
2. Preprocesses and partitions the records.
3. Trains an XGBoost multiclass classifier.
4. Calibrates the classifier probabilities using isotonic calibration.
5. Routes the least-confident 5% of validation predictions for analyst review.
6. Simulates analyst adjudication.
7. Retrains the classifier using the reviewed records.
8. Compares AI-only and HITL performance.
9. Evaluates robustness under label-noise conditions.
10. Examines SHAP explanation stability.

Two simulated analyst-error models are provided:

* **Realistic model:** Used for the primary thesis results. Analyst errors can occur only when the original model prediction is incorrect and genuine judgement is required.
* **Pessimistic model:** Used as a worst-case sensitivity analysis. Errors may also corrupt predictions that were originally correct by assigning a uniformly selected alternative class.

The pessimistic model is intended to bound potential harm. It should not be interpreted as a representative model of actual SOC analyst behaviour.

## Repository contents

```text
.
├── HITL_IDS_Pipeline_v3_realistic_reproduction(1).ipynb
├── HITL_IDS_Pipeline_v3_pessimistic_reproduction.ipynb
├── test_pipeline.py
├── requirements.txt
├── README.md
└── ARTEFACTS.md
```

### Notebook purposes

| Notebook                                               | Purpose                                                                   |
| ------------------------------------------------------ | ------------------------------------------------------------------------- |
| `HITL_IDS_Pipeline_v3_realistic_reproduction(1).ipynb` | Primary experiment using the realistic analyst-error model                |
| `HITL_IDS_Pipeline_v3_pessimistic_reproduction.ipynb`  | Worst-case sensitivity analysis using the pessimistic analyst-error model |

Both notebooks contain executed outputs. The realistic notebook contains one completed primary unbalanced run. The pessimistic notebook contains the corresponding pessimistic sensitivity run.

## Dataset

The project uses the corrected CSE-CIC-IDS2018 release described by Liu et al. The corrected release is required because the original dataset distribution contains documented labelling problems.

The corrected dataset can be obtained from:

https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html

Place the ten daily CSV files in:

```text
data/CSECICIDS2018_improved/
```

The complete population contains:

* 63,195,145 records.
* 25 original classes.
* Approximately 35 GB of CSV data.

Four classes containing fewer than 50 population records are excluded because they do not contain enough examples for the stratified evaluation procedure.

After exclusion, the retained population contains:

* 63,195,060 records.
* 21 classes.

## Environment

Python 3.11 is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

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

The executed notebooks report:

* XGBoost 3.2.0.
* SHAP 0.51.0.
* Availability of `FrozenEstimator` for probability calibration.

Minor visual differences may occur if figures are regenerated using a different Matplotlib release.

## Shared experimental configuration

Both notebooks use the same dataset, sample, partitioning method, routing budget and classifier configuration. They differ in the simulated analyst-error model.

| Parameter                                  |                     Value |
| ------------------------------------------ | ------------------------: |
| Dataset                                    | Corrected CSE-CIC-IDS2018 |
| Sampling profile                           |                Unbalanced |
| Sampling fraction                          |           0.55% per class |
| Maximum records per class                  |                      None |
| Minimum sampled records per retained class |                        50 |
| Minimum population class size              |                        50 |
| Attempted-label policy                     |                      Keep |
| Records after deduplication                |                   320,320 |
| Features                                   |                        80 |
| Retained classes                           |                        21 |
| Training records                           |                   192,192 |
| Validation records                         |                    64,064 |
| Test records                               |                    64,064 |
| Analyst review budget                      |                        5% |
| SHAP concentration threshold               |                      0.15 |
| Analyst error rate                         |                       10% |
| Cross-validation folds                     |                         5 |
| Degradation seeds per condition            |                         5 |
| Random seed                                |                        42 |

The XGBoost classifier uses:

```python
n_estimators = 300
max_depth = 8
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
objective = "multi:softprob"
eval_metric = "mlogloss"
tree_method = "hist"
random_state = 42
```

Balanced sample weights are calculated from the training labels.

## SHAP and routing

Confidence is defined as the maximum predicted class probability after isotonic calibration.

The decision gate routes the 5% of validation predictions with the lowest confidence. This represents a predefined analyst-capacity budget.

SHAP is used for explanation and diagnostic analysis within the confidence-routed subset. SHAP concentration does not independently add high-confidence predictions to the review queue in the reported implementation.

The executed routing result was:

| Measure                                         | Result |
| ----------------------------------------------- | -----: |
| Validation predictions                          | 64,064 |
| Routed predictions                              |  3,203 |
| Routed proportion                               |  5.00% |
| Low-confidence predictions                      |  3,203 |
| Weak-explanation flags within the routed subset |  1,384 |
| Accuracy among auto-accepted predictions        | 1.0000 |
| Accuracy among routed predictions               | 0.9953 |

The routing gate concentrated all 15 validation errors within the 3,203 routed predictions.

## Running the realistic primary model

Open:

```text
HITL_IDS_Pipeline_v3_realistic_reproduction(1).ipynb
```

Confirm the following configuration in Section 2:

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

Restart the kernel and run all cells from the beginning.

The executed realistic run was saved as:

```text
results_reproduction/run_unbalanced_2026-08-23_21-03-47/
```

## Running the pessimistic sensitivity model

Open:

```text
HITL_IDS_Pipeline_v3_pessimistic_reproduction.ipynb
```

The notebook is configured with:

```python
DATA_DIR = "./data/CSECICIDS2018_improved"
RESULTS_ROOT = "./results_reproduction"

SAMPLING_PROFILE = "unbalanced"

ANALYST_ERROR_MODEL = "pessimistic"
NOISE_ANALYST_MODEL = "pessimistic"

ROUTE_BUDGET = 0.05
CONCENTRATION_THRESHOLD = 0.15
ANALYST_ERROR_RATE = 0.10

SEED = 42
```

Restart the kernel and run all cells from the beginning.

The pessimistic notebook uses the fixed output directory:

```text
results_reproduction/run_unbalanced_pessimistic_seed42_reproduction/
```

Rerunning this notebook can replace files inside that directory. Copy or rename the existing directory before rerunning if the original outputs must be preserved.

## Simulated analyst models

### Realistic analyst model

Under the realistic model:

* Correct model predictions are approved.
* Incorrect attack predictions are corrected to the ground-truth attack class.
* Incorrect predictions whose true class is benign are rejected as false alarms.
* An analyst error can occur only when the model prediction is already incorrect.
* When such an error occurs, the analyst leaves the original incorrect model prediction unchanged.

The executed realistic review produced:

| Decision       | Count |
| -------------- | ----: |
| Approve        | 3,188 |
| Correct        |     9 |
| Reject         |     5 |
| Error          |     1 |
| Total reviewed | 3,203 |

Fourteen labels were correctly changed. One existing model error was left uncorrected. The realistic analyst did not corrupt an originally correct prediction.

### Pessimistic analyst model

Under the pessimistic model:

* The same approve, correct and reject decisions are available.
* Analyst errors may also occur when the model prediction was originally correct.
* An erroneous review of an originally correct prediction reassigns it to a uniformly selected alternative class.
* With 21 retained classes, the alternative is selected from the remaining 20 classes.
* This represents a worst-case corruption model rather than a representative model of analyst confusion.

The executed pessimistic review produced:

| Measure                                                       | Count |
| ------------------------------------------------------------- | ----: |
| Approve                                                       | 2,861 |
| Correct                                                       |     9 |
| Reject                                                        |     6 |
| Error                                                         |   327 |
| Successful corrections or rejections                          |    15 |
| Incorrect labels introduced on originally correct predictions |   327 |
| Existing model errors not corrected                           |     0 |
| Total labels changed                                          |   342 |
| Total incorrect reviewed labels after adjudication            |   327 |

## Primary realistic results

### AI-only and HITL test comparison

| Metric                           | AI-only | Realistic HITL | Difference |
| -------------------------------- | ------: | -------------: | ---------: |
| Accuracy                         |  0.9994 |         0.9995 |    +0.0001 |
| Macro precision                  |  0.9292 |         0.9367 |    +0.0076 |
| Macro recall                     |  0.9524 |         0.9619 |    +0.0095 |
| Macro F1                         |  0.9319 |         0.9421 |    +0.0102 |
| Weighted F1                      |  0.9995 |         0.9995 |    +0.0001 |
| Matthews correlation coefficient |  0.9953 |         0.9960 |    +0.0007 |

### Realistic augmentation control

| Configuration                                          | Macro F1 |
| ------------------------------------------------------ | -------: |
| AI-only baseline                                       |   0.9319 |
| Routed records added without human correction          |   0.9389 |
| Routed records added with realistic analyst correction |   0.9421 |

The total macro-F1 change was +0.0102:

* Contribution from additional routed training records: +0.0070.
* Contribution from simulated human correction: +0.0032.

## Pessimistic sensitivity results

### AI-only and pessimistic HITL comparison

| Metric                           | AI-only | Pessimistic HITL | Difference |
| -------------------------------- | ------: | ---------------: | ---------: |
| Accuracy                         |  0.9994 |           0.9972 |    -0.0022 |
| Macro precision                  |  0.9292 |           0.7010 |    -0.2282 |
| Macro recall                     |  0.9524 |           0.9660 |    +0.0137 |
| Macro F1                         |  0.9319 |           0.7920 |    -0.1399 |
| Weighted F1                      |  0.9995 |           0.9976 |    -0.0019 |
| Matthews correlation coefficient |  0.9953 |           0.9776 |    -0.0178 |

### Pessimistic augmentation control

| Configuration                                        | Macro F1 |
| ---------------------------------------------------- | -------: |
| AI-only baseline                                     |   0.9319 |
| Routed records added without human correction        |   0.9389 |
| Routed records added with pessimistic analyst labels |   0.7920 |

The total macro-F1 change was -0.1399:

* Contribution from additional routed training records: +0.0070.
* Contribution from simulated human labels: -0.1469.

The pessimistic result bounds potential harm under deliberately destructive analyst corruption. It is not an estimate of typical real-world analyst performance.

## Cross-validation

### Realistic model

| Configuration  | Mean macro F1 | Standard deviation |
| -------------- | ------------: | -----------------: |
| AI-only        |        0.9480 |             0.0106 |
| Realistic HITL |        0.9476 |             0.0128 |

### Pessimistic model

| Configuration    | Mean macro F1 | Standard deviation |
| ---------------- | ------------: | -----------------: |
| AI-only          |        0.9480 |             0.0106 |
| Pessimistic HITL |        0.7288 |             0.0159 |

These cross-validation results are descriptive. The folds were not treated as matched samples, so no paired statistical test was applied to these fold means.

## SHAP explanation stability

Both notebooks report the same baseline SHAP explanation-stability results:

| Measure                      |  Value |
| ---------------------------- | -----: |
| Mean Spearman correlation    | 0.9523 |
| Minimum Spearman correlation | 0.9349 |
| Mean Kendall correlation     | 0.8407 |
| Mean top-10 Jaccard overlap  | 0.5678 |

The Spearman and Kendall results indicate stable overall feature ordering. The lower top-10 Jaccard overlap shows that membership of the highest-ranked feature subset varies more across folds.

## Label-noise experiments

Section 19 evaluates:

* Symmetric label noise.
* Attack-to-benign label noise.
* Noise rates of 0%, 7.5%, 20% and 40%.
* Five seeds per condition.
* AI-only and HITL performance on the unchanged clean test partition.

The two notebooks use different analyst assumptions during these experiments:

| Notebook             | Noise analyst model |
| -------------------- | ------------------- |
| Realistic notebook   | Realistic           |
| Pessimistic notebook | Pessimistic         |

The realistic notebook is the source of the primary label-noise results reported in the thesis. The pessimistic notebook provides an additional worst-case noise sensitivity analysis.

Detailed values are written to:

```text
noise_multiseed.csv
```

The corresponding figure is written to:

```text
fig5_noise_multiseed.png
```

## Generated outputs

Both notebooks write output files under `results_reproduction/`.

```text
results_reproduction/
├── population_class_counts.csv
├── sampled_unbalanced.csv
├── run_unbalanced_<timestamp>/
│   ├── baseline_xgb.json
│   ├── calibrated_model.pkl
│   ├── comparison_ai_vs_hitl.csv
│   ├── control_augmentation.json
│   ├── fig1_confusion_baseline.png
│   ├── fig2_calibration.png
│   ├── fig3_shap_global.png
│   ├── fig4_budget_tradeoff.png
│   ├── fig5_noise_multiseed.png
│   ├── hitl_xgb.json
│   ├── noise_multiseed.csv
│   ├── preprocessing.pkl
│   ├── results_summary.json
│   ├── results_table.md
│   ├── sample_representativeness.csv
│   ├── shap_stability.csv
│   └── threshold_sweep.csv
└── run_unbalanced_pessimistic_seed42_reproduction/
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

## Verification tests

The repository includes `test_pipeline.py`, containing 23 synthetic-data tests covering:

* Sampling targets.
* Protection of rare classes.
* Class caps.
* Deduplication.
* Partition leakage.
* Stratification.
* Training-only scaling.
* Fixed-budget routing.
* Tied confidence values.
* SHAP concentration.
* Analyst decision semantics.
* Label-noise injection.
* End-to-end retraining.
* Deterministic training.

Run the test file with:

```bash
pytest test_pipeline.py -v
```

The test file mirrors selected helper functions from the notebook. The executed notebooks and their recorded outputs remain the primary source for the reported experimental results.

## Scope and limitations

The experimental evidence is limited to:

* One corrected intrusion-detection benchmark.
* One executed unbalanced sampling profile.
* One classifier family.
* One simulated analyst-review cycle.
* Simulated rather than human-provided labels.
* A fixed 5% review-capacity budget.
* One primary random seed, with five-seed replication used for the label-noise experiments.

The findings should therefore be interpreted as evidence for the evaluated configuration. They do not establish that the same performance changes will generalise to other datasets, time periods, organisations, analyst populations or classifier families.

The realistic model is the primary operational assumption. The pessimistic model is a deliberately destructive sensitivity bound and is not intended to represent typical analyst behaviour.

## Reproducibility notes

* The primary random seed is 42.
* Multi-seed label-noise experiments use seeds 42 to 46.
* The population scan and sampled dataset are cached under `results_reproduction/`.
* The realistic notebook creates a timestamped output directory.
* The pessimistic notebook uses a fixed named output directory.
* Both notebooks use the same sampled unbalanced dataset.
* The realistic notebook is the source of the primary reported results.
* The pessimistic notebook is the source of the worst-case analyst-error sensitivity results.
* Generated models and large sampled datasets are not stored in the repository.

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
