# A Human-in-the-Loop Validation for AI-Driven Intrusion Detection Threat Classification

MSc Artificial Intelligence dissertation — Sehrish Shahid (12011153)
Faculty of IT & Creative Media, Bahrain Polytechnic, 2026

Implementation and experimental artefacts for a closed-loop human-in-the-loop validation
framework for multiclass network intrusion detection. An XGBoost classifier produces calibrated
confidence scores and SHAP attributions; a decision gate routes the least-confident predictions
for analyst adjudication within a fixed capacity budget; adjudicated labels are returned to the
model through a retraining cycle.

---

## Repository structure

```
.
├── HITL_IDS_Pipeline_v2.ipynb     Main experimental pipeline (22 sections)
├── tests/
│   └── test_pipeline.py           Verification suite (23 tests)
├── requirements.txt               Pinned dependencies
├── environment.yml                Conda environment specification
├── ARTEFACTS.md                   Description of generated outputs
└── results/                       Written by the pipeline (not tracked)
    ├── population_class_counts.csv
    ├── sampled_unbalanced.csv
    └── run_<profile>_<timestamp>/
```

Datasets and trained model artefacts are excluded from version control. The dataset is
redistributed by its originators under their own terms, and model files exceed repository size
limits. Both are reproducible from the instructions below.

---

## Environment

Python 3.11. Either method works.

**Conda**

```bash
conda env create -f environment.yml
conda activate ms_env
```

**pip**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
python -c "import xgboost, shap, sklearn; print(xgboost.__version__, shap.__version__, sklearn.__version__)"
# expected: 3.2.0 0.51.0 1.9.0
```

> **Note on scikit-learn.** `CalibratedClassifierCV(cv="prefit")` was removed in 1.8. The pipeline
> detects `FrozenEstimator` and selects the appropriate construction, so both older and current
> releases are supported.

---

## Verification suite

```bash
pytest tests/ -v
```

Expected: **23 passed**, approximately 10 seconds. The suite runs on synthetic data and requires
no dataset download.

Tests verify the invariants the methodology depends upon:

| Group | Verifies |
|---|---|
| Sampling | Proportional sampling preserves the population distribution; floor protects rare classes; cap limits dominant classes; targets never exceed population |
| Preprocessing | Deduplication is ineffective before identifier removal; no train/test leakage; stratification preserves proportions; scaler fitted on training data only |
| Decision gate | Routed proportion equals the budget exactly; routing survives tied confidences; least-confident records selected; concentration bounded in [0,1] with correct semantics |
| Simulated analyst | Approve/correct/reject semantics; realistic model never corrupts correct predictions; pessimistic model does; zero error rate recovers ground truth |
| Noise injection | Symmetric rate applies to all records; attack-to-benign rate applies to attack records only; high rates never eliminate all classes; reproducible under seed |
| End-to-end | Full gate → adjudication → retraining cycle completes; training deterministic under fixed seed |

---

## Data acquisition

The corrected CSE-CIC-IDS2018 release (Liu et al., 2022) is required. The original distribution
contains documented labelling errors and is **not** suitable.

Download from the authors' page:
<https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html>

Place the ten daily CSV files in:

```
data/CSECICIDS2018_improved/
```

Approximately 35 GB, 63,195,145 records across 25 classes.

---

## Reproducing the reported results

1. Open `HITL_IDS_Pipeline_v3_realistic_reproduction.ipynb` and select the `ms_env` kernel.
2. In **Section 2 — Configuration**, confirm:

```python
DATA_DIR            = "./data/CSECICIDS2018_improved"
SAMPLING_PROFILE    = "unbalanced"
ROUTE_BUDGET        = 0.05
ANALYST_ERROR_MODEL = "realistic"
SEED                = 42
```

3. Run all cells.

Results are written to a timestamped directory under `results/`. Nothing is overwritten between
runs.

**Expected runtime.** First execution approximately two hours, dominated by the multi-seed
degradation experiment in Section 19 (roughly 94 minutes; 80 models trained). Subsequent runs are
faster: the population scan and the sampled dataset are both cached to disk.

**Sensitivity analysis.** Set `ANALYST_ERROR_MODEL = "pessimistic"` and re-run Sections 12–17 only.
This writes to a new run directory, leaving the primary results intact.

**Comparison profile.** Set `SAMPLING_PROFILE = "balanced"` and run from Section 2. A separate
sample cache is maintained per profile.

---

## Reproducibility

- All stochastic operations are seeded (`SEED = 42`).
- Every experimental parameter is defined in a single configuration cell.
- Each run writes its complete configuration alongside its metrics in `results_summary.json`, so
  any reported figure can be attributed to the settings that produced it.
- Sampling stages are cached, so the same sample is reproduced exactly across runs.
- Verification checks — leakage audit, representativeness test, class-count guard, explanation
  stability — execute during normal operation and report their results as output.

---

## Reported configuration

| Parameter | Value |
|---|---|
| Dataset | CSE-CIC-IDS2018, corrected release |
| Sampling | 0.55% per class, no cap, floor 50 |
| Records after deduplication | 320,320 |
| Features / classes | 80 / 21 |
| Partition | 192,192 / 64,064 / 64,064 |
| Classifier | XGBoost, 300 estimators, depth 8, lr 0.1, balanced weights |
| Calibration | Isotonic, fitted on validation |
| Analyst capacity budget | 5% |
| Concentration threshold | 0.15 |
| Analyst error rate / model | 0.10, realistic |
| Cross-validation | 5-fold stratified |
| Degradation replication | 5 seeds per condition |

---

## Citation

```bibtex
@mastersthesis{shahid2026hitl,
  title  = {A Human-in-the-Loop Validation for AI-Driven Intrusion
            Detection Threat Classification},
  author = {Shahid, Sehrish},
  school = {Bahrain Polytechnic},
  year   = {2026},
  type   = {{MSc} dissertation}
}
```

## Acknowledgement of AI tool use

Generative AI tools were used as productivity aids during development, for code drafting,
debugging assistance, and literature search support, in accordance with the programme handbook.
All generated content was reviewed, tested and adapted by the author. The AI usage statement in
the dissertation records the scope of this use.
