# Artefact Manifest

Files written by `HITL_IDS_Pipeline_v2.ipynb` to `results/`.

## Shared across runs

| File | Description |
|---|---|
| `population_class_counts.csv` | Class distribution of the full 63,195,145-record population, from the pass-1 label scan. Cached so the scan runs once. |
| `sampled_<profile>.csv` | The sampled dataset for a given profile. Cached so identical samples are reused across configuration changes. |

## Per run — `run_<profile>_<timestamp>/`

### Models and preprocessing

| File | Description |
|---|---|
| `baseline_xgb.json` | AI-only classifier |
| `hitl_xgb.json` | Classifier retrained on analyst-adjudicated labels |
| `calibrated_model.pkl` | Isotonic calibrator wrapping the baseline |
| `preprocessing.pkl` | Fitted scaler, label encoder, feature list, sampling profile |

### Metrics

| File | Description |
|---|---|
| `results_summary.json` | Every reported metric with the configuration that produced it. Primary record. |
| `results_table.md` | AI-only vs HITL comparison, formatted for inclusion in the dissertation |
| `comparison_ai_vs_hitl.csv` | The same comparison in tabular form |
| `sample_representativeness.csv` | Per-class population and sample proportions, deviation, adjustment flag |
| `threshold_sweep.csv` | Routing budget against residual error among accepted and routed predictions |
| `shap_stability.csv` | Per-feature mean absolute SHAP and rank standard deviation across folds |
| `noise_multiseed.csv` | Degradation results: means, standard deviations, Cohen's d, p-values, seeds won |
| `control_augmentation.json` | Augmentation control decomposition |

### Figures

| File | Dissertation reference |
|---|---|
| `fig1_confusion_baseline.png` | Figure 5.1 |
| `fig2_calibration.png` | Figure 5.2 |
| `fig3_shap_global.png` | Figure 5.3 |
| `fig4_budget_tradeoff.png` | Figure 5.4 |
| `fig5_noise_multiseed.png` | Figure 5.5 |

## Runs reported in the dissertation

| Run | Purpose |
|---|---|
| `run_unbalanced_<timestamp>` | Primary results. Proportional sampling, realistic analyst error model. Source of all figures and of Tables 5.1–5.13. |
| `run_unbalanced_<timestamp>` (second) | Sensitivity analysis. Pessimistic analyst error model. Source of Table 5.14. |
| `run_balanced_<timestamp>` | Comparison profile, referenced in Section 6.7. |

Record the exact directory names of the runs used for the submitted figures, and archive those
directories alongside the dissertation.
