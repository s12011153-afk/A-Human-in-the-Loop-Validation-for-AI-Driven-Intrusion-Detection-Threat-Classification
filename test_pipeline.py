"""
Verification suite for the HITL intrusion detection pipeline.

These tests verify the invariants the methodology depends upon. They run on
small synthetic data rather than CSE-CIC-IDS2018, so the suite completes in
seconds and requires no dataset download.

Run with:   pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

SEED = 42


# ---------------------------------------------------------------------------
# Functions under test (mirroring the notebook implementation)
# ---------------------------------------------------------------------------
def compute_targets(pop_counts, fraction, floor, cap):
    targets, floored, capped = {}, [], []
    for lbl, n in pop_counts.items():
        n = int(n)
        proportional = int(round(n * fraction))
        target = max(proportional, min(floor, n))
        if target > proportional:
            floored.append(lbl)
        if cap is not None and target > cap:
            target = cap
            capped.append(lbl)
        targets[lbl] = min(target, n)
    return pd.Series(targets), floored, capped


def concentration(shap_vals, preds, positions):
    out = np.zeros(len(positions))
    for k, i in enumerate(positions):
        row = shap_vals[i, :, preds[i]] if np.ndim(shap_vals) == 3 else shap_vals[i]
        a = np.abs(row)
        s = a.sum()
        out[k] = 0.0 if s == 0 else a.max() / s
    return out


def route_by_budget(conf, budget):
    k = max(1, int(round(len(conf) * budget)))
    mask = np.zeros(len(conf), dtype=bool)
    mask[np.argsort(conf)[:k]] = True
    return mask


def simulated_analyst(y_true, y_pred, benign_enc, error_rate, rng, model="realistic"):
    decisions = np.empty(len(y_true), dtype=object)
    final = y_pred.copy()
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            if model == "pessimistic" and rng.random() < error_rate:
                decisions[i] = "error"; final[i] = p
            else:
                decisions[i] = "approve"; final[i] = p
            continue
        if rng.random() < error_rate:
            decisions[i] = "error"; final[i] = p
            continue
        if t == benign_enc:
            decisions[i] = "reject"; final[i] = benign_enc
        else:
            decisions[i] = "correct"; final[i] = t
    return decisions, final


def inject_label_noise(y_clean, rate, mode, rng, benign_enc, n_classes):
    y_noisy = y_clean.copy()
    if rate <= 0:
        return y_noisy, np.array([], dtype=int)
    if mode == "attack_to_benign":
        cand = np.where(y_clean != benign_enc)[0]
        k = max(0, min(len(cand) - 1, int(round(len(cand) * rate))))
        idx = rng.choice(cand, size=k, replace=False) if k else np.array([], dtype=int)
        y_noisy[idx] = benign_enc
    else:
        idx = rng.choice(len(y_clean), size=int(round(len(y_clean) * rate)), replace=False)
        for i in idx:
            alt = [c for c in range(n_classes) if c != y_clean[i]]
            y_noisy[i] = rng.choice(alt)
    return y_noisy, idx


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    """Imbalanced multiclass data with a benign majority, mirroring the real profile."""
    rng = np.random.default_rng(SEED)
    classes = ["BENIGN", "DoS", "DDoS", "Botnet", "Rare"]
    props = [0.90, 0.04, 0.03, 0.02, 0.01]
    n, n_feat = 4000, 10
    X = rng.normal(size=(n, n_feat))
    y_lbl = rng.choice(classes, size=n, p=props)
    for i, c in enumerate(classes):
        X[y_lbl == c, i % n_feat] += i * 2.0
    y = np.array([classes.index(c) for c in y_lbl])
    return X.astype(np.float32), y, classes


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def test_proportional_sampling_preserves_distribution():
    """Without a cap, sampled proportions should track the population."""
    pop = pd.Series({"BENIGN": 1_000_000, "DoS": 50_000, "DDoS": 30_000})
    targets, floored, capped = compute_targets(pop, 0.01, 50, None)
    assert not capped
    pop_share = pop / pop.sum()
    smp_share = targets / targets.sum()
    assert np.allclose(pop_share.values, smp_share.values, atol=0.001)


def test_floor_protects_rare_classes():
    """A class whose proportional share is below the floor is raised to it."""
    pop = pd.Series({"BENIGN": 1_000_000, "Rare": 200})
    targets, floored, _ = compute_targets(pop, 0.01, 50, None)
    assert "Rare" in floored
    assert targets["Rare"] == 50


def test_cap_limits_dominant_class():
    pop = pd.Series({"BENIGN": 1_000_000, "DoS": 50_000})
    targets, _, capped = compute_targets(pop, 0.10, 50, 20_000)
    assert "BENIGN" in capped
    assert targets["BENIGN"] == 20_000


def test_target_never_exceeds_population():
    pop = pd.Series({"Tiny": 10})
    targets, _, _ = compute_targets(pop, 0.5, 500, None)
    assert targets["Tiny"] == 10


# ---------------------------------------------------------------------------
# Deduplication and leakage
# ---------------------------------------------------------------------------
def test_deduplication_requires_identifier_removal():
    """With a unique id column present, deduplication removes nothing."""
    df = pd.DataFrame({"id": [1, 2, 3], "f1": [0.5, 0.5, 0.5], "Label": ["A", "A", "A"]})
    assert len(df.drop_duplicates()) == 3                       # id makes rows unique
    assert len(df.drop_duplicates(subset=["f1", "Label"])) == 1  # after removal


def test_no_leakage_between_partitions(synthetic):
    """No test record may duplicate a training record across all features."""
    X, y, _ = synthetic
    df = pd.DataFrame(X)
    df["Label"] = y
    df = df.drop_duplicates().reset_index(drop=True)
    Xd, yd = df.drop(columns="Label").values, df["Label"].values
    X_tr, X_te, _, _ = train_test_split(Xd, yd, test_size=0.2, stratify=yd, random_state=SEED)
    tr = pd.DataFrame(X_tr).round(6).drop_duplicates()
    te = pd.DataFrame(X_te).round(6)
    assert te.merge(tr, how="inner").shape[0] == 0


def test_stratification_preserves_class_proportions(synthetic):
    X, y, _ = synthetic
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    for c in np.unique(y):
        assert abs((y_tr == c).mean() - (y_te == c).mean()) < 0.02


def test_scaler_fitted_on_training_only(synthetic):
    """Test statistics must not influence the transformation."""
    X, y, _ = synthetic
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    assert np.allclose(sc.mean_, X_tr.mean(axis=0))
    assert not np.allclose(sc.mean_, X.mean(axis=0), atol=1e-6)


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------
def test_routing_budget_is_exact():
    """The routed proportion must equal the budget regardless of distribution shape."""
    rng = np.random.default_rng(SEED)
    for n in (1000, 5000):
        for budget in (0.01, 0.05, 0.20):
            conf = rng.uniform(0, 1, n)
            assert route_by_budget(conf, budget).sum() == round(n * budget)


def test_routing_budget_survives_tied_confidences():
    """A degenerate distribution must not collapse routing to 0% or 100%."""
    conf = np.full(1000, 0.999999)      # all identical
    mask = route_by_budget(conf, 0.05)
    assert mask.sum() == 50


def test_routing_selects_least_confident():
    conf = np.linspace(0, 1, 100)
    mask = route_by_budget(conf, 0.10)
    assert mask[:10].all() and not mask[10:].any()


def test_concentration_bounds_and_semantics():
    """Concentration is in [0,1]; one dominant feature scores high, uniform scores low."""
    dominant = np.zeros((1, 5, 1)); dominant[0, :, 0] = [10.0, 0.1, 0.1, 0.1, 0.1]
    uniform  = np.zeros((1, 5, 1)); uniform[0, :, 0]  = [1.0, 1.0, 1.0, 1.0, 1.0]
    preds = np.array([0])
    c_dom = concentration(dominant, preds, [0])[0]
    c_uni = concentration(uniform, preds, [0])[0]
    assert 0.0 <= c_uni <= c_dom <= 1.0
    assert c_dom > 0.9 and c_uni == pytest.approx(0.2)


def test_concentration_handles_zero_attribution():
    zero = np.zeros((1, 5, 1))
    assert concentration(zero, np.array([0]), [0])[0] == 0.0


# ---------------------------------------------------------------------------
# Simulated analyst
# ---------------------------------------------------------------------------
def test_analyst_decision_semantics():
    """Correct predictions approved; wrong attack types corrected; benign rejected."""
    y_true = np.array([1, 2, 0])
    y_pred = np.array([1, 3, 3])
    rng = np.random.default_rng(SEED)
    dec, final = simulated_analyst(y_true, y_pred, benign_enc=0, error_rate=0.0, rng=rng)
    assert list(dec) == ["approve", "correct", "reject"]
    assert list(final) == [1, 2, 0]


def test_realistic_model_never_corrupts_correct_predictions():
    """Under the realistic model, an already-correct prediction cannot be degraded."""
    y_true = y_pred = np.zeros(500, dtype=int)
    rng = np.random.default_rng(SEED)
    dec, final = simulated_analyst(y_true, y_pred, 0, error_rate=0.5, rng=rng, model="realistic")
    assert (dec == "approve").all()
    assert (final == y_true).all()


def test_pessimistic_model_does_corrupt_correct_predictions():
    y_true = y_pred = np.zeros(500, dtype=int)
    rng = np.random.default_rng(SEED)
    dec, _ = simulated_analyst(y_true, y_pred, 0, error_rate=0.5, rng=rng, model="pessimistic")
    assert (dec == "error").sum() > 0


def test_zero_error_rate_recovers_ground_truth():
    rng_data = np.random.default_rng(1)
    y_true = rng_data.integers(0, 5, 200)
    y_pred = rng_data.integers(0, 5, 200)
    _, final = simulated_analyst(y_true, y_pred, 0, 0.0, np.random.default_rng(SEED))
    assert (final == y_true).all()


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------
def test_symmetric_noise_changes_expected_proportion():
    rng = np.random.default_rng(SEED)
    y = np.random.default_rng(1).integers(0, 5, 1000)
    y_n, idx = inject_label_noise(y, 0.20, "symmetric", rng, 0, 5)
    assert len(idx) == 200
    assert (y_n[idx] != y[idx]).all()          # every selected label actually changed


def test_attack_to_benign_rate_is_relative_to_attacks():
    """Rate applies to attack records, not the whole dataset."""
    y = np.array([0] * 940 + [1] * 60)          # 6% attacks, as in the real profile
    rng = np.random.default_rng(SEED)
    y_n, idx = inject_label_noise(y, 0.50, "attack_to_benign", rng, 0, 2)
    assert len(idx) == 30                        # 50% of 60, not 50% of 1000


def test_attack_to_benign_never_eliminates_all_classes():
    """A high rate must leave at least one attack record, keeping training possible."""
    y = np.array([0] * 940 + [1] * 60)
    y_n, _ = inject_label_noise(y, 1.0, "attack_to_benign", np.random.default_rng(SEED), 0, 2)
    assert len(np.unique(y_n)) >= 2


def test_noise_is_reproducible_under_seed():
    y = np.random.default_rng(1).integers(0, 5, 500)
    a, _ = inject_label_noise(y, 0.3, "symmetric", np.random.default_rng(SEED), 0, 5)
    b, _ = inject_label_noise(y, 0.3, "symmetric", np.random.default_rng(SEED), 0, 5)
    assert (a == b).all()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def test_pipeline_runs_end_to_end(synthetic):
    """Gate, adjudication and retraining complete and produce a usable model."""
    X, y, classes = synthetic
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                                stratify=y_tmp, random_state=SEED)
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s, X_te_s = sc.transform(X_tr), sc.transform(X_val), sc.transform(X_te)

    def build():
        return xgb.XGBClassifier(n_estimators=40, max_depth=4, tree_method="hist",
                                 objective="multi:softprob", eval_metric="mlogloss",
                                 random_state=SEED, n_jobs=1)

    base = build()
    base.fit(X_tr_s, y_tr, sample_weight=compute_sample_weight("balanced", y_tr), verbose=False)

    proba = base.predict_proba(X_val_s)
    routed = route_by_budget(proba.max(1), 0.05)
    idx = np.where(routed)[0]
    assert len(idx) == round(len(X_val_s) * 0.05)

    _, corrected = simulated_analyst(y_val[idx], proba.argmax(1)[idx], 0, 0.10,
                                     np.random.default_rng(SEED))
    X_h = np.vstack([X_tr_s, X_val_s[idx]])
    y_h = np.concatenate([y_tr, corrected])
    assert len(y_h) == len(y_tr) + len(idx)

    hitl = build()
    hitl.fit(X_h, y_h, sample_weight=compute_sample_weight("balanced", y_h), verbose=False)
    assert hitl.predict(X_te_s).shape == y_te.shape


def test_training_is_deterministic_under_seed(synthetic):
    """Identical seeds must produce identical predictions."""
    X, y, _ = synthetic
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    def build():
        return xgb.XGBClassifier(n_estimators=30, max_depth=3, tree_method="hist",
                                 objective="multi:softprob", eval_metric="mlogloss",
                                 random_state=SEED, n_jobs=1)

    a, b = build(), build()
    a.fit(Xs, y, verbose=False)
    b.fit(Xs, y, verbose=False)
    assert (a.predict(Xs) == b.predict(Xs)).all()
