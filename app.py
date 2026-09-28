
import copy
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AdaptAI Transferability Demo", layout="wide")
st.title("AdaptAI — AI Transferability Assessment & Adaptation Simulator")
st.caption("Research prototype. Demonstration data/model only — not for clinical decision-making.")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def safe_auc(y, p):
    """AUROC fails if a split happens to contain only one class."""
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def metrics(y, p):
    """Core metrics for discrimination and calibration."""
    return {
        "AUROC": safe_auc(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p)
    }

def psi(expected, actual, bins=10):
    """
    Population Stability Index (PSI).
    We use source quantiles as reference bins.
    Rule-of-thumb thresholds are descriptive only:
      <0.10 small shift, 0.10-0.25 moderate, >0.25 substantial.
    """
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)
    if expected.nunique() < 2 or actual.nunique() < 2:
        return 0.0

    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf

    e = pd.cut(expected, cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    a = pd.cut(actual, cuts, include_lowest=True).value_counts(normalize=True, sort=False)
    e = np.clip(e.values, 1e-6, None)
    a = np.clip(a.values, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))

def shift_label(v):
    if v < 0.10:
        return "LOW"
    if v < 0.25:
        return "MODERATE"
    return "HIGH"

def prepare(df, features, medians):
    """Keep required features, coerce to numeric, and impute using SOURCE medians."""
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(medians[c])
    return X

def calibration_recalibrate(y_train, p_train, p_test):
    """
    Logistic recalibration using the source model's log-odds as one predictor.
    This changes probability calibration without rebuilding the whole model.
    """
    eps = 1e-6
    tr_logit = np.log(np.clip(p_train, eps, 1-eps) / (1-np.clip(p_train, eps, 1-eps)))
    te_logit = np.log(np.clip(p_test, eps, 1-eps) / (1-np.clip(p_test, eps, 1-eps)))
    cal = LogisticRegression()
    cal.fit(tr_logit.reshape(-1, 1), y_train)
    return cal.predict_proba(te_logit.reshape(-1, 1))[:, 1]

def fine_tune_source_model(source_model, X_target_scaled, y_target, epochs=12):
    """
    Continue training the already-trained neural network on labelled target data.
    This is the prototype's transfer-learning strategy: SOURCE weights are retained,
    then adapted using local target examples.
    """
    m = copy.deepcopy(source_model)
    # partial_fit continues from existing learned weights.
    for _ in range(epochs):
        m.partial_fit(X_target_scaled, y_target, classes=np.array([0, 1]))
    return m

# ==========================================================
# LOAD DEFAULT MODEL AND FILES
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("source_model.joblib")

bundle = load_model()
features = bundle["features"]
source_medians = bundle["source_medians"]

st.sidebar.header("Inputs")
source_upload = st.sidebar.file_uploader("Source hospital CSV", type="csv")
target_upload = st.sidebar.file_uploader("Target hospital CSV", type="csv")

source = pd.read_csv(source_upload) if source_upload else pd.read_csv("source_hospital.csv")
target = pd.read_csv(target_upload) if target_upload else pd.read_csv("target_hospital.csv")

st.sidebar.write("Source:", len(source), "records")
st.sidebar.write("Target:", len(target), "records")

if "outcome" not in source.columns or "outcome" not in target.columns:
    st.error("Both demo files need an 'outcome' column for this validation prototype.")
    st.stop()

missing_features = [c for c in features if c not in target.columns or c not in source.columns]
if missing_features:
    st.error(f"Missing required features: {missing_features}")
    st.stop()

# ==========================================================
# SCREEN 1 — SOURCE / TARGET OVERVIEW
# ==========================================================

st.header("1. Source–Target Setup")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source N", f"{len(source):,}")
c2.metric("Target N", f"{len(target):,}")
c3.metric("Source prevalence", f"{source.outcome.mean():.1%}")
c4.metric("Target prevalence", f"{target.outcome.mean():.1%}")

# ==========================================================
# SCREEN 2 — TRANSFERABILITY PROFILE
# ==========================================================

st.header("2. Transferability Assessment")

shift_rows = []
for f in features:
    ps = psi(source[f], target[f])
    ks = ks_2samp(source[f].dropna(), target[f].dropna()).statistic
    miss_s = source[f].isna().mean()
    miss_t = target[f].isna().mean()
    shift_rows.append({
        "Feature": f,
        "PSI": round(ps, 3),
        "PSI level": shift_label(ps),
        "KS statistic": round(float(ks), 3),
        "Source missing %": round(100 * miss_s, 1),
        "Target missing %": round(100 * miss_t, 1)
    })

shift_df = pd.DataFrame(shift_rows)
st.dataframe(shift_df, use_container_width=True)

X_source = prepare(source, features, source_medians)
X_target = prepare(target, features, source_medians)
Xs = bundle["scaler"].transform(X_source)
Xt = bundle["scaler"].transform(X_target)

p_source = bundle["model"].predict_proba(Xs)[:, 1]
p_target = bundle["model"].predict_proba(Xt)[:, 1]
m_source = metrics(source.outcome.values, p_source)
m_target = metrics(target.outcome.values, p_target)

a, b, c = st.columns(3)
a.metric("Source AUROC", f"{m_source['AUROC']:.3f}")
b.metric("Target AUROC", f"{m_target['AUROC']:.3f}",
         delta=f"{m_target['AUROC']-m_source['AUROC']:+.3f}")
c.metric("Target Brier", f"{m_target['Brier']:.3f}")

# Simple profile — deliberately transparent rather than a black-box single score.
high_shift = int((shift_df["PSI level"] == "HIGH").sum())
moderate_shift = int((shift_df["PSI level"] == "MODERATE").sum())
prev_delta = abs(target.outcome.mean() - source.outcome.mean())

st.subheader("Transferability Profile")
p1, p2, p3, p4 = st.columns(4)
p1.metric("High-shift features", high_shift)
p2.metric("Moderate-shift features", moderate_shift)
p3.metric("Prevalence difference", f"{prev_delta:.1%}")
p4.metric("AUROC degradation", f"{m_source['AUROC']-m_target['AUROC']:.3f}")

# ==========================================================
# SCREEN 3 — GAP / RISK PROFILE
# ==========================================================

st.header("3. Gap / Risk Profile")
chart_df = shift_df.set_index("Feature")[["PSI", "KS statistic"]]
st.bar_chart(chart_df)

max_psi = shift_df["PSI"].max()
if max_psi > 0.25:
    st.warning("Substantial source–target population shift is present in at least one feature.")
if prev_delta > 0.05:
    st.warning("Outcome prevalence differs materially between source and target in this demo.")
if m_target["AUROC"] < m_source["AUROC"] - 0.05:
    st.warning("The source model loses discrimination at the target site.")

# ==========================================================
# SCREEN 4 — ADAPTATION SIMULATOR
# ==========================================================

st.header("4. Adaptation Simulator")
st.write(
    "The target dataset is split chronologically-by-row for this demo: "
    "the first 70% forms the local adaptation pool and the final 30% is untouched test data."
)

split = int(len(target) * 0.70)
adapt_pool = target.iloc[:split].copy()
test = target.iloc[split:].copy()

fractions = st.multiselect(
    "Target-data fractions to simulate",
    [0.10, 0.25, 0.50, 0.75, 1.00],
    default=[0.10, 0.25, 0.50, 1.00]
)
epochs = st.slider("Fine-tuning epochs", 3, 30, 12)

if st.button("Run adaptation simulation", type="primary"):
    Xtest = prepare(test, features, source_medians)
    Xtest_s = bundle["scaler"].transform(Xtest)
    ytest = test.outcome.values

    # Direct transfer is identical for every target-data fraction.
    direct_p = bundle["model"].predict_proba(Xtest_s)[:, 1]
    results = []

    for frac in sorted(fractions):
        n = max(30, int(len(adapt_pool) * frac))
        train = adapt_pool.sample(n=n, random_state=int(frac * 1000))
        Xtr = prepare(train, features, source_medians)
        Xtr_s = bundle["scaler"].transform(Xtr)
        ytr = train.outcome.values

        # Strategy A: Direct transfer
        dm = metrics(ytest, direct_p)
        results.append({"Fraction": frac, "N local": n, "Strategy": "Direct", **dm})

        # Strategy B: Recalibration
        ptr = bundle["model"].predict_proba(Xtr_s)[:, 1]
        recal_p = calibration_recalibrate(ytr, ptr, direct_p)
        rm = metrics(ytest, recal_p)
        results.append({"Fraction": frac, "N local": n, "Strategy": "Recalibration", **rm})

        # Strategy C: Transfer learning / fine-tuning
        ft = fine_tune_source_model(bundle["model"], Xtr_s, ytr, epochs=epochs)
        ft_p = ft.predict_proba(Xtest_s)[:, 1]
        fm = metrics(ytest, ft_p)
        results.append({"Fraction": frac, "N local": n, "Strategy": "Fine-tune source", **fm})

        # Strategy D: Local retraining from scratch.
        # A fresh scaler/model uses ONLY local target data.
        local_scaler = StandardScaler()
        Xtr_local = local_scaler.fit_transform(Xtr)
        Xtest_local = local_scaler.transform(Xtest)
        local = MLPClassifier(
            hidden_layer_sizes=(24, 12), max_iter=250,
            alpha=0.001, random_state=42
        )
        local.fit(Xtr_local, ytr)
        lp = local.predict_proba(Xtest_local)[:, 1]
        lm = metrics(ytest, lp)
        results.append({"Fraction": frac, "N local": n, "Strategy": "Local retrain", **lm})

    res = pd.DataFrame(results)
    st.session_state["results"] = res

if "results" in st.session_state:
    res = st.session_state["results"].copy()
    st.dataframe(res.round(3), use_container_width=True)

    st.subheader("AUROC by adaptation strategy")
    plot_auc = res.pivot(index="N local", columns="Strategy", values="AUROC")
    st.line_chart(plot_auc)

    st.subheader("Brier score by adaptation strategy — lower is better")
    plot_brier = res.pivot(index="N local", columns="Strategy", values="Brier")
    st.line_chart(plot_brier)

    # ======================================================
    # SCREEN 5 — TRANSPARENT RECOMMENDATION
    # ======================================================
    st.header("5. Model Transfer Readiness")

    # Choose best tested strategy using AUROC first, then Brier as tie-breaker.
    # In a real study this should be replaced by pre-specified clinical thresholds,
    # uncertainty intervals, subgroup checks, and external governance review.
    eligible = res.dropna(subset=["AUROC", "Brier"]).copy()
    best = eligible.sort_values(["AUROC", "Brier"], ascending=[False, True]).iloc[0]

    st.success(
        f"Best tested configuration in this run: {best['Strategy']} "
        f"using N={int(best['N local'])} local cases "
        f"(AUROC={best['AUROC']:.3f}, Brier={best['Brier']:.3f})."
    )

    st.write("**Interpretation:**")
    if best["Strategy"] == "Recalibration":
        st.write("The source model retained useful ranking, while local probability calibration improved.")
    elif best["Strategy"] == "Fine-tune source":
        st.write("Continuing from source-model weights gave the strongest tested source-informed adaptation.")
    elif best["Strategy"] == "Local retrain":
        st.write("In this simulation, learning only from local data performed best.")
    else:
        st.write("Direct transport performed as well as or better than the tested adaptation alternatives.")

    st.info(
        "This is an experimental recommendation, not deployment approval. "
        "A real workflow should add confidence intervals, subgroup fairness, temporal validation, "
        "clinical utility, drift monitoring, data-quality checks and human/governance approval."
    )
