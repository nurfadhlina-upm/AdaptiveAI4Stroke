
"""
quick_test.py
A minimal non-Streamlit example showing how to load the model and
measure direct source-to-target performance.
"""
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

bundle = joblib.load("source_model.joblib")
target = pd.read_csv("target_hospital.csv")

features = bundle["features"]
medians = bundle["source_medians"]

X = target[features].copy()
for col in features:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(medians[col])

X_scaled = bundle["scaler"].transform(X)
probability = bundle["model"].predict_proba(X_scaled)[:, 1]
y = target["outcome"]

print("Model:", bundle["model_name"])
print("Target AUROC:", round(roc_auc_score(y, probability), 3))
print("Target AUPRC:", round(average_precision_score(y, probability), 3))
print("Target Brier:", round(brier_score_loss(y, probability), 3))
