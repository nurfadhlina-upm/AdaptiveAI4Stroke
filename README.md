# AdaptAI Transferability Demo

This package is a runnable research prototype for:

1. source-target data comparison,
2. transferability assessment,
3. gap/risk profiling,
4. adaptation simulation,
5. transparent model-transfer recommendation.

## Files

- `app.py` — Streamlit dashboard.
- `source_hospital.csv` — 8,000 synthetic source-hospital records.
- `target_hospital.csv` — 1,200 synthetic target-hospital records with deliberate distribution shift.
- `source_model.joblib` — trained source MLP model + scaler + feature metadata.
- `quick_test.py` — minimal example for loading/testing the model.
- `requirements.txt` — Python dependencies.

## Run

Open a terminal in this folder:

    pip install -r requirements.txt
    streamlit run app.py

Then open the local Streamlit URL shown in the terminal.

## Variables

The demonstration model uses:
- age
- systolic blood pressure
- glucose
- BMI
- hypertension
- atrial fibrillation
- smoking
- prior TIA
- NIHSS

`outcome` is a synthetic binary outcome.

## What "fine-tune source" means here

The source model is an `MLPClassifier`. The saved source weights are copied and then
continued on labelled target cases with `partial_fit()`. This is a genuine
source-weight continuation demonstration, but it is intentionally simple.

It is NOT yet the final clinical adaptation method. A publication-grade version
should compare multiple adaptation methods and use nested/temporal validation,
bootstrap confidence intervals, calibration curves, subgroup analysis and
pre-specified decision criteria.

## Important

All patient records and outcomes are synthetic. The model is for software/research
demonstration only and must not be used for patient care.
