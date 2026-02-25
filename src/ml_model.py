import joblib
import pandas as pd

model_pipeline = joblib.load("credit_model.pkl")


def estimate_ext_sources(dti, income, years_employed):

    if dti < 0.3:
        base = 0.8
    elif dti < 0.5:
        base = 0.6
    elif dti < 0.8:
        base = 0.4
    elif dti < 1.0:
        base = 0.25
    else:
        base = 0.1

    income_bonus = 0.05 if income > 600000 else 0
    emp_bonus = 0.05 if years_employed > 5 else 0

    score = max(min(base + income_bonus + emp_bonus, 0.9), 0.05)

    return (
        score,
        max(score - 0.05, 0.05),
        max(score - 0.1, 0.05)
    )


def predict_default_probability(
    income,
    credit,
    annuity,
    age,
    years_employed
):

    dti = annuity / income

    days_birth = -(age * 365)
    days_employed = -(years_employed * 365)

    ext1, ext2, ext3 = estimate_ext_sources(dti, income, years_employed)

    input_data = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "DAYS_EMPLOYED": days_employed,
        "DAYS_BIRTH": days_birth,
        "DTI": dti
    }])

    prob = model_pipeline.predict_proba(input_data)[0][1]

    return round(float(prob), 3), round(dti, 3)