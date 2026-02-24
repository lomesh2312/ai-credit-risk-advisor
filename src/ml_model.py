import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/application_train.csv")

data = data.copy()

data["DTI"] = data["AMT_ANNUITY"] / data["AMT_INCOME_TOTAL"]

features = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH",
    "DTI"
]

target = "TARGET"

data = data[features + [target]]
data = data.fillna(data.median())

X = data[features]
y = data[target]


model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ))
])

model_pipeline.fit(X, y)


def estimate_ext_sources(dti, income, years_employed):


    if dti < 0.3:
        base_score = 0.8
    elif dti < 0.5:
        base_score = 0.6
    elif dti < 0.8:
        base_score = 0.4
    elif dti < 1.0:
        base_score = 0.25
    else:
        base_score = 0.1   


    income_bonus = 0.05 if income > 600000 else 0


    emp_bonus = 0.05 if years_employed > 5 else 0

    final_score = max(min(base_score + income_bonus + emp_bonus, 0.9), 0.05)


    return (
        final_score,
        max(final_score - 0.05, 0.05),
        max(final_score - 0.1, 0.05)
    )


def predict_default_probability(
    income,
    credit,
    annuity,
    age,
    years_employed
):

    dti = annuity / income


    days_birth = - (age * 365)
    days_employed = - (years_employed * 365)

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