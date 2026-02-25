import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv("data/application_train.csv")

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
        n_jobs=-1,
        class_weight="balanced"
    ))
])

model_pipeline.fit(X, y)

joblib.dump(model_pipeline, "credit_model.pkl")

print("✅ Strong RandomForest model saved")