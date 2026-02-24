import joblib
import pandas as pd
import os

# Load trained model
model = joblib.load("models/credit_model.pkl")

def analyze_risk(input_data):
    df = pd.DataFrame([input_data])
    
    probability = model.predict_proba(df)[0][1]
    
    return probability

def generate_explanation(input_data, probability):
    reasons = []
    
    if input_data["AMT_INCOME_TOTAL"] < 100000:
        reasons.append("Low income level")
        
    if input_data["EXT_SOURCE_1"] < 0.3:
        reasons.append("Weak credit history")
        
    if probability > 0.7:
        risk_level = "High Risk"
    elif probability > 0.4:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
        
    return risk_level, reasons



if __name__ == "__main__":
    
    df = pd.read_csv("data/application_train.csv", encoding="latin1")
    
    X = df.drop("TARGET", axis=1)
    
    sample_user = X.iloc[0].to_dict()
    
    prob = analyze_risk(sample_user)
    risk, reasons = generate_explanation(sample_user, prob)
    
    print("Probability:", prob)
    print("Risk Level:", risk)
    print("Reasons:", reasons)