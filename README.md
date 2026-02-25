# 🏦 AI Credit Risk Advisor  
### Hybrid ML + Rule-Based + RAG Powered Loan Risk Assessment System

---

## 🚀 Overview

AI Credit Risk Advisor is an intelligent loan risk assessment system that combines:

- 📊 Machine Learning (Random Forest Model)
- 📏 Rule-Based Risk Evaluation
- 📚 Retrieval-Augmented Generation (RAG)
- 🤖 LLM-powered Explanation Engine
- 🎯 Hybrid Decision Logic with Safety Override

The system predicts loan default probability and generates structured, explainable financial risk reports.

---

## 🎯 What Problem Does This Solve?

Traditional loan approval systems either:

- Depend only on rigid rule-based checks  
- Or rely entirely on black-box machine learning models  

This system combines both approaches to ensure:

- Accurate probability prediction  
- Regulatory-safe rule enforcement  
- Transparent explanations  
- Consistent final decisions  

---

# 👤 User Inputs

The user must provide the following financial details:

1. **Annual Income**
2. **Loan Amount (Credit)**
3. **Annual EMI (Annuity)**
4. **Age**
5. **Years Employed**
6. **Credit History**
   - Good
   - Weak

---

# 📊 System Outputs

After submission, the system generates:

### ✅ Hybrid Risk Assessment Result
Final decision: **Low / Medium / High**

### 📏 Rule-Based Risk
Risk derived from financial rules and thresholds

### 🤖 ML Default Probability
Predicted probability of loan default (percentage)

### 📉 Debt-to-Income Ratio (DTI)
Calculated as:

DTI = Annual EMI / Annual Income

### 📝 Explanation
Clear structured explanation including:
- Detected risk indicators
- Credit behavior impact
- DTI interpretation
- Final reasoning

### 🎨 Risk Color Indicator
- 🟢 Green → Low Risk  
- 🟡 Orange → Medium Risk  
- 🔴 Red → High Risk  

---

## 🧠 System Architecture

### 1️⃣ Machine Learning Layer
- Model: Random Forest Classifier
- Predicts loan default probability
- Uses structured financial features

---

### 2️⃣ Rule-Based Risk Engine
Evaluates:
- Weak credit history
- High debt burden
- Low income
- DTI thresholds

---

### 3️⃣ Hybrid Decision Logic

Final Risk = Weighted combination of:
- ML risk score
- Rule-based score

🔒 **Safety Clamp Logic Applied**
Rule-based risk cannot be overridden downward by ML.
This ensures financial and regulatory consistency.

---

### 4️⃣ RAG (Retrieval-Augmented Generation)

- Uses FAISS vector search
- Embedding model: SentenceTransformers
- Retrieves relevant loan guidelines
- Generates contextual financial explanations

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- FAISS
- SentenceTransformers
- HuggingFace Transformers
- NumPy

---

## 📂 Project Structure

```
AI-Credit-Risk-Advisor/
│
├── app.py
├── src/
│   └── ml_model.py
├── rag_engine.py
├── save_model.py
├── credit_model.pkl
├── knowledge_base/
│   └── loan_guidelines.txt
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone <your-repo-url>
cd AI-Credit-Risk-Advisor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

Deployed using Streamlit Cloud.

---

## 📈 Why This Project Is Strong

✔ End-to-End ML Pipeline  
✔ Hybrid Risk Framework  
✔ Regulatory-Safe Override Logic  
✔ Explainable AI with RAG  
✔ Clean Web Interface  
✔ Public Deployment  

This is not just a prediction model —  
it is a structured AI-powered financial risk evaluation system.

---

## 🚀 Future Enhancements

- Feature Importance Visualization
- Risk Probability Gauge
- PDF Report Generation
- Model Retraining Pipeline
- Advanced Behavioral Credit Features

---

## 👨‍💻 Author

Developed as an intelligent fintech risk assessment prototype combining machine learning, explainable AI, and hybrid risk modeling.