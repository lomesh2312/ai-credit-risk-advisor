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

## 🎯 Problem Statement

Traditional loan approval systems either:
- Rely only on rigid rule-based checks, OR
- Depend entirely on machine learning black-box models

This project solves that by creating a **hybrid intelligent risk framework** that ensures:

- Accurate probability prediction
- Regulatory-safe rule enforcement
- Human-readable explanations
- Consistent final decisions

---

## 🧠 System Architecture

### 1️⃣ Machine Learning Layer
- Model: Random Forest Classifier
- Predicts: Loan Default Probability
- Features:
  - Income
  - Loan Amount (Credit)
  - Annual EMI (Annuity)
  - Age
  - Years Employed
  - Debt-to-Income Ratio (DTI)

---

### 2️⃣ Rule-Based Risk Engine
Evaluates structured financial risk factors:

- Weak credit history
- Low income
- High debt burden
- Debt-to-Income Ratio thresholds

Risk Levels:
- Low
- Medium
- High

---

### 3️⃣ Hybrid Risk Decision System

Final Risk = Weighted Combination of:
- ML Score
- Rule-Based Score

🔒 Safety Clamp Logic:
Rule-based risk cannot be overridden downward by ML prediction.

This ensures regulatory consistency and financial safety.

---

### 4️⃣ RAG (Retrieval-Augmented Generation)

- Knowledge base stored in loan guidelines file
- FAISS vector index for semantic search
- Sentence Transformers for embeddings
- Generates contextual financial explanations

---

### 5️⃣ Sentiment Analysis

Analyzes textual input sentiment using a fine-tuned transformer model.

---

## 📊 Output Example

The system provides:

- Hybrid Risk Assessment Result
- Rule-Based Risk
- ML Default Probability
- Debt-to-Income Ratio
- Structured Explanation
- Risk Color Indicator

---

## 🛠 Tech Stack

- Python
- Streamlit (UI)
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
git clone https://github.com/lomesh2312/ai-credit-risk-advisor.git
cd AI-Credit-Risk-Advisor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using Streamlit Cloud.

---

## 🏦 Key Design Decisions

- ML predicts structural financial risk
- Rules capture behavioral red flags
- Hybrid model prevents underestimation
- Explainability prioritized for transparency
- Modular architecture for scalability

---

## 📈 Why This Project Is Strong

✔ End-to-End ML Pipeline  
✔ Feature Engineering  
✔ Hybrid Decision Framework  
✔ Regulatory-Safe Override Logic  
✔ RAG-Based Explainable AI  
✔ Deployed Web Application  

This is not just a prediction model —  
it is a structured AI risk evaluation system.

---

## 👨‍💻 Author

Developed as an intelligent fintech risk assessment prototype combining machine learning, explainable AI, and hybrid risk modeling.

---

## 📌 Future Improvements

- Feature Importance Visualization
- Risk Probability Gauge
- PDF Report Generation
- Model Retraining Pipeline
- Advanced Credit Behavior Features

---

### 🔥 Final Note

AI Credit Risk Advisor demonstrates how machine learning and rule-based systems can work together to build safer, explainable, and production-ready financial AI systems.