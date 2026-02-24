import streamlit as st
from src.ml_model import predict_default_probability
from src.rag_engine import generate_response
import time

st.set_page_config(
    page_title="AI Credit Risk Advisor",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
    <style>
    .risk-card {
        padding: 25px;
        border-radius: 15px;
        background-color: #111827;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .risk-title {
        font-size: 22px;
        font-weight: bold;
    }
    .big-risk {
        font-size: 40px;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏦 AI Credit Risk Advisor")
st.caption("Hybrid Rule-Based + Machine Learning Credit Assessment System")

st.divider()


col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income", min_value=1000)
    credit = st.number_input("Loan Amount (Credit)", min_value=1000)
    annuity = st.number_input("Annual EMI (Annuity)", min_value=1000)

with col2:
    age = st.number_input("Age", min_value=18, max_value=100)
    years_employed = st.number_input("Years Employed", min_value=0)
    weak_credit = st.selectbox("Credit History", ["Good", "Weak"])

st.divider()


if st.button("🚀 Assess Risk"):

    with st.spinner("Analyzing financial profile..."):
        time.sleep(1)

        query_text = ""
        if weak_credit == "Weak":
            query_text = "weak credit history"

        explanation, rule_risk_label, dti = generate_response(
            query_text,
            income,
            annuity
        )

        default_prob, _ = predict_default_probability(
            income,
            credit,
            annuity,
            age,
            years_employed
        )


        rule_score = {"Low": 0, "Medium": 1, "High": 2}[rule_risk_label]

        if default_prob > 0.65:
            ml_score = 2
        elif default_prob > 0.4:
            ml_score = 1
        else:
            ml_score = 0

        final_score = (rule_score * 0.5) + (ml_score * 0.5)

        if final_score >= 1.2:
            final_risk = "High"
            color = "#ef4444"
        elif final_score >= 0.6:
            final_risk = "Medium"
            color = "#f59e0b"
        else:
            final_risk = "Low"
            color = "#10b981"

    st.divider()


    st.markdown(f"""
        <div class="risk-card">
            <div class="risk-title">Hybrid Risk Assessment Result</div>
            <div class="big-risk" style="color:{color};">{final_risk}</div>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    colA, colB, colC = st.columns(3)

    colA.metric("Rule-Based Risk", rule_risk_label)
    colB.metric("ML Default Probability", f"{default_prob * 100:.1f}%")
    colC.metric("Debt-to-Income Ratio", round(dti, 2))

    st.write("")
    st.progress(default_prob)

    st.write("")
    st.subheader("Explanation")
    st.text(explanation)