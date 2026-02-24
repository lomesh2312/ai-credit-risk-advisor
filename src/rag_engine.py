from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import re
from ml_model import predict_default_probability

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    return np.array(embeddings).astype("float32")


def load_knowledge_base():
    with open("knowledge_base/loan_guidelines.txt", "r") as f:
        text = f.read()

    chunks = [line.strip() for line in text.split("\n") if line.strip()]
    return chunks


def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_similar(query, index, chunks, k=4):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k=k)
    return [chunks[i] for i in indices[0]]


def extract_financial_features(query):
    income = credit = annuity = age = years_employed = None

    patterns = {
        "income": r"income\s*(\d+)",
        "credit": r"credit\s*(\d+)",
        "annuity": r"annuity\s*(\d+)",
        "age": r"age\s*(\d+)",
        "employed": r"employed\s*(\d+)"
    }

    query_lower = query.lower()

    for key, pattern in patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            value = int(match.group(1))
            if key == "income":
                income = value
            elif key == "credit":
                credit = value
            elif key == "annuity":
                annuity = value
            elif key == "age":
                age = value
            elif key == "employed":
                years_employed = value

    return income, credit, annuity, age, years_employed


def generate_response(query, income=None, annuity=None):

    risk_score = 0
    reasons = []
    query_lower = query.lower()

    if "weak credit history" in query_lower:
        risk_score += 2
        reasons.append("Weak credit history indicates poor repayment behavior.")

    if "low income" in query_lower:
        risk_score += 2
        reasons.append("Low income reduces repayment capacity.")

    if "high debt" in query_lower:
        risk_score += 2
        reasons.append("High debt increases default probability.")

    dti = None
    if income and annuity and income > 0:
        dti = annuity / income

        if dti > 0.5:
            risk_score += 2
            reasons.append("Debt-to-Income ratio is extremely high (>50%).")
        elif dti > 0.3:
            risk_score += 1
            reasons.append("Debt-to-Income ratio is moderately high (30–50%).")

    # Risk classification
    if risk_score >= 4:
        risk_level = "High"
    elif risk_score >= 2:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    explanation = f"Risk Level (Rule-Based): {risk_level}\n\n"
    explanation += "Reasons:\n"

    if reasons:
        for r in reasons:
            explanation += f"- {r}\n"
    else:
        explanation += "- No major risk indicators detected.\n"

    if dti is not None:
        explanation += f"\nCalculated Debt-to-Income Ratio: {round(dti,2)}\n"

    return explanation, risk_level, dti


sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    return result["label"], result["score"]


if __name__ == "__main__":

    print("\n🔹 Building Knowledge Base...\n")

    chunks = load_knowledge_base()
    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)


    query = "Customer income 30000 credit 500000 annuity 45000 age 23 employed 1 and has weak credit history"

    similar_docs = search_similar(query, index, chunks)

    income, credit, annuity, age, years_employed = extract_financial_features(query)

    answer, rule_risk_label, dti = generate_response(
        query,
        income,
        annuity
    )

    sentiment_label, sentiment_score = analyze_sentiment(query)

    print("\n🔹 AI Credit Advisor Report\n")
    print(answer)

    default_prob = None

    if all(v is not None for v in [income, credit, annuity, age, years_employed]):

        default_prob, dti_from_ml = predict_default_probability(
            income=income,
            credit=credit,
            annuity=annuity,
            age=age,
            years_employed=years_employed
        )

        print(f"\nDefault Probability (ML Model): {default_prob}")

    else:
        print("\nFinancial data not sufficient for ML prediction.")

    print(f"\nSentiment: {sentiment_label} ({round(sentiment_score,2)})")

    
    if default_prob is not None:

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
        elif final_score >= 0.6:
            final_risk = "Medium"
        else:
            final_risk = "Low"

        print(f"\nFinal Risk Decision (Hybrid Model): {final_risk}")