
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# =========================
# LOAD MODELS
# =========================
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# =========================
# APP
# =========================
app = FastAPI(
    title="Loan Approval API 🚀",
    version="1.0"
)

# =========================
# INPUT SCHEMA
# =========================
class LoanRequest(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

# =========================
# HOME
# =========================
@app.get("/")
def home():
    return {"status": "API running 🚀"}

# =========================
# PREDICT
# =========================
@app.post("/predict")
def predict(data: LoanRequest):

    df = pd.DataFrame([data.dict()])

    # clean text
    df["education"] = df["education"].strip().title()
    df["self_employed"] = df["self_employed"].strip().title()

    # encode
    df = pd.get_dummies(df)

    # align columns
    df = df.reindex(columns=features, fill_value=0)

    # scale
    df_scaled = scaler.transform(df)

    # probability
    prob = model.predict_proba(df_scaled)[0][1]

    # decision logic (3-level system)
    if prob >= 0.65:
        decision = "Approved ✅"
        risk = "Low"
    elif prob >= 0.45:
        decision = "Review ⚠️"
        risk = "Medium"
    else:
        decision = "Rejected ❌"
        risk = "High"

    return {
        "probability": round(float(prob), 2),
        "decision": decision,
        "risk_level": risk
    }
