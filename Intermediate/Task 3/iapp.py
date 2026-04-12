import streamlit as st
import joblib
import numpy as np
import os

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_predictor.pkl")
model = joblib.load(MODEL_PATH)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.title("🏦 Loan Approval Prediction System")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio("Navigation", ["Predict", "About"])

# ---------------- ABOUT PAGE ----------------
if page == "About":
    st.write("""
    ### 🏦 Loan Prediction App
    This app predicts whether a loan will be approved or rejected.

    Built using:
    - Streamlit
    - Machine Learning (Sklearn)
    - Python
    """)

# ---------------- PREDICTION PAGE ----------------
elif page == "Predict":

    st.subheader("Enter Applicant Details")

    # Inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Amount Term", min_value=0)
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # ---------------- ENCODING ----------------
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_employed = 1 if self_employed == "Yes" else 0

    if dependents == "3+":
        dependents = 3
    else:
        dependents = int(dependents)

    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # ---------------- FEATURE ENGINEERING ----------------
    total_income = applicant_income + coapplicant_income
    loan_ratio = loan_amount / (total_income + 1)
    emi = loan_amount / (loan_term + 1)

    # ---------------- FEATURE ARRAY ----------------
    features = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area,
        total_income,
        loan_ratio,
        emi
    ]])

    # ---------------- SAFETY CHECK ----------------
    expected_features = model.n_features_in_
    given_features = features.shape[1]

    st.write("Model expects:", expected_features)
    st.write("You provided:", given_features)

    # 🔥 FIX: ALIGN FEATURES SAFELY
    if given_features != expected_features:
        st.warning("⚠ Feature mismatch detected. Auto-adjusting input...")

        if given_features > expected_features:
            features = features[:, :expected_features]

        else:
            features = np.pad(features, ((0, 0), (0, expected_features - given_features)), "constant")

    # ---------------- PREDICTION ----------------
    if st.button("Predict Loan Status"):

        prediction = model.predict(features)[0]

        # Probability (if available)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0][1]
            st.success(f"Approval Probability: {round(prob * 100, 2)} %")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
