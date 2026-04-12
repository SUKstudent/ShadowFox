import streamlit as st
import joblib
import numpy as np
import os

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_predictor.pkl")
model = joblib.load(MODEL_PATH)

st.title("🏦 Loan Approval Prediction App")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "About"])

# ---------------- ABOUT ----------------
if page == "About":
    st.write("ML Loan Prediction App 🚀 built using Streamlit")

# ---------------- PREDICTION ----------------
elif page == "Predict":

    st.subheader("Enter Details")

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

    dependents = 3 if dependents == "3+" else int(dependents)
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    # ---------------- FEATURE ENGINEERING ----------------
    total_income = applicant_income + coapplicant_income
    loan_income_ratio = loan_amount / (total_income + 1)
    emi = loan_amount / (loan_term + 1)

    # ---------------- CREATE INPUT ----------------
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
        loan_income_ratio,
        emi
    ]])

    # ---------------- AUTO SAFETY FIX ----------------
    expected = model.n_features_in_
    actual = features.shape[1]

    st.write("Expected features:", expected)
    st.write("Given features:", actual)

    # 🔥 FIX: adjust automatically if mismatch
    if actual != expected:
        st.warning("Adjusting features automatically to match model...")

        if actual > expected:
            features = features[:, :expected]  # trim extra
        else:
            features = np.pad(features, ((0,0),(0, expected-actual)), 'constant')

    # ---------------- PREDICTION ----------------
    if st.button("Predict Loan Status"):
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
