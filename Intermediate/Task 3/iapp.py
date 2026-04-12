import streamlit as st
import joblib
import numpy as np
import os

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_predictor.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Loan Prediction App", layout="centered")

st.title("🏦 Loan Approval Prediction System")

page = st.sidebar.radio("Navigation", ["Home", "Predict", "About"])

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("## 🏠 Welcome")
    st.write("Predict loan approval using ML model.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.markdown("## ℹ About")
    st.write("Built using Streamlit + Machine Learning")

# ---------------- PREDICT ----------------
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

    # ---------------- FINAL FEATURES (11 ONLY) ----------------
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
        property_area
    ]])

    # ---------------- CHECK ----------------
    if features.shape[1] != model.n_features_in_:
        st.error(f"""
❌ Feature mismatch!

Model expects: {model.n_features_in_}  
You provided: {features.shape[1]}

👉 Fix: model and app must match training data exactly.
""")
        st.stop()

    # ---------------- PREDICTION ----------------
    if st.button("Predict Loan Status"):

        prediction = model.predict(features)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(features)[0][1]
            st.success(f"Approval Probability: {round(prob * 100, 2)} %")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
