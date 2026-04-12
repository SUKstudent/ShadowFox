import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("Intermediate/Task 3/loan_predictor.pkl")

st.title("🏦 Loan Approval Prediction App")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

# ---------------- HOME ----------------
if page == "Home":
    st.write("Welcome to Loan Prediction App 🚀")

# ---------------- PREDICT ----------------
elif page == "Predict":

    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    applicant_income = st.number_input("Applicant Income")
    coapplicant_income = st.number_input("Coapplicant Income")
    loan_amount = st.number_input("Loan Amount")
    loan_term = st.number_input("Loan Amount Term")
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

    # ---------------- FEATURE ARRAY (FIXED ORDER) ----------------
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

    st.write("Feature shape:", features.shape)

    # ---------------- PREDICTION ----------------
    if st.button("Predict Loan Status"):
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

# ---------------- ABOUT ----------------
elif page == "About":
    st.write("ML Loan Prediction App using Streamlit 🚀")
