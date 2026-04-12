import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("loan_predictor.pkl")

st.title("🏦 Loan Approval Prediction App")

st.write("Fill the details below to check loan approval status.")

# Inputs
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

# Encoding
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

dependents = 3 if dependents == "3+" else int(dependents)

property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Prediction button
if st.button("Predict Loan Status"):
    features = np.array([[gender, married, dependents, education,
                          self_employed, applicant_income,
                          coapplicant_income, loan_amount,
                          loan_term, credit_history,
                          property_area]])

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
