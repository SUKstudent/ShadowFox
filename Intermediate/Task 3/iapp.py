import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("Intermediate/Task 3/loan_model.pkl")

# ---------------- NAVIGATION ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Prediction", "ℹ️ About"])

# ---------------- HOME PAGE ----------------
if page == "🏠 Home":
    st.title("🏦 Loan Approval System")
    st.write("Welcome! Use this app to predict loan approval.")
    st.image("https://img.icons8.com/color/480/bank-building.png", width=200)

# ---------------- PREDICTION PAGE ----------------
elif page == "🔮 Prediction":
    st.title("Loan Prediction")

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

    if st.button("Predict"):
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

# ---------------- ABOUT PAGE ----------------
elif page == "ℹ️ About":
    st.title("About This App")
    st.write("""
    This is a Machine Learning project for Loan Approval Prediction.
    
    Built using:
    - Python 🐍
    - Streamlit 🌐
    - Scikit-learn 🤖
    """)
