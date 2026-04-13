import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Custom dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Controls")
option = st.sidebar.radio("Choose Action", ["View Data", "Model Info", "Predict Price"])

# Load dataset
df = pd.read_csv("car.csv")

# Load model
model = joblib.load("car_price_prediction_model.pkl")

# Title
st.title("🚗 Car Price Prediction App")
st.caption("Estimate car resale value using Machine Learning")

# VIEW DATA
if option == "View Data":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df)

    st.markdown("---")

    st.subheader("📈 Dataset Statistics")
    st.write(df.describe())

# MODEL INFO
elif option == "Model Info":
    st.subheader("🤖 Model Information")

    st.write("""
    - Model Used: Linear Regression  
    - Problem Type: Regression  
    - Target Variable: Selling Price  
    """)

    st.markdown("---")

    st.subheader("📌 Features Used")
    st.write("""
    - Year  
    - Present Price  
    - Kilometers Driven  
    - Fuel Type  
    - Seller Type  
    - Transmission  
    """)

# PREDICTION
elif option == "Predict Price":
    st.subheader("🔮 Enter Car Details")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year of Purchase", 2000, 2024, 2015)
        present_price = st.number_input("Showroom Price (in lakhs)", 5.0)
        kms_driven = st.number_input("Kilometers Driven", 30000)

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    # Encoding
    fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
    seller_type_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    # Input format
    input_data = np.array([[year, present_price, kms_driven,
                            fuel_type_diesel,
                            seller_type_individual,
                            transmission_manual]])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)} lakhs")
