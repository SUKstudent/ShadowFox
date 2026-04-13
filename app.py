import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Dark theme styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Select Section", ["Dataset Overview", "Model Details", "Price Estimator"])

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct paths
DATA_PATH = os.path.join(BASE_DIR, "Intermediate", "Task_2", "car.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Intermediate", "Task_2", "car_price_prediction_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Load model
model = joblib.load(MODEL_PATH)

# Title
st.title("🚗 Car Price Prediction App")
st.caption("Estimate car resale value using Machine Learning")

# ================= DATASET =================
if option == "Dataset Overview":
    st.subheader("📊 Dataset Snapshot")
    st.dataframe(df)

    st.markdown("---")

    st.subheader("📈 Statistical Summary")
    st.write(df.describe())

# ================= MODEL =================
elif option == "Model Details":
    st.subheader("🤖 About the Model")

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

# ================= PREDICTION =================
elif option == "Price Estimator":
    st.subheader("🔮 Predict Car Price")

    col1, col2 = st.columns(2)

    with col1:
        year = st.number_input("Year of Purchase", 2000, 2024, 2015)
        present_price = st.number_input("Showroom Price (in lakhs)", 5.0)
        kms_driven = st.number_input("Kilometers Driven", 30000)

    with col2:
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    # Correct encoding (matches training)
    fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
    seller_type_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    # FINAL input format (matches model)
    input_data = np.array([[year, present_price, kms_driven,
                            fuel_type_diesel,
                            fuel_type_petrol,
                            seller_type_individual,
                            transmission_manual]])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)} lakhs") 
