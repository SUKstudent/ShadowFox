import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ================= CONFIG =================
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Select Section", 
                          ["Dataset Overview", "Model Details", "Price Estimator"])

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Intermediate", "Task_2", "car_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "Intermediate", "Task_2", "car_price_prediction_model.pkl")

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

# ================= TITLE =================
st.title("🚗 Car Price Prediction App")
st.caption("Estimate car resale value using Machine Learning")

# ================= DATASET =================
if option == "Dataset Overview":
    st.subheader("📊 Cleaned Dataset")
    st.success("✅ Dataset cleaned (duplicates removed before training)")
    
    st.write(f"Shape: {df.shape}")
    st.dataframe(df)

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

    # Feature importance (bonus)
    try:
        importance = model.coef_
        features = ["Year", "Present Price", "Kms Driven",
                    "Diesel", "Petrol", "Seller", "Manual"]

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })

        st.subheader("📊 Feature Importance")
        st.bar_chart(imp_df.set_index("Feature"))
    except:
        st.info("Feature importance not available.")

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

    # Encoding
    fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_type_petrol = 1 if fuel_type == "Petrol" else 0
    seller_type_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    # Input
    input_data = np.array([[year, present_price, kms_driven,
                            fuel_type_diesel,
                            fuel_type_petrol,
                            seller_type_individual,
                            transmission_manual]])

    if st.button("Predict Price"):
        prediction = model.predict(input_data)[0]

        st.success(f"💰 Estimated Price: ₹ {round(prediction, 2)} lakhs")

        # Smart suggestion
        if prediction < present_price * 0.5:
            st.warning("⚠️ High depreciation detected. Consider selling soon.")
        else:
            st.success("✅ Good resale value!")

        # Download result
        result_df = pd.DataFrame({
            "Year": [year],
            "Predicted Price (Lakhs)": [prediction]
        })

        st.download_button("📥 Download Result",
                           result_df.to_csv(index=False),
                           "prediction.csv")

# ================= FOOTER =================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Internship Project")
