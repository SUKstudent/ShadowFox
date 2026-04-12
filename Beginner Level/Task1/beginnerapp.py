import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ----------------------------
# Load TFLite model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🐎 Horse vs 🧑 Human App")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📤 Predict", "ℹ️ About"])

# ----------------------------
# HOME PAGE
# ----------------------------
if page == "🏠 Home":
    st.title("Welcome 👋")
    st.write("""
    This is a deep learning app that classifies images as:
    
    - 🐎 Horse  
    - 🧑 Human  

    Upload an image in the Predict page to test the model.
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Horse_and_rider.jpg", use_container_width=True)

# ----------------------------
# PREDICT PAGE
# ----------------------------
elif page == "📤 Predict":
    st.title("Upload Image for Prediction")

    file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert("RGB").resize((150, 150))
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # preprocess
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # prediction
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if output > 0.5:
            st.success(f"Prediction: Human 🧑 ({output:.2f})")
        else:
            st.success(f"Prediction: Horse 🐎 ({1-output:.2f})")

# ----------------------------
# ABOUT PAGE
# ----------------------------
elif page == "ℹ️ About":
    st.title("About This Project")

    st.write("""
    This project is built using:

    - TensorFlow (CNN Model)
    - TensorFlow Lite (for lightweight deployment)
    - Streamlit (for web app)
    - Dataset: horses_or_humans (TFDS)

    ### Features:
    ✔ Image Classification  
    ✔ Deep Learning Model  
    ✔ Lightweight TFLite deployment  
    ✔ Simple UI with navigation  
    """)
