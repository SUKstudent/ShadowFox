```python
# ============================================
# 📌 IMAGE CLASSIFICATION WEB APP (STREAMLIT)
# ============================================

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from io import BytesIO

# ============================================
# 🔧 PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🤖",
    layout="centered"
)

# ============================================
# 🎨 CUSTOM STYLING
# ============================================

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #333;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# 🧠 LOAD MODEL (CACHED)
# ============================================

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("Horses and Humans.h5")
        return model
    except Exception as e:
        st.error("❌ Model not found. Make sure 'Horses and Humans.h5' is in repo.")
        st.stop()

model = load_model()

# ============================================
# 🏷️ CLASS LABELS
# ============================================

CLASS_NAMES = ["horses", "humans"]

# ============================================
# 🖼️ IMAGE PREPROCESSING FUNCTION
# ============================================

def preprocess_image(img):
    try:
        img = img.convert("RGB")
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error("⚠️ Error processing image")
        return None

# ============================================
# 🔍 PREDICTION FUNCTION
# ============================================

def predict_image(img_array):
    try:
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        if confidence > 0.5:
            label = CLASS_NAMES[1]
            confidence_score = confidence
        else:
            label = CLASS_NAMES[0]
            confidence_score = 1 - confidence

        return label, confidence_score
    except Exception as e:
        st.error("⚠️ Prediction failed")
        return None, None

# ============================================
# 🌐 LOAD IMAGE FROM URL
# ============================================

def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        st.error("❌ Invalid image URL")
        return None

# ============================================
# 🏠 UI HEADER
# ============================================

st.markdown('<div class="title">🤖 AI Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Classify images as Horses or Humans</div>', unsafe_allow_html=True)

st.write("---")

# ============================================
# 📥 INPUT OPTIONS
# ============================================

option = st.radio(
    "Choose Input Method:",
    ["Upload Image", "Image URL"]
)

image = None

# ============================================
# 📤 FILE UPLOAD
# ============================================

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# ============================================
# 🌐 URL INPUT
# ============================================

elif option == "Image URL":
    url = st.text_input("Enter Image URL")

    if url:
        image = load_image_from_url(url)

# ============================================
# 🖼️ DISPLAY IMAGE
# ============================================

if image is not None:
    st.image(image, caption="Selected Image", use_column_width=True)

    st.write("---")

    # ============================================
    # 🔍 PREDICT BUTTON
    # ============================================

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):

            img_array = preprocess_image(image)

            if img_array is not None:
                label, confidence = predict_image(img_array)

                if label:
                    st.success(f"✅ Prediction: {label.upper()}")
                    st.info(f"📊 Confidence: {confidence * 100:.2f}%")

# ============================================
# 📊 SIDEBAR INFO
# ============================================

st.sidebar.header("ℹ️ About")

st.sidebar.write("""
This app uses a Deep Learning model built with TensorFlow.

🔹 Model Type: CNN  
🔹 Classes: Horses vs Humans  
🔹 Input Size: 150x150  
""")

st.sidebar.write("---")

st.sidebar.header("📌 Instructions")

st.sidebar.write("""
1. Upload an image OR paste URL  
2. Click Predict  
3. View results instantly  
""")

# ============================================
# 🚨 ERROR HANDLING
# ============================================

st.write("---")

st.caption("⚡ Built with Streamlit & TensorFlow")
```
