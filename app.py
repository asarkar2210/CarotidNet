import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="CarotidNet", layout="wide", page_icon="logo.png")


MODEL_PATH = 'final_multimodal_model.h5'
SCALER_PATH = 'scaler.pkl'


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


col_logo, col_title = st.columns([1, 15])

with col_logo:
    try:
        st.image("logo.png", width=60)
    except FileNotFoundError:
        st.markdown("<h1 style='text-align: center;'>🩸</h1>", unsafe_allow_html=True)

with col_title:
    st.title("CarotidNet")

st.markdown("### Multimodal Stroke Risk Predictor")

col1, col2 = st.columns(2)


with col1:
    st.header("1. Patient Vitals")
    age = st.number_input("Age", min_value=1, max_value=100, value=65)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=28.5)
    glucose = st.number_input("Avg Glucose Level", min_value=50.0, max_value=300.0, value=105.0)
    
    st.subheader("History")
    hypertension = st.selectbox("Hypertension?", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease?", ["No", "Yes"])
    

    hyp_val = 1 if hypertension == "Yes" else 0
    hd_val = 1 if heart_disease == "Yes" else 0
    
    # Categorical Inputs (For display only - simplified for demo)
    gender = st.selectbox("Gender", ["Male", "Female"])
    smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
    


with col2:
    st.header("2. Ultrasound Scan")
    uploaded_file = st.file_uploader("Upload Carotid Ultrasound (TIFF/JPG)", type=["tiff", "tif", "jpg", "png"])
    
    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Scan")
        
        
        img_array = np.array(image.convert('RGB'))
        img_array = cv2.resize(img_array, (224, 224))
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Batch shape (1, 224, 224, 3)
    else:
        
        placeholder = np.full((300, 400, 3), 220, dtype=np.uint8)
        st.image(placeholder, caption="No Scan Uploaded", use_column_width=True)


st.write("") 
_, col_btn, _ = st.columns([1, 2, 1])

if col_btn.button("Analyse Risk", type="primary", use_container_width=True):
    if uploaded_file is None:
        st.warning("⚠️ Please upload an ultrasound image first.")
    else:

        raw_data = np.array([[age, hyp_val, hd_val, glucose, bmi, 0, 1, 2, 1, 1]])
        

        scaled_data = scaler.transform(raw_data)
        

        prediction = model.predict([img_array, scaled_data])
        risk_score = float(prediction[0][0])
        

        st.divider()
        st.markdown(f"<h2 style='text-align: center;'>Predicted Stroke Risk: <strong>{risk_score*100:.2f}%</strong></h2>", unsafe_allow_html=True)
        
        if risk_score >= 0.70:
            st.error("🚨 HIGH RISK: Significant indicators present. Comprehensive clinical evaluation recommended.")
        elif risk_score >= 0.40:
            st.warning("⚠️ ELEVATED RISK: Moderate indicators present. Neurological screening advised.")
        else:
            st.success("✅ LOW RISK: Baseline indicators normal. Maintain routine check-ups.")
            
        st.info("ℹ️ **Clinical Disclaimer:** This tool is an academic demonstration of machine learning capabilities. It is not an FDA-approved diagnostic device. All predictions must be verified by a qualified healthcare professional.")