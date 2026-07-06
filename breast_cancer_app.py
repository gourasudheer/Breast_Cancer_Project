import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# --------------------------------------------------------
# Page Configuration
# --------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------------
# Load Model
# --------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# --------------------------------------------------------
# Feature Names
# --------------------------------------------------------
feature_names = [
'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
'mean compactness','mean concavity','mean concave points','mean symmetry',
'mean fractal dimension','radius error','texture error','perimeter error',
'area error','smoothness error','compactness error','concavity error',
'concave points error','symmetry error','fractal dimension error',
'worst radius','worst texture','worst perimeter','worst area',
'worst smoothness','worst compactness','worst concavity',
'worst concave points','worst symmetry','worst fractal dimension'
]

st.title("🩺 Breast Cancer Prediction using ANN")
st.write("Enter the patient measurements below.")

# --------------------------------------------------------
# Input
# --------------------------------------------------------
col1, col2 = st.columns(2)

inputs = []

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        value = col1.number_input(feature, value=0.0, format="%.6f")
    else:
        value = col2.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

# --------------------------------------------------------
# Prediction
# --------------------------------------------------------
if st.button("Predict"):

    sample = np.array(inputs).reshape(1, -1)

    sample = scaler.transform(sample)

    probability = model.predict(sample)[0][0]

    prediction = 1 if probability > 0.5 else 0

    st.subheader("Prediction")

    if prediction == 1:
        st.success("✅ Benign (Non-Cancerous)")
    else:
        st.error("⚠️ Malignant (Cancerous)")

    st.write(f"Prediction Probability : **{probability:.4f}**")