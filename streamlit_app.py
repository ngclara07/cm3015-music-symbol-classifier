"""
Streamlit App: Music Symbol Classifier
--------------------------------------

This app demonstrates single-image prediction using the final best-performing
Support Vector Machine classifier.
"""

import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from PIL import Image


# =======================
# CONFIGURATION
# =======================

MODEL_PATH = Path("saved_models/music_symbol_svm_bundle.joblib")


# =======================
# LOAD MODEL
# =======================

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run the SVM model-saving cell first."
        )

    return joblib.load(MODEL_PATH)


# =======================
# PREPROCESSING
# =======================

def preprocess_uploaded_image(uploaded_file, image_size=(64, 64)):
    """
    Preprocess uploaded image so that it matches the training representation:
    - grayscale conversion
    - non-white content cropping
    - square centring on white background
    - resizing to 64x64
    - flattening and normalisation to [0, 1]
    """
    img = Image.open(uploaded_file).convert("L")

    arr = np.asarray(img)
    mask = arr < 250

    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        img = img.crop((x0, y0, x1, y1))

    w, h = img.size
    side = max(w, h)

    square = Image.new("L", (side, side), color=255)
    square.paste(img, ((side - w) // 2, (side - h) // 2))

    square = square.resize(image_size)

    img_array = np.asarray(square, dtype=np.float32) / 255.0
    img_vector = img_array.flatten().reshape(1, -1)

    return square, img_vector


# =======================
# STREAMLIT UI
# =======================

st.set_page_config(
    page_title="Music Symbol Classifier",
    page_icon="🎼",
    layout="centered"
)

st.title("Music Symbol Image Classifier")

st.write(
    "This prototype uses the final best-performing Support Vector Machine "
    "classifier to classify uploaded music-symbol images into one of ten "
    "musical notation classes."
)

bundle = load_model_bundle()

svm_model = bundle["model"]
label_encoder = bundle["label_encoder"]
class_names = bundle["class_names"]
image_size = bundle["image_size"]

st.sidebar.header("Model Information")
st.sidebar.write(f"Model: {bundle['model_name']}")
st.sidebar.write(f"Kernel: {bundle.get('kernel', 'rbf')}")
st.sidebar.write(f"C: {bundle.get('C', 'N/A')}")
st.sidebar.write(f"Gamma: {bundle.get('gamma', 'N/A')}")
st.sidebar.write(f"Classes: {len(class_names)}")

uploaded_file = st.file_uploader(
    "Upload a music symbol image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    processed_img, img_vector = preprocess_uploaded_image(
        uploaded_file,
        image_size=image_size
    )

    predicted_encoded = svm_model.predict(img_vector)[0]
    predicted_class = label_encoder.inverse_transform([predicted_encoded])[0]

    st.subheader("Uploaded Image After Preprocessing")
    st.image(processed_img, caption="64×64 preprocessed input", width=200)

    st.subheader("Prediction Result")
    st.success(f"Predicted class: **{predicted_class}**")

    if hasattr(svm_model, "decision_function"):
        decision_scores = svm_model.decision_function(img_vector)

        if decision_scores.ndim == 2:
            scores = decision_scores[0]
            top_indices = np.argsort(scores)[::-1][:3]

            st.subheader("Top Decision Scores")

            top_predictions = {
                label_encoder.inverse_transform([idx])[0]: round(float(scores[idx]), 4)
                for idx in top_indices
            }

            st.write(top_predictions)

else:
    st.info("Upload an image to obtain a prediction.")
