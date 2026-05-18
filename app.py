# app.py
import os
import io
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Pneumonia Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAMES = ["xray_model.keras", "xray_model.h5"]  # search order
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
# ----------------------------

st.title("🩺 Pneumonia Detector — Chest X-ray")
st.write("Upload chest X-ray images and the model will predict Normal vs Pneumonia.")

# ---------- Helper: load model (cached) ----------
@st.cache_resource(show_spinner=False)
def load_saved_model():
    """Try to load model from the working directory using preferred filenames."""
    last_error = None
    for fname in MODEL_FILENAMES:
        if os.path.exists(fname):
            try:
                model = load_model(fname)
                return model, fname, None
            except Exception as e:
                last_error = f"Failed to load {fname}: {e}"
    return None, None, last_error or "No model file found."

model, loaded_fname, load_err = load_saved_model()

# ---------- Sidebar: options ----------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Prediction threshold (PNEUMONIA > threshold)", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("---")
st.sidebar.markdown("Model search order:")
for f in MODEL_FILENAMES:
    st.sidebar.text(f"- {f}")

# ---------- Model status ----------
if model is None:
    st.error("No model loaded.")
    if load_err:
        st.info(load_err)
    st.warning("Place your model file (xray_model.keras or xray_model.h5) in the app folder and rerun.")
    st.stop()
else:
    st.success(f"Loaded model: {loaded_fname}")

# ---------- File uploader ----------
file_types = ["jpg", "jpeg", "png"]
uploaded_files = st.file_uploader("Upload X-ray images", type=file_types, accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more chest X-ray images to get predictions.")
    st.stop()

# ---------- Prediction helpers ----------
def preprocess_pil(img_pil: Image.Image):
    """Resize and scale PIL image to model input."""
    img = img_pil.convert("RGB").resize(IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), img  # batch dimension + resized PIL image

def predict_batch(model, batch_array):
    """Return probabilities for batch (shape: (n,))."""
    preds = model.predict(batch_array, verbose=0)
    preds = np.array(preds).reshape(-1)
    return preds

# ---------- Run predictions ----------
results = []
with st.spinner("Running predictions..."):
    for up in uploaded_files:
        if up is None:
            continue
        try:
            image_bytes = up.read()
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            st.error(f"Error reading {up.name}: {e}")
            continue

        x, resized_img = preprocess_pil(pil_img)
        prob = predict_batch(model, x)[0]

        if prob > threshold:
            label = "(Pneumonia)"
            percentage = prob * 100
        else:
            label = "(Normal)"
            percentage = (1 - prob) * 100

        results.append({
            "filename": getattr(up, "name", "uploaded_image"),
            "pred_label": label,
            "percentage": f"{percentage:.2f}%",
            "resized_image": resized_img
        })

# ---------- Display results ----------
st.subheader("Predictions")
cols = st.columns(2)
for i, res in enumerate(results):
    col = cols[i % 2]
    with col:
        st.image(res["resized_image"], caption=f"{res['filename']}", use_column_width=True)
        st.markdown(f"**Result:** {res['pred_label']} ({res['percentage']})")
        st.markdown("---")

# show table summary
df = pd.DataFrame([{k: v for k, v in r.items() if k in ["filename", "pred_label", "percentage"]} for r in results])
if not df.empty:
    st.subheader("Summary table")
    st.dataframe(df, height=200)

    # allow CSV download
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

st.success("Done ✅")
