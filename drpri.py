import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
from PIL import Image
import os

if not os.path.exists("DR_model.keras"):
    import gdown
    gdown.download( id= "1-KWRtzskATpsRA0aGhU-cW9vUOYZPQ1_", output="DR_model.keras", quiet=False)
# ----------------------
# Load the trained model
# ----------------------

model = keras.models.load_model("DR_model.keras")

# ----------------------
# Streamlit App
# ----------------------
st.title("Diabetic Retinopathy Detection")

st.write("Upload a retina image to predict the DR stage.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Open image with PIL
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Read image for OpenCV
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Check if image is loaded correctly
    if img is not None:
        img = cv2.resize(img, (224, 224))
        # Prediction
        prd = np.argmax(model.predict(img.reshape(1, 224, 224, 3)), axis=1)[0]
        # Class names
        classes = ["Mild", "Moderate", "Severe", "Proliferate", "No"]
        st.success(classes[prd])
