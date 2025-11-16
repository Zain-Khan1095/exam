import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢", layout="centered")
st.markdown("<h2 style='text-align:center;'>MNIST Digit Predictor</h2>", unsafe_allow_html=True)
st.write("Upload a 28x28 grayscale digit image (or any square image with a single digit) to predict.")

# Load model (automatically cached)
@st.cache_resource
def load_cnn_model():
    return load_model('mnist_cnn_model.h5')

model = load_cnn_model()

uploaded_file = st.file_uploader("Choose an image of a digit", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    img = ImageOps.invert(img)  # Invert ONLY if your model was trained on white digits/black background
    img = img.resize((28, 28), Image.LANCZOS)
    
    st.image(img, caption="Preprocessed Image for Model", width=150)
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    
    try:
        pred = model.predict(arr)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))
        st.success(f"Predicted digit: {pred_class} (Confidence: {confidence:.2%})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

