import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model (make sure the path is correct)
model = load_model('ED2.keras')

# Emotion labels
emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_emotion(img):
    # Preprocess the image for the model
    img = img.convert('RGB')  # Ensure it's in RGB mode
    img = img.resize((48, 48))  # Resize to match the model's expected input size
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])  # Get the index of the highest probability

    return emotion_labels[emotion_index]

# Streamlit UI
st.title("Face Emotion Detection")

# Image upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict emotion
    emotion = predict_emotion(image)
    
    # Display prediction
    st.write(f"Predicted Emotion: **{emotion}**")
