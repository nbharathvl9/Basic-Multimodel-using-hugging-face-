import streamlit as st
from transformers import pipeline
from PIL import Image

# Load NLP and image classification models
sentiment_classifier = pipeline("sentiment-analysis", framework="tf")  # Using TensorFlow
image_classifier = pipeline("image-classification", model="microsoft/resnet-50")

# Streamlit UI
st.title("AI-powered Sentiment & Image Classification")

# Sentiment Analysis Section
st.header("Sentiment Analysis")
user_input = st.text_area("Enter text for sentiment analysis:")

if st.button("Analyze Sentiment"):
    if user_input:
        result = sentiment_classifier(user_input)
        st.write("Sentiment:", result[0]['label'])
        st.write("Confidence:", round(result[0]['score'], 4))

# Image Classification Section
st.header("Image Classification")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify Image"):
        results = image_classifier(image)
        st.write("Top Predictions:")
        for res in results:
            st.write(f"**{res['label']}** - Confidence: {round(res['score'], 4)}")

