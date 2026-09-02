import streamlit as st
from transformers import pipeline
from PIL import Image

st.title("AI Image Classifier")

@st.cache_resource
def load_classifier():
    return pipeline("image-classification")

classifier = load_classifier()

uploaded_file = st.file_uploader(
    "Upload the image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, use_container_width=True)

    if st.button("Classify"):
        result = classifier(image)
        st.write(result)

        st.success(f'The image is classified as: {result[0]["label"]}')
