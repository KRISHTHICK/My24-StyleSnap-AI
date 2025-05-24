import streamlit as st
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
from utils import extract_colors, get_fashion_suggestions, generate_palette_image, generate_pdf

st.set_page_config(page_title="StyleSnap AI", layout="centered")
st.title("🎨 StyleSnap AI - Outfit Color Analyzer & Matcher")

uploaded_image = st.file_uploader("Upload your outfit photo", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    colors = extract_colors(image)
    st.subheader("🎯 Dominant Colors")
    st.image(generate_palette_image(colors), caption="Color Palette", use_column_width=True)

    suggestions = get_fashion_suggestions(colors)
    st.subheader("🛍️ Suggested Matches:")
    for idx, suggestion in enumerate(suggestions):
        st.markdown(f"**{idx+1}.** {suggestion}")

    if st.button("📥 Download PDF Lookbook"):
        generate_pdf(colors, suggestions)
        with open("lookbook.pdf", "rb") as f:
            st.download_button("Download Lookbook", f, file_name="lookbook.pdf")

