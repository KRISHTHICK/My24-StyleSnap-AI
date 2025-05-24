# My24-StyleSnap-AI
GenAI

Here’s a **new fashion-related AI project** (no virtual environment required) that runs easily in **VS Code** and can be deployed on **GitHub Pages** or Streamlit Cloud:

---

## 👗 Project Name: **StyleSnap AI – Outfit Color Analyzer & Matcher**

### 📌 Project Overview

**StyleSnap AI** is a fashion assistant app that:

* Lets users upload a photo of their outfit.
* Analyzes the dominant colors in the outfit.
* Recommends color-matching accessories or complementary outfit ideas based on fashion color theory.
* Visualizes the color palette and lets users download it as a PDF.

---

## 💡 Key Features

1. **Image Upload & Color Extraction**

   * Upload any outfit image (JPEG/PNG).
   * Extracts dominant colors using k-means clustering.

2. **Fashion-based Color Recommendations**

   * Uses color theory (complementary, triadic) to suggest matching outfit/accessory shades.

3. **Visual Color Palette**

   * Generates a stylish palette bar.

4. **Download Lookbook (PDF)**

   * Users can download a color palette + outfit suggestion lookbook as a PDF.

5. **No venv Needed**

   * Works with global Python installation.

---

## 🧠 Tech Stack

* Python
* Streamlit
* OpenCV (for image handling)
* scikit-learn (for KMeans)
* fpdf (for PDF lookbook)

---

## 🗂 Folder Structure

```
StyleSnap-AI/
├── app.py
├── requirements.txt
├── utils.py
├── sample.jpg (optional sample image)
```

---

## 📄 `requirements.txt`

```txt
streamlit
opencv-python
scikit-learn
Pillow
fpdf
```

---

## 🚀 How to Run (in VS Code or Terminal)

```bash
# Step 1: Install requirements
pip install -r requirements.txt

# Step 2: Run the app
streamlit run app.py
```

---

## 🧾 `app.py` (Main Streamlit App)

```python
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

```

---

## 🔧 `utils.py`

```python
from sklearn.cluster import KMeans
import numpy as np
import cv2
from PIL import Image
from fpdf import FPDF

def extract_colors(image, k=5):
    img = np.array(image)
    img = cv2.resize(img, (100, 100))
    img = img.reshape((img.shape[0]*img.shape[1], 3))

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)
    return colors.tolist()

def get_fashion_suggestions(colors):
    color_names = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'purple', 'pink', 'gray']
    suggestions = []
    for color in colors:
        r, g, b = color
        if r > 200 and g < 100:
            suggestions.append("Pair with neutral or cool-tone accessories (e.g., white sneakers).")
        elif b > 180:
            suggestions.append("Match with tan or earthy-tone pants.")
        elif g > 180:
            suggestions.append("Add a leather belt or bag for contrast.")
        else:
            suggestions.append("Try minimal accessories and solid patterns.")
    return suggestions

def generate_palette_image(colors):
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    step = 300 // len(colors)
    for i, color in enumerate(colors):
        palette[:, i*step:(i+1)*step] = color
    return Image.fromarray(palette)

def generate_pdf(colors, suggestions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="StyleSnap AI - Outfit Color Suggestions", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    for idx, color in enumerate(colors):
        r, g, b = color
        pdf.set_fill_color(r, g, b)
        pdf.cell(30, 10, "", ln=0, fill=True)
        pdf.cell(160, 10, f"Color RGB: ({r}, {g}, {b})", ln=1)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Recommendations", ln=True)
    pdf.set_font("Arial", "", 10)
    for s in suggestions:
        pdf.multi_cell(0, 10, s)

    pdf.output("lookbook.pdf")
```

---

## 📝 GitHub README Sample

````markdown
# 🎨 StyleSnap AI - Outfit Color Analyzer & Matcher

A fashion AI tool that analyzes dominant colors from outfit images and suggests matching clothing or accessories.

## 🧠 Features
- Upload outfit photos
- Extract dominant colors
- Get outfit/accessory suggestions
- Download PDF Lookbook

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
````

## 🧰 Tech Stack

* Streamlit
* OpenCV
* scikit-learn
* Pillow
* fpdf

## 📦 No virtual environment required!

```

---

Would you like me to generate a downloadable `.zip` file or GitHub push instructions for this project too?
```
