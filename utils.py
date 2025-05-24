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
