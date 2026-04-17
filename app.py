import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
import smtplib
from email.mime.text import MIMEText
import csv
from datetime import datetime
import os
import gdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Soil Classifier", layout="centered")

soil_labels = ['alluvial', 'black', 'clay', 'red', 'yellow']
model_accuracy = "96.5%"

# -------------------------
# DOWNLOAD MODEL
# -------------------------
MODEL_PATH = "best_soil_model.pth"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/uc?id=10x3uB72g0nzx2NEDtO9nobC1RDAlzbWU",
        MODEL_PATH,
        quiet=False
    )

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(soil_labels))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# WEATHER
# -------------------------
def get_weather(city):
    api_key = "e600cfd7f0d948f281183753262402"
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    try:
        data = requests.get(url).json()
        if "error" in data:
            return None, None
        return data["current"]["humidity"], data["current"]["temp_c"]
    except:
        return None, None

# -------------------------
# CIVIL ANALYSIS
# -------------------------
def civil_analysis(soil):
    return {
        "clay": {"Bearing": "75–150 kN/m²", "Settlement": "High", "Drainage": "Poor", "Suitability": "Not ideal"},
        "black": {"Bearing": "50–100 kN/m²", "Settlement": "Very high", "Drainage": "Very poor", "Suitability": "Risky"},
        "alluvial": {"Bearing": "100–200 kN/m²", "Settlement": "Moderate", "Drainage": "Good", "Suitability": "Good"},
        "red": {"Bearing": "150–300 kN/m²", "Settlement": "Low", "Drainage": "Excellent", "Suitability": "Highly suitable"},
        "yellow": {"Bearing": "120–250 kN/m²", "Settlement": "Moderate", "Drainage": "Moderate", "Suitability": "Suitable"}
    }[soil]

# -------------------------
# RISK + THEORY
# -------------------------
def risk_alert(analysis, humidity):
    score = 0
    reasons = []
    theory = []

    if "very high" in analysis["Settlement"].lower():
        score += 3
        reasons.append("Very high settlement")
        theory.append("Leads to structural instability")
    elif "high" in analysis["Settlement"].lower():
        score += 2
        reasons.append("High settlement")
        theory.append("May cause cracks")

    if "poor" in analysis["Drainage"].lower():
        score += 2
        reasons.append("Poor drainage")
        theory.append("Reduces soil strength")

    if humidity > 80:
        score += 2
        reasons.append("High moisture")
        theory.append("Decreases bearing capacity")

    if score >= 6:
        return "HIGH RISK", reasons, theory, "Avoid construction"
    elif score >= 3:
        return "MODERATE RISK", reasons, theory, "Use precautions"
    else:
        return "LOW RISK", reasons, theory, "Safe"

# -------------------------
# PDF GENERATION
# -------------------------
def generate_pdf(data):
    file_path = "soil_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    for key, value in data.items():
        elements.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return file_path

# -------------------------
# UI
# -------------------------
st.title("🌱 Soil Classification System")

st.sidebar.title("Model Info")
st.sidebar.write(f"Accuracy: {model_accuracy}")

st.sidebar.subheader("Team Members")
st.sidebar.write("Aagam Jain, Harini, Dharshan, Deva, Parzaan")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg","png"])
city = st.text_input("Enter City")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    soil = soil_labels[predicted.item()]
    confidence = confidence.item()*100

    st.subheader(f"Soil: {soil}")
    st.write(f"Confidence: {confidence:.2f}%")

    if city:
        humidity, temp = get_weather(city)

        if humidity:
            analysis = civil_analysis(soil)
            risk, reasons, theory, recommendation = risk_alert(analysis, humidity)

            st.write(f"Temp: {temp}°C | Humidity: {humidity}%")

            st.subheader("Civil Analysis")
            for k,v in analysis.items():
                st.write(f"{k}: {v}")

            st.subheader("Risk")
            st.write(risk)
            st.write(reasons)
            st.write(theory)
            st.write("Recommendation:", recommendation)

            # PDF data
            report_data = {
                "Soil": soil,
                "Confidence": f"{confidence:.2f}%",
                "Humidity": humidity,
                "Risk": risk,
                "Recommendation": recommendation
            }

            if st.button("Generate PDF"):
                pdf = generate_pdf(report_data)
                with open(pdf, "rb") as f:
                    st.download_button("Download Report", f, file_name="soil_report.pdf")
