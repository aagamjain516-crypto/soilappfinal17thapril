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
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(soil_labels))

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

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
# GRAIN SIZE
# -------------------------
def grain_size_estimate(soil_type):
    mapping = {
        "red": "Coarse to medium particle size with good drainage.",
        "alluvial": "Mixed particle size including sand and silt.",
        "clay": "Very fine particle size with low permeability.",
        "black": "Fine particle size with high moisture retention.",
        "yellow": "Moderate particle size with balanced properties."
    }
    return mapping.get(soil_type, "Unknown")

# -------------------------
# CIVIL ANALYSIS
# -------------------------
def civil_analysis(soil, humidity):
    data = {
        "clay": {
            "Bearing Capacity": "75–150 kN/m²",
            "Settlement": "High settlement",
            "Foundation": "Pile/raft foundation",
            "Drainage": "Poor drainage",
            "Suitability": "Not ideal for heavy structures",
            "Precautions": "Stabilization required"
        },
        "black": {
            "Bearing Capacity": "50–100 kN/m²",
            "Settlement": "Very high shrink-swell",
            "Foundation": "Deep foundation",
            "Drainage": "Very poor drainage",
            "Suitability": "Risky soil",
            "Precautions": "Moisture control needed"
        },
        "alluvial": {
            "Bearing Capacity": "100–200 kN/m²",
            "Settlement": "Moderate settlement",
            "Foundation": "Raft footing",
            "Drainage": "Good drainage",
            "Suitability": "Good for construction",
            "Precautions": "Check compaction"
        },
        "red": {
            "Bearing Capacity": "150–300 kN/m²",
            "Settlement": "Low settlement",
            "Foundation": "Shallow foundation",
            "Drainage": "Excellent drainage",
            "Suitability": "Highly suitable",
            "Precautions": "Minimal treatment"
        },
        "yellow": {
            "Bearing Capacity": "120–250 kN/m²",
            "Settlement": "Low to moderate",
            "Foundation": "Shallow foundation",
            "Drainage": "Moderate drainage",
            "Suitability": "Suitable",
            "Precautions": "Compaction needed"
        }
    }
    return data.get(soil)

# -------------------------
# QUALITY
# -------------------------
def soil_quality_grade(soil_type, humidity):
    grade = {
        "red": "Grade A",
        "alluvial": "Grade B",
        "clay": "Grade C",
        "black": "Grade C",
        "yellow": "Grade B"
    }[soil_type]

    if humidity > 80:
        grade += " (High moisture)"
    return grade

# -------------------------
# ADVANCED RISK
# -------------------------
def risk_alert(analysis, humidity):
    settlement = analysis["Settlement"].lower()
    drainage = analysis["Drainage"].lower()
    suitability = analysis["Suitability"].lower()

    score = 0
    reasons = []

    if "very high" in settlement:
        score += 3
        reasons.append("Very high settlement")
    elif "high" in settlement:
        score += 2
        reasons.append("High settlement")
    elif "moderate" in settlement:
        score += 1
        reasons.append("Moderate settlement")

    if "poor" in drainage:
        score += 2
        reasons.append("Poor drainage")

    if "not ideal" in suitability or "risky" in suitability:
        score += 2
        reasons.append("Low suitability")

    if humidity > 80:
        score += 2
        reasons.append("High moisture")

    if score >= 6:
        return "HIGH RISK", reasons
    elif score >= 3:
        return "MODERATE RISK", reasons
    else:
        return "LOW RISK", reasons

# -------------------------
# EMAIL
# -------------------------
def send_email_report(soil_type, humidity, quality, grain_size, risk):
    sender = "aagamjain816@gmail.com"
    receiver = "aagamjain516@gmail.com"
    password = "eqtbmlqavgcpzfrv"

    body = f"""
Soil: {soil_type}
Humidity: {humidity}
Quality: {quality}
Grain: {grain_size}
Risk: {risk}
"""

    msg = MIMEText(body)
    msg["Subject"] = "Soil Report"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        st.success("Email sent")
    except:
        st.error("Email failed")

# -------------------------
# LOGGING
# -------------------------
def log_data(soil_type, humidity, risk):
    with open("iot_soil_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), soil_type, humidity, risk])

# -------------------------
# UI
# -------------------------
st.title("🌱 Deep Learning ResNet50 Soil Classification")

st.sidebar.title("Model Info")
st.sidebar.write(f"Accuracy: {model_accuracy}")

st.sidebar.subheader("👥 Team Members")
st.sidebar.write("• Aagam Jain - 25BCE0220")
st.sidebar.write("• Harini R V - 25BCV0045")
st.sidebar.write("• Dharshan Boopalan - 25BEC0447")
st.sidebar.write("• Deva Harsha - 25BCE2011")
st.sidebar.write("• Parzaan - 25BCE2309")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg","png","jpeg"])
city = st.text_input("Enter City")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

    soil_type = soil_labels[predicted.item()]
    confidence = confidence.item()*100

    st.subheader(f"Soil Type: {soil_type.upper()}")
    st.write(f"Confidence: {confidence:.2f}%")

    if city:
        humidity, temp = get_weather(city)

        if humidity:
            analysis = civil_analysis(soil_type, humidity)
            quality = soil_quality_grade(soil_type, humidity)
            risk, reasons = risk_alert(analysis, humidity)
            grain = grain_size_estimate(soil_type)

            st.write(f"Temperature: {temp}°C | Humidity: {humidity}%")

            st.subheader("🏗️ Civil Analysis")
            for k,v in analysis.items():
                st.write(f"**{k}:** {v}")

            st.subheader("⚠️ Risk Assessment")
            if risk=="HIGH RISK": st.error(risk)
            elif risk=="MODERATE RISK": st.warning(risk)
            else: st.success(risk)

            for r in reasons:
                st.write(f"- {r}")

            st.write("Quality:", quality)
            st.write("Grain Size:", grain)

            if st.button("Send Email"):
                send_email_report(soil_type, humidity, quality, grain, risk)

            log_data(soil_type, humidity, risk)

        else:
            st.error("City not found")
