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
from risk_engine import construction_risk
import pandas as pd

#(PDF)
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
            return None, None, None
        return (
            data["current"]["humidity"],
            data["current"]["temp_c"],
            data["current"]["precip_mm"]
        )
    except:
        return None, None, None

# -------------------------
# GRAIN SIZE
# -------------------------
def grain_size_estimate(soil_type):
    if soil_type == "red":
        return "Coarse to medium particle size with good drainage."
    elif soil_type == "alluvial":
        return "Mixed particle size including sand and silt."
    elif soil_type == "clay":
        return "Very fine particle size with low permeability."
    elif soil_type == "black":
        return "Fine particle size with high moisture retention."
    else:
        return "Unknown grain characteristics."

# -------------------------
# EMAIL
# -------------------------
def send_email_report(soil_type, humidity, quality, grain_size, risk):
    sender = "aagamjain816@gmail.com"
    receiver = "aagamjain516@gmail.com"
    password = "eqtbmlqavgcpzfrv"

    body = f"""
Soil Type: {soil_type}
Humidity: {humidity}%

Quality: {quality}
Grain Size: {grain_size}
Risk: {risk}
"""

    msg = MIMEText(body)
    msg["Subject"] = "Soil Monitoring Report"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        st.success("Email sent successfully")
    except:
        st.error("Email sending failed")

# -------------------------
# CIVIL ANALYSIS
# -------------------------
def civil_analysis(soil, humidity):
    if soil == "clay":
        return {
            "Bearing Capacity": "75–150 kN/m²",
            "Settlement": "High settlement due to compressibility",
            "Foundation": "Pile or raft foundation recommended",
            "Drainage": "Poor drainage, retains water",
            "Suitability": "Not ideal for heavy structures",
            "Precautions": "Soil stabilization required"
        }
    elif soil == "black":
        return {
            "Bearing Capacity": "50–100 kN/m²",
            "Settlement": "Very high shrink-swell behavior",
            "Foundation": "Deep foundation",
            "Drainage": "Very poor drainage",
            "Suitability": "Risky soil",
            "Precautions": "Moisture control needed"
        }
    elif soil == "alluvial":
        return {
            "Bearing Capacity": "100–200 kN/m²",
            "Settlement": "Moderate settlement",
            "Foundation": "Raft footing",
            "Drainage": "Good drainage",
            "Suitability": "Good for construction",
            "Precautions": "Check compaction"
        }
    elif soil == "red":
        return {
            "Bearing Capacity": "150–300 kN/m²",
            "Settlement": "Low settlement",
            "Foundation": "Shallow foundation",
            "Drainage": "Excellent drainage",
            "Suitability": "Highly suitable",
            "Precautions": "Minimal treatment"
        }
    elif soil == "yellow":
        return {
            "Bearing Capacity": "120–250 kN/m²",
            "Settlement": "Low to moderate",
            "Foundation": "Shallow foundation",
            "Drainage": "Moderate drainage",
            "Suitability": "Suitable",
            "Precautions": "Compaction needed"
        }

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
# RISK
# -------------------------
def risk_alert(analysis, humidity):
    if "very high" in analysis["Settlement"].lower():
        return "HIGH RISK"
    elif humidity > 80:
        return "MODERATE RISK"
    return "LOW RISK"

# -------------------------
# PDF FUNCTION
# -------------------------
def generate_pdf(report_data):
    file_path = "soil_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    for key, value in report_data.items():
        elements.append(Paragraph(f"<b>{key}:</b> {value}", styles["Normal"]))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return file_path

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
st.title("🌱 Deep learning ResNet50 - Based Soil Classification")

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("Model Info")
st.sidebar.write(f"Accuracy: {model_accuracy}")

st.sidebar.subheader("Team Members")
st.sidebar.write("• Aagam Jain - 25BCE0220")
st.sidebar.write("• Harini R V - 25BCV0045")
st.sidebar.write("• Dharshan Boopalan - 25BEC0447")
st.sidebar.write("• Deva Harsha - 25BCE2011")
st.sidebar.write("• Parzaan - 25BCE2309")

uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "png", "jpeg"])
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
    confidence = confidence.item() * 100

    st.subheader(f"Soil Type: {soil_type.upper()}")
    st.write(f"Confidence: {confidence:.2f}%")

    if city:
        humidity, temp, rainfall = get_weather(city)

        if humidity:
            analysis = civil_analysis(soil_type, humidity)
            quality = soil_quality_grade(soil_type, humidity)
            risk = risk_alert(analysis, humidity)
            grain = grain_size_estimate(soil_type)

            construction_risks = construction_risk(
                soil_type,
                moisture=humidity,
                temp=temp,
                humidity=humidity,
                rainfall=rainfall if rainfall else 0
            )

            st.write(f"Temperature: {temp} °C | Humidity: {humidity}%")

            st.subheader("🏗️ Civil Engineering Analysis")
            for key, value in analysis.items():
                st.write(f"**{key}:** {value}")

            st.write("Quality:", quality)
            st.write("Risk:", risk)
            st.write("Grain Size:", grain)

            # FUTURE
            st.subheader("📅 Future Soil Risk Prediction")
            for days in [30, 60, 90]:
                future_risks = construction_risk(
                    soil_type,
                    moisture=humidity - (days//30)*5,
                    temp=temp + (days//30)*2,
                    humidity=humidity,
                    rainfall=rainfall + (days//30)*20 if rainfall else 0
                )
                score_map = {"Low": 1, "Moderate": 2, "High": 3}
                future_score = sum(score_map[level] for level, _ in future_risks.values())
                st.write(f"After {days} days → Risk Score: {future_score}")

            # CURRENT RISK
            st.subheader("🏗️ AI-Based Construction Risk Prediction")
            for key, (level, msg) in construction_risks.items():
                if level == "High":
                    st.error(f"{key}: {level} ⚠️\n{msg}")
                elif level == "Moderate":
                    st.warning(f"{key}: {level}\n{msg}")
                else:
                    st.success(f"{key}: {level}\n{msg}")

            score_map = {"Low": 1, "Moderate": 2, "High": 3}
            overall_score = sum(score_map[level] for level, _ in construction_risks.values())

            st.metric("🏗️ Overall Construction Risk Score", overall_score)

            # RECOMMENDATION
            st.subheader("🤖 Recommendation")
            if overall_score >= 8:
                st.error("Avoid construction or use deep foundation")
            elif overall_score >= 5:
                st.warning("Use soil stabilization techniques")
            else:
                st.success("Safe for construction")

            # GRAPH
            st.session_state.history.append({
                "Moisture": humidity,
                "Temperature": temp,
                "Risk": overall_score
            })

            if len(st.session_state.history) > 20:
                st.session_state.history.pop(0)

            st.subheader("📊 Soil & Risk Trends")
            df = pd.DataFrame(st.session_state.history)
            st.line_chart(df)

            # PDF
            report_data = {
                "Soil Type": soil_type,
                "Confidence": f"{confidence:.2f}%",
                "Temperature": temp,
                "Humidity": humidity,
                "Quality": quality,
                "Risk": risk,
                "Grain Size": grain,
                "Construction Risk Score": overall_score
            }

            if st.button("Generate PDF Report"):
                pdf_file = generate_pdf(report_data)
                with open(pdf_file, "rb") as f:
                    st.download_button("Download Report", f, file_name="soil_report.pdf")

            if st.button("Send Email Report"):
                send_email_report(soil_type, humidity, quality, grain, risk)

            log_data(soil_type, humidity, risk)

        else:
            st.error("City not found")
