import streamlit as st
import joblib
import numpy as np
import cv2
import requests
from tensorflow.keras.models import load_model

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Civic Issue Detection",
    layout="centered"
)

st.title("🛠️ Civic Issue Detection & Prioritization")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    text_model = joblib.load("text_issue_classifier.joblib")
    image_model = load_model("civic_issue_model.h5")
    return text_model, image_model

text_model, image_model = load_models()

IMAGE_CLASSES = ["garbage", "pothole", "streetlight", "water_leakage"]

# -----------------------------
# PRIORITY LOGIC (UNCHANGED)
# -----------------------------
BASE_SCORE = {
    "garbage": 10,
    "streetlight": 15,
    "pothole": 20,
    "water_leakage": 25
}

PRIORITY_RULES = {
    "garbage": {"Low": 20, "Medium": 50},
    "streetlight": {"Low": 25, "Medium": 55},
    "pothole": {"Low": 30, "Medium": 60},
    "water_leakage": {"Low": 35, "Medium": 65}
}

SOLUTIONS = {
    "garbage": {
        "Low": "Schedule routine waste collection.",
        "Medium": "Send sanitation team immediately.",
        "High": "Emergency sanitation drive and disinfection."
    },
    "streetlight": {
        "Low": "Schedule inspection.",
        "Medium": "Assign electrical maintenance team.",
        "High": "Immediate repair for public safety."
    },
    "pothole": {
        "Low": "Schedule road inspection.",
        "Medium": "Place warning signs and plan repair.",
        "High": "Dispatch road maintenance team immediately."
    },
    "water_leakage": {
        "Low": "Schedule pipeline inspection.",
        "Medium": "Dispatch maintenance team.",
        "High": "Immediate repair to stop water wastage."
    }
}

# -----------------------------
# SEVERITY BOOST (UNCHANGED)
# -----------------------------
def severity_boost(text, issue):
    text = text.lower()

    if issue == "water_leakage":
        if "flood" in text or "continuous" in text:
            return 40
        if "pipe" in text and "broken" in text:
            return 25

    if issue == "pothole":
        if "deep" in text or "big" in text:
            return 25
        if "accident" in text or "dangerous" in text:
            return 40

    if issue == "streetlight":
        if "dark" in text or "night" in text:
            return 25
        if "unsafe" in text:
            return 40

    if issue == "garbage":
        if "overflowing" in text:
            return 25
        if "smell" in text or "unbearable" in text:
            return 40

    return 0


def decide_priority(issue, score):
    rules = PRIORITY_RULES[issue]
    if score <= rules["Low"]:
        return "Low"
    elif score <= rules["Medium"]:
        return "Medium"
    else:
        return "High"


# -----------------------------
# UI INPUTS
# -----------------------------
st.subheader("Input Complaint")

image_url = st.text_input("Image URL")
text = st.text_area("Complaint Text")
location = st.text_input("Location")

predict_btn = st.button(" Analyze Complaint")

# -----------------------------
# PREDICTION
# -----------------------------
if predict_btn:
    if not text.strip():
        st.error("Complaint text is required.")
        st.stop()

    # IMAGE PROCESSING
    try:
        resp = requests.get(image_url, timeout=10)
        img_arr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        image_probs = image_model.predict(img)[0]
        image_confidence = float(np.max(image_probs))
        image_issue = IMAGE_CLASSES[np.argmax(image_probs)]

    except Exception:
        st.error("Failed to load image from URL.")
        st.stop()

    # TEXT PROCESSING
    text_probs = text_model.predict_proba([text])[0]
    text_confidence = float(np.max(text_probs))
    text_issue = text_model.classes_[np.argmax(text_probs)]

    # LOW CONFIDENCE SAFETY
    if image_confidence < 0.4 and text_confidence < 0.4:
        st.warning("Low confidence in both image and text.")
        st.write("Priority Score: 0")
        st.write("Solution: Please provide clearer input.")
        st.stop()

    # FUSION LOGIC (UNCHANGED)
    TEXT_THRESHOLD = 0.3
    IMAGE_THRESHOLD = 0.8

    if text_confidence >= TEXT_THRESHOLD:
        final_issue = text_issue
        decision_source = "text"
        final_confidence = text_confidence
    elif image_confidence >= IMAGE_THRESHOLD:
        final_issue = image_issue
        decision_source = "image"
        final_confidence = image_confidence
    else:
        st.warning("Unable to confidently determine issue.")
        st.stop()

    # PRIORITY SCORE
    boost = severity_boost(text, final_issue)
    score = min(BASE_SCORE[final_issue] + boost, 100)
    priority = decide_priority(final_issue, score)
    solution = SOLUTIONS[final_issue][priority]

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader(" Result")

    st.write(f"**issue:** {final_issue}")
    st.write(f"**priority:** {priority}")
    st.write(f"**priority_score:** {score}")
    st.write(f"**solution:** {solution}")
    st.write(f"**Decision Source:** {decision_source}")
    st.write(f"**Final Confidence:** {round(final_confidence, 2)}")
    st.write(f"**Image Confidence:** {round(image_confidence, 2)}")
    st.write(f"**Text Confidence:** {round(text_confidence, 2)}")
    st.write(f"**Location:** {location}")
