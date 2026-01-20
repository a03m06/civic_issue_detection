from fastapi import FastAPI, Form, HTTPException
import joblib
import numpy as np
import cv2
import traceback
import requests
from tensorflow.keras.models import load_model

# FASTAPI APP 
app = FastAPI()

#  LOAD MODELS 
try:
    text_model = joblib.load("text_issue_classifier.joblib")
except Exception as e:
    raise RuntimeError(f"Failed to load text model: {e}")

try:
    image_model = load_model("civic_issue_model.h5")
except Exception as e:
    raise RuntimeError(f"Failed to load image model: {e}")


IMAGE_CLASSES = ["garbage", "pothole", "streetlight", "water_leakage"]

# PRIORITY LOGIC
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

# SEVERITY BOOST 
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


#  API ENDPOINT 
@app.post("/predict")
async def predict(
    image_url: str = Form(...),
    text: str = Form(...),
    location: str = Form(...)
):
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        #  IMAGE LOADING FROM URL 
        try:
            resp = requests.get(image_url, timeout=10)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Image download failed")

            img_arr = np.frombuffer(resp.content, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image")

        except requests.exceptions.RequestException:
            raise HTTPException(status_code=400, detail="Unable to fetch image from URL")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        image_probs = image_model.predict(img)[0]
        image_confidence = float(np.max(image_probs))
        image_issue = IMAGE_CLASSES[np.argmax(image_probs)]

        # TEXT PROCESSING 
        text_probs = text_model.predict_proba([text])[0]
        text_confidence = float(np.max(text_probs))
        text_issue = text_model.classes_[np.argmax(text_probs)]

        #  LOW CONFIDENCE SAFETY 
        if image_confidence < 0.4 and text_confidence < 0.4:
            return {
                "issue": "uncertain",
                "priority": "None",
                "priority_score": 0,
                "solution": "Low confidence prediction. Please provide clearer input."
            }

        #  FUSION (TEXT DOMINANT)
        TEXT_THRESHOLD = 0.3
        IMAGE_THRESHOLD = 0.8

        # Case 1: Text confident : use text
        if text_confidence >= TEXT_THRESHOLD:
            final_issue = text_issue
            decision_source = "text"
            final_confidence = text_confidence

        # Case 2: Text weak, image very confident : use image
        elif text_confidence < TEXT_THRESHOLD and image_confidence >= IMAGE_THRESHOLD:
            final_issue = image_issue
            decision_source = "image"
            final_confidence = image_confidence

        # Case 3: Both weak
        else:
            return {
                "issue": "uncertain",
                "priority": "None",
                "priority_score": 0,
                "solution": "Low confidence in both text and image."
            }

        #  PRIORITY DECISION 
        boost = severity_boost(text, final_issue)
        score = BASE_SCORE[final_issue] + boost
        score = min(score, 100)

        priority = decide_priority(final_issue, score)
        solution = SOLUTIONS[final_issue][priority]

        return {
            "issue": final_issue,
            "priority": priority,
            "priority_score": score,
            "solution": solution,
            "final_confidence": round(final_confidence, 2),
            "image_confidence": round(image_confidence, 2),
            "text_confidence": round(text_confidence, 2),
            "decision_source": decision_source,
            "location": location
        }

    except HTTPException as he:
        raise he

    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal ML processing error"
        )


# RUN SERVER
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
