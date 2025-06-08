from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from typing import Dict
import logging
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="Drawing Emotion Analysis Service")

logging.basicConfig(level=logging.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajustar en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

EMOTION_LABELS = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]

def calculate_stroke_thickness(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    dist_transform = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)
    thickness = np.mean(dist_transform[dist_transform > 0])
    logging.debug(f"Stroke thickness calculated: {thickness}")
    return float(thickness)

def calculate_colorfulness(image: np.ndarray) -> float:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    colorfulness = np.std(a) + np.std(b)
    logging.debug(f"Colorfulness calculated: {colorfulness}")
    return float(colorfulness)

def calculate_shape_complexity(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    complexity = sum(cv2.arcLength(cnt, True) for cnt in contours)
    logging.debug(f"Shape complexity calculated: {complexity}")
    return float(complexity)

def estimate_emotions(features: Dict[str, float]) -> Dict[str, float]:
    stroke = features.get("stroke_thickness", 0)
    color = features.get("colorfulness", 0)
    complexity = features.get("shape_complexity", 0)

    emotions = {
        "enojo": min(max(stroke / 5.0, 0), 1),
        "alegría": min(max(color / 50.0, 0), 1),
        "tristeza": min(max((5.0 - complexity / 100.0) / 5.0, 0), 1),
        "neutral": 0.2,
        "asco": 0.0,
        "miedo": 0.0,
        "sorpresa": 0.0,
    }

    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}

    logging.debug(f"Estimated emotions: {emotions}")
    return emotions

def dominant_emotion(emotions: Dict[str, float]) -> str:
    dom_emotion = max(emotions, key=emotions.get)
    logging.debug(f"Dominant emotion: {dom_emotion}")
    return dom_emotion

@app.get("/")
async def root():
    return {"message": "Drawing Emotion Analysis Service alive"}

@app.get("/health")
async def health_check():
    return {"model_loaded": True}

@app.post("/analyze-drawing")
async def analyze_drawing(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        logging.error(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG and PNG allowed.")

    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logging.error("Failed to decode image from bytes")
            raise HTTPException(status_code=400, detail="Could not decode image.")

        stroke_thickness = calculate_stroke_thickness(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        colorfulness = calculate_colorfulness(img)
        shape_complexity = calculate_shape_complexity(img)

        features = {
            "stroke_thickness": stroke_thickness,
            "colorfulness": colorfulness,
            "shape_complexity": shape_complexity,
        }

        emotions = estimate_emotions(features)
        dom_emotion = dominant_emotion(emotions)
        recommendation = await generate_ai_recommendation(dom_emotion, emotions)

        return {
            "success": True,
            "data": {
                "features": features,
                "emotions": emotions,
                "dominant_emotion": dom_emotion,
                "recommendation": recommendation
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Unexpected error during analyze-drawing")
        raise HTTPException(status_code=500, detail="Internal error. Please try again later.")

# 💡 Recomendación por IA vía Hugging Face
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HUGGINGFACE_MODEL = "tiiuae/falcon-7b-instruct"  # Puedes cambiarlo por otro compatible

async def generate_ai_recommendation(dominant_emotion: str, emotions: Dict[str, float]) -> str:
    secondary = sorted(
        [(emo, val) for emo, val in emotions.items() if emo != dominant_emotion],
        key=lambda x: x[1],
        reverse=True
    )

    secondary_emotion = secondary[0][0] if secondary and secondary[0][1] > 0.2 else "ninguna emoción secundaria"

    prompt = (
        f"La emoción principal detectada en el dibujo es '{dominant_emotion}'. "
        f"También se detectaron señales de: {secondary_emotion}. "
        "¿Qué consejo emocional puedes dar basado en esto?"
    )

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"
    }

    payload = {"inputs": prompt}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL}",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                result = response.json()
                return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
            else:
                return "No se pudo generar una recomendación emocional en este momento."
    except Exception as e:
        logging.error(f"Error al obtener recomendación de IA: {str(e)}")
        return "No se pudo generar una recomendación emocional en este momento."
