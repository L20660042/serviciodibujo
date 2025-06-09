import logging
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import io
from PIL import Image

# Inicialización de FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Inicialización del pipeline para análisis de emociones
emotion_recognition_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Modelos para el análisis de dibujo
drawing_analysis_model = pipeline("image-classification", model="your-drawing-model")

# Define el modelo para respuestas del análisis
class EmotionAnalysisResponse(BaseModel):
    emotions: dict
    dominant_emotion: str
    emotional_advice: str

# Endpoint para verificar que el servicio está activo
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Función para generar el consejo emocional
def generate_advice(dominant_emotion: str) -> str:
    advice = ""
    if dominant_emotion == "enojo":
        advice = "Mediante los trazos y el grosor se detecta que estás enojado, deberías salir a distraerte y relajarte."
    elif dominant_emotion == "alegría":
        advice = "Parece que estás muy feliz. ¡Disfruta el momento y comparte esa felicidad con otros!"
    elif dominant_emotion == "tristeza":
        advice = "Se percibe tristeza en tus trazos. Podrías intentar hablar con un amigo o hacer algo que te guste."
    elif dominant_emotion == "miedo":
        advice = "Los trazos indican que te sientes temeroso. Trata de hablar sobre lo que te preocupa y enfrenta tus miedos."
    elif dominant_emotion == "asco":
        advice = "Se nota que algo te causa desagrado. Podrías intentar relajarte y reflexionar sobre lo que te hace sentir incómodo."
    elif dominant_emotion == "neutral":
        advice = "Tu dibujo no muestra emociones fuertes. Tal vez estés en un estado equilibrado, pero no dudes en descansar."
    elif dominant_emotion == "sorpresa":
        advice = "Parece que algo te sorprendió. ¡Tómate un tiempo para procesar lo que ocurrió!"
    return advice

# Endpoint para analizar imágenes (dibujos)
@app.post("/analyze-drawing", response_model=EmotionAnalysisResponse)
async def analyze_drawing(file: UploadFile = File(...)):
    # Leer la imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Convertir la imagen a un formato adecuado para el análisis
    analysis = drawing_analysis_model(image)
    
    # Aquí se obtienen las emociones a partir del análisis
    emotions = analysis['labels']  # Dependiendo de la estructura de la salida
    dominant_emotion = emotions[0]  # Emoción dominante
    emotional_advice = generate_advice(dominant_emotion)  # Generar consejo emocional
    
    # Formatear la respuesta
    return EmotionAnalysisResponse(emotions=emotions, dominant_emotion=dominant_emotion, emotional_advice=emotional_advice)
