import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import io
from PIL import Image

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize the image classification model
drawing_analysis_model = pipeline("image-classification", model="google/vit-base-patch16-224-in21k")

# Model for analysis response
class EmotionAnalysisResponse(BaseModel):
    emotions: dict
    dominant_emotion: str
    emotional_advice: str

# Health check endpoint
@app.get("/health")
async def health():
    return {"model_loaded": True}

# Function to generate emotional advice based on the dominant emotion
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

# Endpoint to analyze images (drawings)
@app.post("/analyze-drawing", response_model=EmotionAnalysisResponse)
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        # Read the image from the uploaded file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Log image size and format
        logging.debug(f"Image size: {image.size}, Image format: {image.format}")

        # Analyze the image using the model
        analysis = drawing_analysis_model(image)

        # Log analysis result
        logging.debug(f"Analysis result: {analysis}")

        emotions = analysis[0]["label"]
        dominant_emotion = emotions[0]
        emotional_advice = generate_advice(dominant_emotion)

        # Return the response
        return EmotionAnalysisResponse(
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            emotional_advice=emotional_advice
        )
    
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return {"error": str(e)}
