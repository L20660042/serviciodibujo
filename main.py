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

# Emotion labels for mapping
emotion_labels = ["enojo", "asco", "miedo", "alegría", "neutral", "tristeza", "sorpresa"]
emotion_icons = {
    "enojo": "😠",
    "asco": "🤢",
    "miedo": "😨",
    "alegría": "😄",
    "neutral": "😐",
    "tristeza": "😢",
    "sorpresa": "😮"
}

# Model for analysis response
class EmotionAnalysisResponse(BaseModel):
    emotions: dict  # Expecting a dictionary for emotions
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

        # Log image size and format for debugging
        logging.debug(f"Image size: {image.size}, Image format: {image.format}")

        # Analyze the image using the model
        analysis = drawing_analysis_model(image)

        # Log the model's output to inspect its structure
        logging.debug(f"Model analysis output: {analysis}")

        # Check if analysis is in the expected format
        if isinstance(analysis, list) and len(analysis) > 0:
            emotions = {}
            dominant_emotion = None
            highest_score = 0

            # Loop through all labels and capture emotions with their scores
            for result in analysis:
                label = result.get('label', '')
                score = result.get('score', 0)
                if label and score > 0.1:  # Only consider scores above 0.1
                    # Map label index to emotion
                    label_index = int(label.replace('LABEL_', ''))  # Convert 'LABEL_0' -> 0
                    emotion = emotion_labels[label_index]
                    emotions[emotion] = score
                    if score > highest_score:
                        dominant_emotion = emotion
                        highest_score = score

            # Sort emotions based on their score (highest to lowest)
            sorted_emotions = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))

            # Generate emotional advice for the dominant emotion
            emotional_advice = generate_advice(dominant_emotion)

            # Return the response
            return EmotionAnalysisResponse(
                emotions=sorted_emotions,
                dominant_emotion=dominant_emotion,
                emotional_advice=emotional_advice
            )

        else:
            logging.error("Unexpected analysis output format.")
            return {"error": "Unexpected analysis output format."}

    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return {"error": str(e)}
