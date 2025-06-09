from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from transformers import pipeline

app = FastAPI()

# Cargar el modelo de Hugging Face para análisis de emociones en dibujos
emotion_model = pipeline('image-classification', model="huggingface/emotion-recognition-model")

# Cargar el modelo de lenguaje de Hugging Face para generar recomendaciones
language_model = pipeline('text-generation', model="gpt2")

@app.get("/")
async def root():
    return {"message": "Drawing Emotion Analysis Service alive"}

@app.get("/health")
async def health_check():
    return {"model_loaded": True}

@app.post("/analyze-drawing")
async def analyze_drawing(file: UploadFile = File(...)):
    # Convertir el archivo de imagen
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Procesar la imagen para detectar emociones
    result = emotion_model(image)

    # Extraer emociones detectadas
    emotions = {emotion['label']: emotion['score'] for emotion in result}
    dominant_emotion = max(emotions, key=emotions.get)

    # Generar una recomendación basada en la emoción dominante usando GPT-2
    recommendation_input = f"Recomendación para alguien que se siente {dominant_emotion}:"
    recommendation = language_model(recommendation_input, max_length=50, num_return_sequences=1)[0]['generated_text']

    # Crear el consejo emocional basado en la emoción detectada
    emotional_advice = f"Basado en tus trazos, parece que estás {dominant_emotion}. {recommendation}"

    return {
        "data": {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "emotional_advice": emotional_advice,
        }
    }
