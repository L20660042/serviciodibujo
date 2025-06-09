from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from transformers import pipeline

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde otros dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las URLs, puedes cambiar "*" por un dominio específico si prefieres más restricción
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todas las cabeceras
)

# Cargar los modelos de Hugging Face una vez para evitar cargar en cada solicitud
emotion_model = pipeline('image-classification', model="google/vit-base-patch16-224-in21k")
language_model = pipeline('text-generation', model="gpt2")

# Rutas
@app.get("/")
async def root():
    return {"message": "Drawing Emotion Analysis Service alive"}

@app.get("/health")
async def health_check():
    return {"model_loaded": True}

MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3 MB

@app.post("/analyze-drawing")
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo (esto también permite calcular el tamaño)
        file_content = await file.read()

        # Obtener el tamaño del archivo
        file_size = len(file_content)

        # Limitar el tamaño máximo del archivo
        MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3 MB
        if file_size > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="El archivo es demasiado grande. El tamaño máximo es 3MB.")
        
        # Convertir la imagen
        image = Image.open(io.BytesIO(file_content))

        # Procesar la imagen para detectar emociones usando el modelo ViT
        result = emotion_model(image)

        # Extraer las clases y sus puntuaciones
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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
