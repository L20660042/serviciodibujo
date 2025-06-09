from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification
import torch

app = FastAPI()

# Habilitar CORS para permitir solicitudes desde otros dominios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las URLs, puedes cambiar "*" por un dominio específico si prefieres más restricción
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todas las cabeceras
)

# Cargar el modelo ViT (Vision Transformer) y su extractor de características
model_name = "google/vit-base-patch16-224-in21k"
extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Cargar el modelo de lenguaje de Hugging Face para generar recomendaciones
language_model = pipeline('text-generation', model="gpt2")

# Limitar el tamaño máximo del archivo (3 MB)
MAX_IMAGE_SIZE = 3 * 1024 * 1024  # 3 MB

@app.get("/")
async def root():
    return {"message": "Drawing Emotion Analysis Service alive"}

@app.get("/health")
async def health_check():
    return {"model_loaded": True}

@app.post("/analyze-drawing")
async def analyze_drawing(file: UploadFile = File(...)):
    try:
        # Leer el archivo de imagen
        file_content = await file.read()
        file_size = len(file_content)

        # Verificar el tamaño del archivo
        if file_size > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail="El archivo es demasiado grande. El tamaño máximo es 3MB.")

        # Convertir la imagen a formato compatible con el modelo
        image = Image.open(io.BytesIO(file_content))

        # Preprocesar la imagen
        inputs = extractor(images=image, return_tensors="pt", padding=True)

        # Realizar la inferencia con ViT (asegurándonos de no hacer modificaciones inplace)
        with torch.no_grad():
            outputs = model(**inputs)

        # Clonamos los logits antes de cualquier operación para evitar modificaciones inplace
        logits = outputs.logits.detach().clone()  # Usamos detach() antes de clonar para evitar la conexión al grafo de cálculo

        # Usamos .max() en lugar de argmax para evitar el error de inplace update
        predicted_class_idx = logits.max(dim=-1)[1].item()

        emotions = {model.config.id2label[predicted_class_idx]: logits[0][predicted_class_idx].item()}

        # Aquí puedes obtener la emoción dominante y otras recomendaciones
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
        # Manejo de excepciones con un mensaje más detallado
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")
