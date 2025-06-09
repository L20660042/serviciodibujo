# Dockerfile
FROM python:3.9-slim

# Crear un directorio de trabajo en el contenedor
WORKDIR /app

# Copiar todos los archivos del proyecto al contenedor
COPY . /app/

# Instalar las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto donde FastAPI escuchará
EXPOSE 8000

# Comando para ejecutar el servicio con Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
