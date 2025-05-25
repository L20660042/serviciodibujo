# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY drawing_emotion_service.py .

# Expose port
EXPOSE 8001

# Command to run the app with uvicorn
CMD ["uvicorn", "drawing_emotion_service:app", "--host", "0.0.0.0", "--port", "8001"]
