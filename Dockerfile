# Use the official Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the contents of the local directory into the container
COPY . /app

# Install the necessary dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port on which the FastAPI app will run
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
