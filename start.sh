#!/bin/bash

# Wait for Ollama
echo "⏳ Waiting for Ollama to be ready..."
ollama serve &
sleep 10

# Run the model
echo "Starting Gemma3 with Ollama..."
ollama run gemma3

# Launching FastAPI in the background
echo "Run API..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Launching gradio APP
echo "Run Gradio APP..."
python gradio_app.py

