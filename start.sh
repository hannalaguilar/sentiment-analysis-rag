#!/bin/bash

# Wait for Ollama
echo "⏳ Waiting for Ollama to be ready..."
ollama serve &
sleep 10
#until curl -s http://ollama:11434 > /dev/null; do
#  sleep 2
#done

#echo "🔍 Checking if model gemma3 is available..."
#if ! curl -s http://ollama:11434/api/tags | grep -q '"name":"gemma3"'; then
#  echo "⬇️ Pulling model gemma3..."
#  curl -X POST http://ollama:11434/api/pull -d '{"name": "gemma3"}'
#
#else
#  echo "✅ Model gemma3 is already available."
#fi

# Run the model
echo "Starting Gemma3 with Ollama..."
ollama run gemma3
# curl -X POST http://ollama:11434/api/pull -d '{"name": "gemma3"}' &

# Launching FastAPI in the background
echo "Run API..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Launching gradio APP
echo "Run Gradio APP..."
python gradio_app.py

