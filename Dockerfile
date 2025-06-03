# Use Pythjon 3.11 image
FROM python:3.11-slim

# Create workdir
WORKDIR /app

COPY requirements.txt /app

# Install basic
RUN apt-get update  \
    && apt-get install -y sudo git curl && apt-get clean

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies listed in the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Ports for FastAPI and Gradio
EXPOSE 8000
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run bash
CMD ["/bin/bash"]

