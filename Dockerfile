FROM python:3.11-slim

# Install system dependencies FIRST
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
COPY . .

# CPU-only torch (much smaller)
RUN pip install --no-cache-dir torch==2.0.0+cpu torchvision==0.15.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir transformers fastapi uvicorn[standard] pydub librosa scikit-learn soundfile numpy

EXPOSE 8000

# CORRECT (Railway auto-sets $PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
