# ──────────────────────────────────────────────────────────────────────────────
# RealEyes Backend Container Definition
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and C compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy full application code
COPY backend/ /app/backend/
COPY .env /app/.env

WORKDIR /app/backend

EXPOSE 5001

CMD ["python", "server.py"]
