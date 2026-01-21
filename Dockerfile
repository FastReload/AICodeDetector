FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  PORT=8080 \
  TRANSFORMERS_CACHE=/tmp/huggingface \
  HF_HOME=/tmp/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  && rm -rf /var/lib/apt/lists/*

# Install PyTorch (Standard GPU-enabled version)
RUN pip install --no-cache-dir torch>=2.1.0

# Copy and install other requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port
EXPOSE 8080

# Healthcheck to Ensure the Container is Ready
HEALTHCHECK CMD curl --fail http://localhost:${PORT:-8080}/_stcore/health || exit 1

# Run Streamlit (address overridable via SERVER_ADDRESS env, defaults to 0.0.0.0 for Cloud Run)
ENV SERVER_ADDRESS=0.0.0.0
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=${SERVER_ADDRESS} --server.enableCORS=false --server.enableXsrfProtection=false"]

