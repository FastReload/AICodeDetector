FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    HF_HOME=/tmp/huggingface

WORKDIR /app

# Minimal OS deps for common ML wheels (kept lean; most deps ship as wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["bash", "-lc", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]

