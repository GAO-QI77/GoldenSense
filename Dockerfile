FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.public.txt /app/requirements.public.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
RUN pip install --no-cache-dir -r /app/requirements.public.txt

ENV HF_HOME=/app/.cache/huggingface
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . /app

CMD ["python3", "scripts/public_stack.py"]
