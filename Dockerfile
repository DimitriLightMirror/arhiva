FROM python:3.11-slim

# System dependencies: Tesseract with Romanian language data + ocrmypdf
# for proper PDF/A output. libgl is needed by some OpenCV builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ron \
    ocrmypdf \
    ghostscript \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/arhiva

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env.example ./.env.example

ENV ARCHIVE_ROOT=/data/archive \
    WATCH_FOLDER=/data/watch \
    DATA_DIR=/data/jobs

RUN mkdir -p /data/archive /data/watch /data/jobs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
