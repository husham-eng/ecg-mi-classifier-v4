FROM python:3.11-slim

WORKDIR /app

# مكتبات نظام مطلوبة لـ opencv-python-headless + tesseract-ocr (لميزة
# اكتشاف وتقطيع اللوحة الكاملة تلقائياً عبر OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
