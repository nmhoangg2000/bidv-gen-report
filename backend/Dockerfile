FROM python:3.11-slim

# System deps: pandoc for docx/pdf text extraction
RUN apt-get update && apt-get install -y \
    pandoc \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
