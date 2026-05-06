# ============================================
# OptimaTradingV2 — Production Dockerfile
# ============================================
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system-level dependencies (if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application source
COPY . .

# Default port (matches fly.toml / Procfile)
EXPOSE 8080

# Start the application
CMD ["uvicorn", "main.main:app", "--host", "0.0.0.0", "--port", "8080"]
