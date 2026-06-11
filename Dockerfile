# Epiphany — Autonomous AI Data Scientist
# Container image. Build:  docker build -t epiphany .
# Run:    docker run -p 8000:8000 --env-file .env epiphany
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 epiphany \
    && mkdir -p artifacts reports data/uploads \
    && chown -R epiphany:epiphany /app
USER epiphany

# Cloud platforms (Cloud Run, Render, Railway, ...) inject $PORT; default 8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
