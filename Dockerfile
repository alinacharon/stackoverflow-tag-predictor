# Use multi-stage build
FROM python:3.11-slim AS builder

# Install only necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy only dependency files
COPY requirements.txt .

# Install dependencies in virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies in smaller chunks to manage memory better
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('wordnet'); \
    nltk.download('stopwords'); \
    nltk.download('averaged_perceptron_tagger_eng')"

# Pre-download USE model
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"

# Final stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Working directory
WORKDIR /app

# Copy only necessary files
COPY api/ api/
COPY models/ models/

# Create necessary directories and set permissions
RUN mkdir -p logs /home/appuser/.cache && \
    chown -R appuser:appuser logs /home/appuser && \
    chmod 755 logs

# Set environment variables
ENV API_PORT=3000
ENV PYTHONPATH=/app
ENV LOG_LEVEL=info
ENV LOG_DIR=/app/logs
ENV TFHUB_CACHE_DIR=/home/appuser/.cache/tfhub_modules

# Switch to non-root user
USER appuser

# Add healthcheck
HEALTHCHECK --interval=15s --timeout=10s --start-period=160s --retries=5 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run uvicorn with optimized settings
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--limit-concurrency", "100", "--timeout-keep-alive", "30", "--log-level", "info"]