# ==============================
# STAGE 1: Builder
# ==============================
FROM python:3.11-slim-bullseye AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Create virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

# Install pip and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('wordnet'); \
    nltk.download('stopwords')"

# ==============================
# STAGE 2: Final image
# ==============================
FROM python:3.11-slim-bullseye

# Create user
RUN useradd -m -u 1000 appuser

# Working directory
WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application and models
COPY api/ api/
COPY models/ models/

# Setup cache and logs
RUN mkdir -p /home/appuser/.cache/tfhub_modules && \
    mkdir -p logs && \
    chown -R appuser:appuser /app /home/appuser && \
    chmod 755 logs

# Environment variables
ENV API_PORT=3000
ENV PYTHONPATH=/app
ENV LOG_LEVEL=info
ENV LOG_DIR=/app/logs
ENV TFHUB_CACHE_DIR=/app/models/use
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=15s --timeout=10s --start-period=160s --retries=5 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run FastAPI with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--limit-concurrency", "100", "--timeout-keep-alive", "30", "--log-level", "info"]