# Use multi-stage build
FROM python:3.10-slim as builder

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
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('wordnet'); \
    nltk.download('stopwords'); \
    nltk.download('averaged_perceptron_tagger_eng')"

# Final stage
FROM python:3.10-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Working directory
WORKDIR /app

# Copy only necessary files
COPY api/ api/

# Create logs directory
RUN mkdir -p logs

# Export port and environment variables
ENV API_PORT=3000
ENV PYTHONPATH=/app
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL}

# Pre-download USE model
RUN python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')"

# Add healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run uvicorn with optimized settings
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1", "--limit-concurrency", "100", "--timeout-keep-alive", "30", "--log-level", "info"]