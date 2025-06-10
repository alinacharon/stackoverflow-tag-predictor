# image python
FROM python:3.10-slim

# system dependency
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# working direction
WORKDIR /app

# Copy all the code first
COPY . .

# Set python dependency
RUN pip install --no-cache-dir -r requirements.txt

# Loading nltk data
RUN python -c "import nltk; \
    nltk.download('punkt'); \
    nltk.download('averaged_perceptron_tagger'); \
    nltk.download('wordnet'); \
    nltk.download('stopwords'); \
    nltk.download('averaged_perceptron_tagger_eng')"

# Create logs directory
RUN mkdir -p logs

# Export port
ENV API_PORT=3000
ENV PYTHONPATH=/app

# uvicorn launch
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "3000", "--reload"]