# image python
FROM python:3.10-slim

# system dependency
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# working direction
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Set python dependency
RUN pip install --no-cache-dir -r requirements.txt

# Loading nltk data
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords

# Copy all the code and model
COPY . .

# Export port
ENV API_PORT=3000

# uvicorn launch
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3000"]