from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import time
from contextlib import asynccontextmanager
import re
import joblib
import numpy as np
import tensorflow_hub as hub
import nltk
from bs4 import BeautifulSoup
from nltk import pos_tag
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Загружаем необходимые nltk пакеты
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('API_PORT', '3000'))

model = None
mlb = None
use_model = None


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN


def clean_text(text: str) -> str:
    tokenizer = TreebankWordTokenizer()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    exceptions = {'pandas', 'keras'}

    text = text.strip()
    text = text.replace('c++', 'cplusplus').replace('c#', 'csharp').replace('next.js', 'nextjs').replace('node.js', 'nodejs')
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace('cplusplus', 'c++').replace('csharp', 'c#').replace('nextjs', 'next.js').replace('nodejs', 'node.js')

    tokens = tokenizer.tokenize(text)
    tagged = pos_tag(tokens)
    tokens = [word if word in exceptions else lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged]
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, mlb, use_model
    logger.info("Loading model, mlb and USE embedding model...")
    model_dir = '../models/'
    model = joblib.load(f'{model_dir}/model.pkl')
    mlb = joblib.load(f'{model_dir}/mlb.pkl')
    use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    logger.info("Models loaded successfully")
    yield
    model = None
    mlb = None
    use_model = None


app = FastAPI(
    title="Stack Overflow Tag Predictor",
    description="API to predict tags for Stack Overflow questions",
    version="1.0.0",
    lifespan=lifespan
)


class Question(BaseModel):
    text: str = Field(..., min_length=1, description="Question text to predict tags for")


class Prediction(BaseModel):
    tags: list[str] = Field(..., min_length=1, description="Predicted tags")


@app.post("/predict", response_model=Prediction)
async def predict(question: Question):
    try:
        global model, mlb, use_model
        if model is None or mlb is None or use_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        logger.info(f"Original text: '{question.text}'")
        start_time = time.time()

        cleaned_text = clean_text(question.text)
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Invalid input text after cleaning")

        embeddings = use_model([cleaned_text])  # shape (1, embedding_dim)
        prediction = model.predict(embeddings)
        tags = mlb.inverse_transform(prediction)[0]

        if not tags:
            raise HTTPException(status_code=400, detail="No tags predicted")

        elapsed = time.time() - start_time
        logger.info(f"Prediction done in {elapsed:.3f} sec. Tags: {tags}")

        return Prediction(tags=list(tags))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}