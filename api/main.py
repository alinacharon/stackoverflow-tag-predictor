from api.logger import logger
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk import pos_tag
from bs4 import BeautifulSoup
import nltk
import tensorflow_hub as hub
import numpy as np
import joblib
import re
from contextlib import asynccontextmanager
import time
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
import os
import sys
import pickle

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

model = None
mlb = None
use_model = None


def init_models():
    global model, mlb, use_model
    try:
        logger.info("Starting model loading...")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Determine the absolute path to the directory containing main.py
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the path to the 'models' directory
        project_root = os.path.dirname(current_script_dir)
        models_dir = os.path.join(project_root, 'models')

        # Load models directly from local path
        model_path = os.path.join(models_dir, 'model.pkl')
        mlb_path = os.path.join(models_dir, 'mlb.pkl')

        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"DEBUG: models_dir computed as: {models_dir}")

        try:
            # Try loading with joblib first
            model = joblib.load(model_path)
            logger.info(
                f"✓ model.pkl loaded successfully via joblib. Type: {type(model)}")
        except Exception as e:
            logger.error(f"Failed to load model with joblib: {e}")
            # Fallback to pickle if joblib fails
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(
                f"✓ model.pkl loaded successfully via pickle. Type: {type(model)}")

        logger.info("Loading mlb.pkl...")
        try:
            # Try loading with joblib first
            mlb = joblib.load(mlb_path)
            logger.info(
                f"✓ mlb.pkl loaded successfully via joblib. Type: {type(mlb)}")
        except Exception as e:
            logger.error(f"Failed to load mlb with joblib: {e}")
            # Fallback to pickle if joblib fails
            with open(mlb_path, 'rb') as f:
                mlb = pickle.load(f)
            logger.info(
                f"✓ mlb.pkl loaded successfully via pickle. Type: {type(mlb)}")

        # Load USE model
        logger.info("Loading USE embedding model...")
        use_model = hub.load(os.getenv(
            'USE_MODEL_URL', "https://tfhub.dev/google/universal-sentence-encoder/4"))
        logger.info("✓ USE model loaded successfully")

        logger.info("Model loading complete!")

    except Exception as e:
        logger.error(f"Error during model loading: {e}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


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
    text = text.replace('c++', 'cplusplus').replace('c#',
                                                    'csharp').replace('next.js', 'nextjs').replace('node.js', 'nodejs')
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace('cplusplus', 'c++').replace('csharp',
                                                    'c#').replace('nextjs', 'next.js').replace('nodejs', 'node.js')

    tokens = tokenizer.tokenize(text)
    tagged = pos_tag(tokens)
    tokens = [word if word in exceptions else lemmatizer.lemmatize(
        word, get_wordnet_pos(tag)) for word, tag in tagged]
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_models()
    yield
    global model, mlb, use_model
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
    text: str = Field(..., min_length=1,
                      description="Question text to predict tags for")


class Prediction(BaseModel):
    tags: list[str] = Field(..., min_length=1, description="Predicted tags")


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.post("/predict", response_model=Prediction)
def predict(question: Question):
    try:
        global model, mlb, use_model
        if model is None or mlb is None or use_model is None:
            raise HTTPException(
                status_code=500, detail="Models not loaded")

        logger.info(f"Original text: '{question.text}'")
        start_time = time.time()

        cleaned_text = clean_text(question.text)
        if not cleaned_text:
            raise HTTPException(
                status_code=400, detail="Invalid input text after cleaning")

        # Get embeddings and make prediction
        embeddings = use_model([cleaned_text])
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


@app.get('/health')
async def health_check():
    if model is not None and mlb is not None and use_model is not None:
        return {'status': 'healthy'}
    return {'status': 'error', 'message': 'Models not loaded'}
