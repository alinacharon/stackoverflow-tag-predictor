from .logger import logger
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import tensorflow as tf
import tensorflow_text
import pickle
from typing import List, Dict, Any
import xgboost as xgb
import json
import traceback
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
    """Initialize models at startup"""
    global model, mlb, use_model

    try:
        logger.info("Starting model loading...")

        # Log current working directory and paths
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(
            f"Current script directory: {os.path.dirname(os.path.abspath(__file__))}")
        logger.info(
            f"Project root: {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

        # Construct absolute paths
        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(project_root, "models")
        model_path = os.path.join(models_dir, "model.pkl")
        mlb_path = os.path.join(models_dir, "mlb.pkl")

        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Files in models directory: {os.listdir(models_dir)}")

        # Check if files exist and log their sizes
        if os.path.exists(model_path):
            logger.info(
                f"File model.pkl size: {os.path.getsize(model_path)} bytes")
        if os.path.exists(mlb_path):
            logger.info(
                f"File mlb.pkl size: {os.path.getsize(mlb_path)} bytes")

        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        logger.info(f"MLB file exists: {os.path.exists(mlb_path)}")

        if os.path.exists(model_path):
            logger.info(
                f"Model file size: {os.path.getsize(model_path)} bytes")
        if os.path.exists(mlb_path):
            logger.info(f"MLB file size: {os.path.getsize(mlb_path)} bytes")

        # Load XGBoost model
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
                model = xgb.XGBClassifier()
                model.load_model(model_data)
                logger.info("✓ XGBoost model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading XGBoost model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Load MLB
        try:
            with open(mlb_path, 'rb') as f:
                mlb = pickle.load(f)
                logger.info("✓ MLB loaded successfully")
        except Exception as e:
            logger.error(f"Error loading MLB: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

        # Load USE model
        try:
            use_model = hub.load(
                "https://tfhub.dev/google/universal-sentence-encoder/4")
            logger.info("✓ USE model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading USE model: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
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
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
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


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings from USE model"""
    try:

        if isinstance(texts, str):
            texts = [texts]

        text_input = tf.constant(texts, dtype=tf.string)
        logger.info(f"Input tensor shape: {text_input.shape}")

        embeddings = use_model(text_input)
        logger.info(f"Raw embeddings shape: {embeddings.shape}")

        embeddings = embeddings.numpy()
        logger.info(f"Final embeddings shape: {embeddings.shape}")

        return embeddings

    except Exception as e:
        logger.error(f"Error in get_embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting embeddings: {str(e)}"
        )


def load_model():
    """Load the model and MLB"""
    try:
        logger.info("Starting model loading...")

        # Get the absolute path to the models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, "models")

        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Current script directory: {current_dir}")
        logger.info(f"Project root: {project_root}")
        logger.info(f"Models directory: {models_dir}")

        # List all files in the models directory
        logger.info(f"Files in models directory: {os.listdir(models_dir)}")

        model_path = os.path.join(models_dir, "model.pkl")
        mlb_path = os.path.join(models_dir, "mlb.pkl")

        # Log file sizes
        logger.info(
            f"File model.pkl size: {os.path.getsize(model_path)} bytes")
        logger.info(f"File mlb.pkl size: {os.path.getsize(mlb_path)} bytes")

        logger.info(f"Attempting to load model from: {model_path}")
        logger.info(f"Model file exists: {os.path.exists(model_path)}")
        logger.info(f"MLB file exists: {os.path.exists(mlb_path)}")
        logger.info(f"Model file size: {os.path.getsize(model_path)} bytes")
        logger.info(f"MLB file size: {os.path.getsize(mlb_path)} bytes")

        # Load the model with the correct version of XGBoost
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            # Создаем новый экземпляр XGBClassifier с теми же параметрами
            if isinstance(model, xgb.XGBClassifier):
                model_params = model.get_params()
                # Добавляем параметр use_label_encoder
                model_params['use_label_encoder'] = False
                model = xgb.XGBClassifier(**model_params)
                # Копируем внутренний бустер
                model._Booster = model._Booster
                # Устанавливаем атрибуты, которые могут быть необходимы
                model._le = model._le if hasattr(model, '_le') else None
                model._estimator_type = 'classifier'
            logger.info(
                f"✓ model.pkl loaded successfully. Type: {type(model)}")

        logger.info("Loading mlb.pkl...")
        with open(mlb_path, 'rb') as f:
            mlb = pickle.load(f)
            logger.info(f"✓ mlb.pkl loaded successfully. Type: {type(mlb)}")

        logger.info(
            "Model loading complete (USE model will be loaded on first use)!")
        return model, mlb

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")

        if not text:
            raise HTTPException(
                status_code=422,
                detail="Text is required"
            )

        if not isinstance(text, str):
            raise HTTPException(
                status_code=422,
                detail="Text must be a string"
            )

        if len(text) > 10000:
            raise HTTPException(
                status_code=422,
                detail="Text is too long (max 10000 characters)"
            )

        try:
            embeddings = get_embeddings(text)
            logger.info(f"Got embeddings with shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting embeddings: {str(e)}"
            )

        try:
            # Используем predict_proba вместо predict
            predictions = model.predict_proba(embeddings)
            logger.info(f"Raw predictions shape: {predictions.shape}")
            # Преобразуем вероятности в бинарные предсказания
            predictions = (predictions > 0.5).astype(int)
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error making prediction: {str(e)}"
            )

        try:
            tags = mlb.inverse_transform(predictions)
            logger.info(f"Transformed tags: {tags}")
        except Exception as e:
            logger.error(f"Error transforming predictions to tags: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error transforming predictions to tags: {str(e)}"
            )

        response = {
            "tags": tags[0].tolist() if len(tags) > 0 else []
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in predict: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@app.get('/health')
async def health_check():
    if model is not None and mlb is not None:
        return {'status': 'healthy'}
    return {'status': 'error', 'message': 'Models not loaded'}
