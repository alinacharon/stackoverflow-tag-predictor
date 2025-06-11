import os
import pickle
import logging
import traceback
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model = None
mlb = None
use_model = None

# Создание FastAPI приложения
app = FastAPI(title="Stack Overflow Tag Predictor API")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]


def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of texts using Universal Sentence Encoder"""
    try:
        # Преобразуем входные данные в тензор
        text_tensor = tf.convert_to_tensor(texts)
        logger.info(f"Input tensor shape: {text_tensor.shape}")

        # Получаем эмбеддинги
        embeddings = use_model(text_tensor)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Преобразуем в numpy массив
        embeddings_np = embeddings.numpy()
        logger.info(f"Numpy embeddings shape: {embeddings_np.shape}")

        return embeddings_np
    except Exception as e:
        logger.error(f"Error in get_embeddings: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


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
                model = pickle.load(f)
                if isinstance(model, xgb.XGBClassifier):
                    model_params = model.get_params()
                    model_params['use_label_encoder'] = False
                    new_model = xgb.XGBClassifier(**model_params)
                    new_model._Booster = model._Booster
                    model = new_model
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


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    init_models()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextInput):
    """Predict tags for a given text"""
    try:
     
        if not request.text or not isinstance(request.text, str):
            raise HTTPException(
                status_code=422, detail="Text must be a non-empty string")

      
        try:
            embeddings = get_embeddings([request.text])
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail="Error generating embeddings")

     
        try:
            predictions = model.predict_proba(embeddings)
            logger.info(f"Raw predictions shape: {predictions.shape}")
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail="Error making predictions")


        try:
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_tags = mlb.classes_[top_indices]
            top_probs = predictions[0][top_indices]

            result = [
                {"tag": tag, "probability": float(prob)}
                for tag, prob in zip(top_tags, top_probs)
            ]

            logger.info(f"Generated predictions: {result}")
            return {"predictions": result}

        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, detail="Error processing predictions")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal server error")
