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
import boto3
from botocore.config import Config

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION')
AWS_ENDPOINT_URL = os.getenv('AWS_ENDPOINT_URL')
AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')

# Configure AWS client
s3_config = Config(
    region_name=AWS_DEFAULT_REGION,
    endpoint_url=AWS_ENDPOINT_URL
)

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=s3_config
)

# nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


# logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            'logs/api.log') if os.path.exists('logs') else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv('API_PORT', '3000'))

model = None
mlb = None
use_model = None


def download_from_s3(bucket_name: str, object_key: str, local_path: str):
    """Download file from S3 bucket"""
    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        logger.info(f"Successfully downloaded {object_key} from {bucket_name}")
    except Exception as e:
        logger.error(f"Error downloading from S3: {e}")
        raise


def init_models():
    global model, mlb, use_model
    try:
        logger.info("Starting model loading...")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Create models directory if it doesn't exist
        models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(models_dir, exist_ok=True)

        # Download models from S3 if they don't exist locally
        model_path = os.path.join(models_dir, 'model.pkl')
        mlb_path = os.path.join(models_dir, 'mlb.pkl')

        if not os.path.exists(model_path):
            download_from_s3(AWS_BUCKET_NAME,
                             'models/model.pkl', model_path)
        if not os.path.exists(mlb_path):
            download_from_s3(AWS_BUCKET_NAME, 'models/mlb.pkl', mlb_path)

        logger.info("Loading model.pkl...")
        model = joblib.load(model_path)
        logger.info(f"✓ model.pkl loaded successfully. Type: {type(model)}")

        logger.info("Loading mlb.pkl...")
        mlb = joblib.load(mlb_path)
        logger.info(f"✓ mlb.pkl loaded successfully. Type: {type(mlb)}")

        logger.info("Loading USE embedding model...")
        use_model = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4")
        logger.info("✓ USE model loaded successfully")

        logger.info("All models loaded successfully!")

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
            raise HTTPException(status_code=500, detail="Model not loaded")

        logger.info(f"Original text: '{question.text}'")
        start_time = time.time()

        cleaned_text = clean_text(question.text)
        if not cleaned_text:
            raise HTTPException(
                status_code=400, detail="Invalid input text after cleaning")

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


@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}
