# StackOverflow Tags Prediction API 🚀

This API predicts tags for StackOverflow questions.

## 📦 Features

- Multi-label classification
- Universal Sentence Encoder for embeddings
- Model using pre-trained embeddings
- FastAPI with asynchronous processing
- Logging support
- Health check endpoint

## 🛠 Tech Stack

- FastAPI
- TensorFlow Hub (Universal Sentence Encoder)
- NLTK for text preprocessing
- BeautifulSoup for HTML cleaning
- Joblib for model loading


## 🔄 Endpoints

- `GET /` - API health check
- `POST /predict` - Predict tags for a question
- `GET /health` - API status check
