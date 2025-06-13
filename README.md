# StackOverflow Tags Prediction API 🚀

This API predicts tags for StackOverflow questions.

## 🌐 Live Demo

The API is available at: [http://13.39.79.254:3000/docs](http://13.39.79.254:3000/docs)

API can be tested directly through the interactive Swagger documentation interface.

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
