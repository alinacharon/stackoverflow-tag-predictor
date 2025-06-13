# Stack Overflow Tag Predictor API - Technical Documentation

## Project Overview

This project implements a machine learning-based API service that automatically predicts relevant tags for Stack Overflow questions. The system uses Natural Language Processing (NLP) and deep learning techniques to analyze question text and suggest appropriate tags.

## System Architecture

### Core Components

1. **FastAPI Application (`main.py`)**

   - Main application entry point
   - Implements REST API endpoints
   - Handles model loading and inference
   - Key features:
     - Question text preprocessing
     - Model prediction pipeline
     - Health check endpoint
     - Error handling and logging

2. **Logging System (`logger.py`)**

   - Custom logging configuration
   - Features:
     - Timed rotating file handler
     - Console output
     - UTF-8 encoding support
     - 7-day log retention

3. **Docker Configuration (`Dockerfile`)**

   - Multi-stage build process
   - Key features:
     - Python 3.11 base image
     - System dependencies management
     - Virtual environment setup
     - Health check implementation


4. **Testing Framework**
   - GitHub Actions workflow (`test.yml`)
   - Pytest implementation (`test_api.py`)
   - Features:
     - Automated testing pipeline
     - Server health verification
     - Model file validation
     - Performance testing
     - Error handling tests

## Technical Implementation Details

### Model Architecture

- Uses Universal Sentence Encoder (USE) for text embeddings
- Implements NLTK for text preprocessing:
  - Tokenization
  - Lemmatization
  - Part-of-speech tagging
  - Stop word removal
- Custom text cleaning pipeline for technical terms

### API Endpoints

1. **Health Check (`/health`)**

   - Verifies model loading status
   - Returns service health status

2. **Prediction (`/predict`)**
   - Accepts question text
   - Returns predicted tags
   - Implements error handling
   - Performance monitoring

### Data Flow

1. Text input received
2. Preprocessing pipeline:
   - HTML cleaning
   - Special character handling
   - Tokenization
   - Lemmatization
3. USE model embedding generation
4. Tag prediction
5. Response formatting


## Development and Deployment

### Local Development

1. Python 3.11 environment
2. Required dependencies in `requirements.txt`
3. Model files in `models/` directory
4. NLTK data downloads

### CI/CD Pipeline

- GitHub Actions workflow
- Automated testing
- Model file verification
- Server health checks

### Docker Deployment

1. Multi-stage build
2. Health monitoring

## Model Management

- Git LFS for large model files

## Testing Strategy

1. Unit tests
2. Integration tests
3. Performance tests
4. Error handling tests
5. Health check verification

## Monitoring and Logging

- Structured logging
- Error tracking
- Health monitoring

## Future Improvements

1. Model versioning system
2. Enhanced error handling
3. Additional API endpoints
4. Extended test coverage

