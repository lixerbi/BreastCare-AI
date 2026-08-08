# BreastCare-AI

Machine learning system for breast cancer risk classification, served through a FastAPI REST API.

## Overview

An end-to-end supervised learning pipeline trained on the Wisconsin Breast Cancer Diagnostic Dataset. Classifies tumors as benign or malignant and returns a risk level through a documented, containerized API.

Pipeline: data cleaning and stratified splitting, Logistic Regression trained with 5-fold cross-validation, hyperparameter tuning via GridSearchCV, FastAPI service layer, Docker deployment.

## Results

| Metric | Score |
|---|---|
| Accuracy | 96.49% |
| ROC-AUC | 99.60% |
| Precision (malignant) | 97% |
| Recall (malignant) | 93% |
| F1 score | 95.12% |

Test set: 114 samples (72 benign, 42 malignant), held out from a total of 569 samples with 30 numerical features each.

## Tech stack

Python, scikit-learn, FastAPI, Pydantic, Docker, Docker Compose, pytest.

## API

Interactive docs served at `/docs`.

```
POST /predict
{
  "features": [17.99, 10.38, 122.8, ...]
}
```
Returns prediction label, malignancy probability, and risk level (Low, Medium, High).

Also available: `GET /health`, `GET /features`, `POST /batch-predict`.

## Setup

```
git clone https://github.com/lixerbi/BreastCare-AI.git
cd BreastCare-AI
pip install -r requirements.txt
python main.py                      # train
uvicorn src.api.app:app --reload    # serve
```

Or with Docker: `docker-compose up --build`.

## Project structure

```
src/
  api/            FastAPI app and schemas
  config.py
  data_loader.py
  preprocessing.py
  train.py
  evaluate.py
  predict.py
tests/
models/
data/
```

## Testing

```
pytest tests/ -v --cov=src
```

## Disclaimer

This is a decision-support and educational project, not a diagnostic device. It is not FDA approved and has not been validated for clinical use. Any real-world deployment would require professional medical oversight and regulatory review.

## Links

GitHub: https://github.com/lixerbi/BreastCare-AI
