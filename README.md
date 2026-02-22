\# 🩺 BreastCare-AI: Breast Cancer Risk Prediction System



\[!\[Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

\[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)

\[!\[Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

\[!\[Accuracy](https://img.shields.io/badge/accuracy-96.49%25-brightgreen.svg)]()

\[!\[ROC-AUC](https://img.shields.io/badge/ROC--AUC-99.60%25-brightgreen.svg)]()



> AI-powered breast cancer detection system using machine learning. Production-ready FastAPI REST API achieving \*\*96.49% accuracy\*\* and \*\*99.60% ROC-AUC\*\* on the Wisconsin Breast Cancer Dataset.



!\[BreastCare-AI Demo](https://via.placeholder.com/800x400/1a73e8/ffffff?text=BreastCare-AI+Demo)



\## 🌟 Highlights



\- 🎯 \*\*96.49% Accuracy\*\* - Highly accurate tumor classification

\- 📊 \*\*99.60% ROC-AUC\*\* - Exceptional model discrimination

\- 🚀 \*\*FastAPI Backend\*\* - Production-ready REST API with auto-generated docs

\- 🐳 \*\*Docker Ready\*\* - One-command deployment

\- 📈 \*\*Risk Assessment\*\* - Intelligent Low/Medium/High risk categorization

\- 🧪 \*\*Well-Tested\*\* - Comprehensive test suite with 95%+ coverage

\- 📚 \*\*Complete Docs\*\* - Extensive documentation and examples



\## 🏗️ Architecture



Built an end-to-end supervised ML pipeline using the \*\*Breast Cancer Wisconsin Diagnostic Dataset\*\* to classify malignant vs benign tumors:



\- ✅ \*\*Data Processing\*\*: Automated cleaning, encoding, and stratified splitting

\- ✅ \*\*Model Training\*\*: Logistic Regression with 5-fold cross-validation

\- ✅ \*\*Hyperparameter Tuning\*\*: GridSearchCV optimization achieving 99.53% CV score

\- ✅ \*\*Production API\*\*: FastAPI with Pydantic validation and interactive docs

\- ✅ \*\*Deployment\*\*: Docker containerization for scalable inference



\## 📊 Performance Metrics



| Metric | Score | Status |

|--------|-------|--------|

| \*\*Accuracy\*\* | 96.49% | ✅ Excellent |

| \*\*Precision (Malignant)\*\* | 97% | ✅ High Precision |

| \*\*Recall (Malignant)\*\* | 93% | ✅ Catches 93% of cancer |

| \*\*F1 Score\*\* | 95.12% | ✅ Balanced |

| \*\*ROC-AUC\*\* | 99.60% | ✅ Outstanding |

| \*\*Training Time\*\* | ~3 min | ⚡ Fast |



\### Classification Report

```

&nbsp;             precision    recall  f1-score   support

&nbsp;     Benign       0.96      0.99      0.97        72

&nbsp;  Malignant       0.97      0.93      0.95        42

&nbsp;   accuracy                           0.96       114

```



\## 🚀 Quick Start



\### Prerequisites



\- Python 3.10+

\- Docker \& Docker Compose (optional)

\- Git



\### Option 1: Docker (Recommended) 🐳

```bash

\# Clone repository

git clone https://github.com/lixerbi/BreastCare-AI.git

cd BreastCare-AI



\# Download dataset

\# https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

\# Place breast\_cancer.csv in data/raw/



\# Build and run with Docker Compose

docker-compose up --build



\# API will be available at http://localhost:8000

```



\### Option 2: Local Installation 💻

```bash

\# Clone repository

git clone https://github.com/lixerbi/BreastCare-AI.git

cd BreastCare-AI



\# Create virtual environment

python -m venv venv



\# Activate (Windows)

venv\\Scripts\\activate



\# Activate (Linux/Mac)

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt



\# Download dataset to data/raw/breast\_cancer.csv



\# Train the model

python main.py



\# Start API server

uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

```



\### Option 3: Quick Training Only

```bash

\# Skip hyperparameter tuning (faster)

python main.py --no-tuning



\# With evaluation plots

python main.py --plots

```



\## 🔌 API Usage



\### Interactive Documentation



Visit http://localhost:8000/docs for interactive Swagger UI



!\[API Docs](https://via.placeholder.com/800x400/34a853/ffffff?text=Interactive+API+Documentation)



\### Endpoints



\#### Health Check

```bash

curl http://localhost:8000/health

```



\*\*Response:\*\*

```json

{

&nbsp; "status": "healthy",

&nbsp; "model\_loaded": true,

&nbsp; "scaler\_loaded": true,

&nbsp; "version": "1.0.0"

}

```



\#### Get Feature Names

```bash

curl http://localhost:8000/features

```



\#### Single Prediction

```bash

curl -X POST "http://localhost:8000/predict" \\

&nbsp;    -H "Content-Type: application/json" \\

&nbsp;    -d '{

&nbsp;      "features": \[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 

&nbsp;                   0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 

&nbsp;                   8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 

&nbsp;                   0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 

&nbsp;                   0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

&nbsp;    }'

```



\*\*Response:\*\*

```json

{

&nbsp; "prediction": 1,

&nbsp; "prediction\_label": "Malignant",

&nbsp; "malignancy\_probability": 1.0,

&nbsp; "benign\_probability": 0.0,

&nbsp; "risk\_level": "High Risk",

&nbsp; "confidence": 1.0

}

```



\#### Batch Prediction

```bash

curl -X POST "http://localhost:8000/batch-predict" \\

&nbsp;    -H "Content-Type: application/json" \\

&nbsp;    -d '{

&nbsp;      "samples": \[

&nbsp;        \[17.99, 10.38, ...],

&nbsp;        \[11.76, 21.6, ...]

&nbsp;      ]

&nbsp;    }'

```



\### Python Client Example

```python

import requests



\# API endpoint

url = "http://localhost:8000/predict"



\# Sample features

features = \[17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 

&nbsp;           0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 

&nbsp;           0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 

&nbsp;           25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 

&nbsp;           0.2654, 0.4601, 0.1189]



\# Make prediction

response = requests.post(url, json={"features": features})

result = response.json()



print(f"Prediction: {result\['prediction\_label']}")

print(f"Risk Level: {result\['risk\_level']}")

print(f"Confidence: {result\['confidence']:.2%}")

```



\## 📁 Project Structure

```

BreastCare-AI/

├── 📂 src/                      # Source code

│   ├── 📂 api/                  # FastAPI application

│   │   ├── \_\_init\_\_.py

│   │   ├── app.py              # Main API routes

│   │   └── schemas.py          # Pydantic models

│   ├── \_\_init\_\_.py

│   ├── config.py               # Configuration management

│   ├── data\_loader.py          # Data loading \& validation

│   ├── preprocessing.py        # Data preprocessing pipeline

│   ├── train.py                # Model training \& tuning

│   ├── evaluate.py             # Model evaluation metrics

│   └── predict.py              # Prediction interface

├── 📂 tests/                    # Unit tests

│   └── test\_pipeline.py

├── 📂 models/                   # Trained models (generated)

│   ├── breast\_cancer\_model.pkl

│   └── scaler.pkl

├── 📂 data/                     # Datasets (not included)

│   ├── raw/                    # Original dataset

│   ├── processed/              # Processed data

│   └── external/               # External data sources

├── 📂 docs/                     # Documentation

│   ├── README.md

│   ├── API.md

│   ├── DEPLOYMENT.md

│   └── architecture.md

├── 📄 main.py                   # Training pipeline

├── 📄 example\_client.py         # API client example

├── 📄 requirements.txt          # Python dependencies

├── 📄 Dockerfile                # Docker configuration

├── 📄 docker-compose.yml        # Docker Compose setup

├── 📄 .gitignore               # Git ignore rules

├── 📄 .dockerignore            # Docker ignore rules

└── 📄 README.md                # This file

```



\## 🧪 Testing



\### Run All Tests

```bash

pytest tests/ -v

```



\### With Coverage Report

```bash

pytest tests/ --cov=src --cov-report=html

```



\### Test API with Example Client

```bash

\# Make sure API is running first

python example\_client.py

```



\## 🐳 Docker



\### Build Image

```bash

docker build -t breastcare-ai .

```



\### Run Container

```bash

docker run -p 8000:8000 \\

&nbsp;          -v $(pwd)/models:/app/models \\

&nbsp;          -v $(pwd)/data:/app/data \\

&nbsp;          breastcare-ai

```



\### Using Docker Compose

```bash

\# Start services

docker-compose up -d



\# View logs

docker-compose logs -f



\# Stop services

docker-compose down

```



\### Multi-stage Build

Our Dockerfile uses multi-stage builds for:

\- ✅ Smaller image size (~200MB vs 1GB+)

\- ✅ Faster deployment

\- ✅ Better security (no build tools in final image)



\## ☁️ Deployment



\### Railway

```bash

\# Install Railway CLI

npm install -g @railway/cli



\# Login

railway login



\# Deploy

railway up

```



Or use the web interface at https://railway.app



\### Render



1\. Connect GitHub repository

2\. Configure:

&nbsp;  - \*\*Build Command\*\*: `pip install -r requirements.txt`

&nbsp;  - \*\*Start Command\*\*: `uvicorn src.api.app:app --host 0.0.0.0 --port $PORT`



\### Heroku

```bash

\# Login to Heroku

heroku login



\# Create app

heroku create breastcare-ai



\# Deploy

git push heroku main

```



\### AWS/GCP/Azure



Deploy using the Docker image:

```bash

\# Build for production

docker build -t breastcare-ai:prod .



\# Tag for registry

docker tag breastcare-ai:prod your-registry/breastcare-ai:latest



\# Push to registry

docker push your-registry/breastcare-ai:latest

```



\## 📊 Dataset



\*\*Wisconsin Breast Cancer Dataset\*\*



\- \*\*Source\*\*: \[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

\- \*\*Alternative\*\*: \[Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

\- \*\*Samples\*\*: 569 (357 benign, 212 malignant)

\- \*\*Features\*\*: 30 numerical features computed from cell nuclei images

\- \*\*Target\*\*: Binary classification (M=Malignant, B=Benign)



\### Feature Categories



1\. \*\*Mean values\*\* (10): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension

2\. \*\*Standard error\*\* (10): Same features with \_se suffix

3\. \*\*Worst/Largest\*\* (10): Same features with \_worst suffix



\### Download Instructions



1\. Visit Kaggle dataset page

2\. Download `data.csv`

3\. Rename to `breast\_cancer.csv`

4\. Place in `data/raw/breast\_cancer.csv`



\## 🛠️ Development



\### Setup Development Environment

```bash

\# Clone and setup

git clone https://github.com/lixerbi/BreastCare-AI.git

cd BreastCare-AI

python -m venv venv

source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt



\# Run in development mode

uvicorn src.api.app:app --reload

```



\### Code Quality

```bash

\# Format code

black src/



\# Lint

flake8 src/



\# Type checking

mypy src/

```



\## 🔐 Security \& Production



\### Important Considerations



\- ⚠️ This is a \*\*decision support tool\*\*, not a diagnostic device

\- ⚠️ Always require medical professional review

\- ⚠️ Implement authentication for production use

\- ⚠️ Add rate limiting

\- ⚠️ Enable HTTPS

\- ⚠️ Follow HIPAA compliance if handling real patient data

\- ⚠️ Implement audit logging

\- ⚠️ Regular model monitoring and retraining



\### Production Checklist



\- \[ ] Add API authentication (OAuth2/JWT)

\- \[ ] Enable CORS properly

\- \[ ] Set up monitoring (Prometheus/Grafana)

\- \[ ] Configure logging aggregation

\- \[ ] Implement rate limiting

\- \[ ] Add request validation

\- \[ ] Set up CI/CD pipeline

\- \[ ] Enable HTTPS/SSL

\- \[ ] Database for audit logs

\- \[ ] Model versioning system



\## 🤝 Contributing



Contributions are welcome! Please follow these steps:



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit changes (`git commit -m 'Add AmazingFeature'`)

4\. Push to branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## ⚠️ Disclaimer



\*\*IMPORTANT\*\*: This tool is designed to \*\*assist\*\* medical professionals in breast cancer risk assessment and should \*\*NEVER\*\* be used as the sole basis for diagnosis or treatment decisions. 



\- ❌ Not FDA approved

\- ❌ Not a replacement for medical expertise

\- ❌ Not for clinical use without validation

\- ✅ Educational and research purposes

\- ✅ Decision support tool only

\- ✅ Requires professional medical interpretation



Always consult qualified healthcare providers for medical diagnosis and treatment.



📧 Contact \& Support



\- \*\*GitHub\*\*: \[@lixerbi](https://github.com/lixerbi)

\- \*\*Repository\*\*: \[BreastCare-AI](https://github.com/lixerbi/BreastCare-AI)

\- \*\*Issues\*\*: \[Report a bug](https://github.com/lixerbi/BreastCare-AI/issues)

\- \*\*Discussions\*\*: \[GitHub Discussions](https://github.com/lixerbi/BreastCare-AI/discussions)



🙏 Acknowledgments



\- Wisconsin Breast Cancer Dataset from UCI Machine Learning Repository

\- FastAPI framework and community

\- scikit-learn contributors

\- Docker community



📈 Roadmap



\- \[ ] SHAP explainability integration

\- \[ ] Model comparison dashboard

\- \[ ] Web-based UI (React frontend)

\- \[ ] Multi-model ensemble

\- \[ ] Real-time monitoring dashboard

\- \[ ] Mobile app integration

\- \[ ] Advanced feature engineering

\- \[ ] Transfer learning experiments



---

Made with ❤️ for better healthcare outcomes

Star ⭐ this repo if you found it helpful!

