\# 🚀 Deployment Guide



Complete guide for deploying BreastCare-AI to various platforms.



\## Table of Contents



1\. \[Docker Deployment](#docker-deployment)

2\. \[Railway](#railway)

3\. \[Render](#render)

4\. \[Heroku](#heroku)

5\. \[AWS](#aws)

6\. \[Google Cloud](#google-cloud)

7\. \[Azure](#azure)



---



\## 🐳 Docker Deployment



\### Local Docker

```bash

\# Build image

docker build -t breastcare-ai .



\# Run container

docker run -p 8000:8000 \\

&nbsp;          -v $(pwd)/models:/app/models \\

&nbsp;          breastcare-ai

```



\### Docker Compose

```bash

\# Start all services

docker-compose up -d



\# View logs

docker-compose logs -f api



\# Stop services

docker-compose down

```



\### Production Docker

```dockerfile

\# Use docker-compose.prod.yml

docker-compose -f docker-compose.prod.yml up -d

```



---



\## 🚂 Railway



\### Method 1: Web Interface



1\. Go to https://railway.app

2\. Sign in with GitHub

3\. Click \*\*New Project\*\* → \*\*Deploy from GitHub\*\*

4\. Select `BreastCare-AI` repository

5\. Railway auto-detects Dockerfile

6\. Click \*\*Deploy\*\*



\### Method 2: CLI

```bash

\# Install Railway CLI

npm install -g @railway/cli



\# Login

railway login



\# Initialize project

railway init



\# Deploy

railway up



\# Get URL

railway domain

```



\### Environment Variables



Add in Railway dashboard:

```

PORT=8000

ENVIRONMENT=production

```



\### Custom Start Command

```bash

uvicorn src.api.app:app --host 0.0.0.0 --port $PORT

```



---



\## 🎨 Render



\### Deploy from GitHub



1\. Go to https://render.com

2\. Click \*\*New\*\* → \*\*Web Service\*\*

3\. Connect GitHub repository

4\. Configure:

```

Name: breastcare-ai

Environment: Docker

Build Command: (auto-detected from Dockerfile)

Start Command: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT

```



\### render.yaml (Auto-deploy)



Create `render.yaml` in repo root:

```yaml

services:

&nbsp; - type: web

&nbsp;   name: breastcare-ai

&nbsp;   env: docker

&nbsp;   plan: free

&nbsp;   healthCheckPath: /health

&nbsp;   envVars:

&nbsp;     - key: PORT

&nbsp;       value: 8000

```



---



\## 🟣 Heroku



\### Prerequisites

```bash

\# Install Heroku CLI

\# Download from https://devcenter.heroku.com/articles/heroku-cli



\# Login

heroku login

```



\### Deploy

```bash

\# Create app

heroku create breastcare-ai-yourname



\# Deploy

git push heroku main



\# Open app

heroku open



\# View logs

heroku logs --tail

```



\### Scale Dynos

```bash

\# Scale up

heroku ps:scale web=1



\# Use larger dyno

heroku ps:resize web=standard-1x

```



---



\## ☁️ AWS



\### AWS Elastic Beanstalk

```bash

\# Install EB CLI

pip install awsebcli



\# Initialize

eb init -p docker breastcare-ai



\# Create environment

eb create breastcare-ai-env



\# Deploy

eb deploy



\# Open

eb open

```



\### AWS ECS (Fargate)



1\. Push image to ECR

2\. Create task definition

3\. Create service

4\. Configure load balancer

```bash

\# Build and push to ECR

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR\_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com



docker build -t breastcare-ai .

docker tag breastcare-ai:latest YOUR\_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breastcare-ai:latest

docker push YOUR\_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/breastcare-ai:latest

```



---



\## 🌐 Google Cloud



\### Cloud Run

```bash

\# Build and deploy

gcloud run deploy breastcare-ai \\

&nbsp; --source . \\

&nbsp; --platform managed \\

&nbsp; --region us-central1 \\

&nbsp; --allow-unauthenticated

```



\### App Engine



Create `app.yaml`:

```yaml

runtime: python310

entrypoint: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT



instance\_class: F2

automatic\_scaling:

&nbsp; min\_instances: 0

&nbsp; max\_instances: 10

```



Deploy:

```bash

gcloud app deploy

```



---



\## 🔵 Azure



\### Azure Container Instances

```bash

\# Login

az login



\# Create resource group

az group create --name breastcare-rg --location eastus



\# Deploy container

az container create \\

&nbsp; --resource-group breastcare-rg \\

&nbsp; --name breastcare-ai \\

&nbsp; --image YOUR\_REGISTRY/breastcare-ai:latest \\

&nbsp; --dns-name-label breastcare-ai \\

&nbsp; --ports 8000

```



\### Azure App Service

```bash

\# Create App Service plan

az appservice plan create \\

&nbsp; --name breastcare-plan \\

&nbsp; --resource-group breastcare-rg \\

&nbsp; --is-linux



\# Create web app

az webapp create \\

&nbsp; --resource-group breastcare-rg \\

&nbsp; --plan breastcare-plan \\

&nbsp; --name breastcare-ai \\

&nbsp; --deployment-container-image-name YOUR\_REGISTRY/breastcare-ai:latest

```



---



\## 🔒 Production Checklist



Before deploying to production:



\### Security

\- \[ ] Enable HTTPS/SSL

\- \[ ] Add API authentication

\- \[ ] Implement rate limiting

\- \[ ] Set up CORS properly

\- \[ ] Remove debug mode

\- \[ ] Secure environment variables



\### Monitoring

\- \[ ] Set up logging

\- \[ ] Configure error tracking (Sentry)

\- \[ ] Add performance monitoring

\- \[ ] Set up uptime monitoring

\- \[ ] Create alerts



\### Scalability

\- \[ ] Configure auto-scaling

\- \[ ] Set up load balancer

\- \[ ] Optimize Docker image

\- \[ ] Configure caching

\- \[ ] Database for audit logs



\### Compliance

\- \[ ] HIPAA compliance (if applicable)

\- \[ ] GDPR compliance

\- \[ ] Data encryption

\- \[ ] Audit logging

\- \[ ] Privacy policy



---



\## 📊 Performance Tips



1\. \*\*Use gunicorn\*\* with multiple workers:

```bash

&nbsp;  gunicorn src.api.app:app -w 4 -k uvicorn.workers.UvicornWorker

```



2\. \*\*Enable compression\*\* in production



3\. \*\*Use CDN\*\* for static assets



4\. \*\*Cache predictions\*\* when appropriate



5\. \*\*Optimize Docker image\*\*:

```dockerfile

&nbsp;  # Multi-stage build reduces size by 70%

```



---



\## 🆘 Troubleshooting



\### Issue: Model not found

```bash

\# Make sure models are in the image

docker run -v $(pwd)/models:/app/models breastcare-ai

```



\### Issue: Port already in use

```bash

\# Change port

docker run -p 8001:8000 breastcare-ai

```



\### Issue: Out of memory

```bash

\# Increase container memory

docker run -m 512m breastcare-ai

```



---



\## 📈 Scaling Strategy



\### Horizontal Scaling

```yaml

\# docker-compose with replicas

services:

&nbsp; api:

&nbsp;   deploy:

&nbsp;     replicas: 3

```



\### Vertical Scaling



Increase resources:

\- CPU: 1 → 2 cores

\- Memory: 512MB → 1GB



\### Load Balancing



Use nginx or cloud load balancer:

```nginx

upstream breastcare {

&nbsp;   server api1:8000;

&nbsp;   server api2:8000;

&nbsp;   server api3:8000;

}

```



---



\## 🎯 Next Steps



After deployment:



1\. Test all endpoints

2\. Monitor performance

3\. Set up CI/CD

4\. Configure backups

5\. Plan for scaling



---



\## 📞 Support



Issues with deployment? 

\- GitHub Issues: https://github.com/lixerbi/BreastCare-AI/issues

\- Discussions: https://github.com/lixerbi/BreastCare-AI/discussions

