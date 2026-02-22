\# 🔌 API Documentation



Complete API reference for BreastCare-AI REST API.



\## Base URL

```

Local: http://localhost:8000

Production: https://your-deployment-url.com

```



\## Authentication



Currently no authentication required. For production deployment, implement:

\- API Keys

\- OAuth2

\- JWT tokens



\## Endpoints



\### 1. Root Endpoint



\*\*GET\*\* `/`



Returns API information and available endpoints.



\*\*Response:\*\*

```json

{

&nbsp; "message": "Breast Cancer Detection API",

&nbsp; "version": "1.0.0",

&nbsp; "endpoints": {

&nbsp;   "docs": "/docs",

&nbsp;   "health": "/health",

&nbsp;   "predict": "/predict",

&nbsp;   "batch\_predict": "/batch-predict",

&nbsp;   "features": "/features"

&nbsp; }

}

```



---



\### 2. Health Check



\*\*GET\*\* `/health`



Check API and model status.



\*\*Response:\*\*

```json

{

&nbsp; "status": "healthy",

&nbsp; "model\_loaded": true,

&nbsp; "scaler\_loaded": true,

&nbsp; "version": "1.0.0"

}

```



\*\*Status Codes:\*\*

\- `200 OK` - Service healthy

\- `503 Service Unavailable` - Model not loaded



---



\### 3. Get Feature Names



\*\*GET\*\* `/features`



Returns list of required feature names in order.



\*\*Response:\*\*

```json

{

&nbsp; "features": \[

&nbsp;   "radius\_mean",

&nbsp;   "texture\_mean",

&nbsp;   "perimeter\_mean",

&nbsp;   ...

&nbsp; ],

&nbsp; "total\_features": 30

}

```



---



\### 4. Single Prediction



\*\*POST\*\* `/predict`



Make a prediction for a single tumor sample.



\*\*Request Body:\*\*

```json

{

&nbsp; "features": \[

&nbsp;   17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,

&nbsp;   0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,

&nbsp;   0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,

&nbsp;   184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189

&nbsp; ]

}

```



\*\*Validation:\*\*

\- Exactly 30 features required

\- All values must be non-negative

\- Values must be numeric



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



\*\*Risk Levels:\*\*

\- `Low Risk`: < 30% malignancy probability

\- `Medium Risk`: 30-70% malignancy probability

\- `High Risk`: > 70% malignancy probability



\*\*Status Codes:\*\*

\- `200 OK` - Prediction successful

\- `422 Unprocessable Entity` - Validation error

\- `503 Service Unavailable` - Model not loaded



---



\### 5. Batch Prediction



\*\*POST\*\* `/batch-predict`



Make predictions for multiple samples.



\*\*Request Body:\*\*

```json

{

&nbsp; "samples": \[

&nbsp;   \[17.99, 10.38, ...],

&nbsp;   \[11.76, 21.6, ...]

&nbsp; ]

}

```



\*\*Limits:\*\*

\- Minimum: 1 sample

\- Maximum: 100 samples per request



\*\*Response:\*\*

```json

{

&nbsp; "predictions": \[

&nbsp;   {

&nbsp;     "prediction": 1,

&nbsp;     "prediction\_label": "Malignant",

&nbsp;     "malignancy\_probability": 1.0,

&nbsp;     "benign\_probability": 0.0,

&nbsp;     "risk\_level": "High Risk",

&nbsp;     "confidence": 1.0

&nbsp;   },

&nbsp;   {

&nbsp;     "prediction": 0,

&nbsp;     "prediction\_label": "Benign",

&nbsp;     "malignancy\_probability": 0.0021,

&nbsp;     "benign\_probability": 0.9979,

&nbsp;     "risk\_level": "Low Risk",

&nbsp;     "confidence": 0.9979

&nbsp;   }

&nbsp; ],

&nbsp; "total\_samples": 2

}

```



---



\## Error Responses



\### Validation Error (422)

```json

{

&nbsp; "detail": \[

&nbsp;   {

&nbsp;     "loc": \["body", "features"],

&nbsp;     "msg": "Exactly 30 features are required",

&nbsp;     "type": "value\_error"

&nbsp;   }

&nbsp; ]

}

```



\### Model Not Loaded (503)

```json

{

&nbsp; "detail": "Model not loaded. Please check server logs."

}

```



\### Internal Server Error (500)

```json

{

&nbsp; "detail": "An unexpected error occurred"

}

```



---



\## Code Examples



\### Python

```python

import requests



url = "http://localhost:8000/predict"

features = \[17.99, 10.38, ...]  # 30 features



response = requests.post(url, json={"features": features})

result = response.json()



print(f"Prediction: {result\['prediction\_label']}")

print(f"Risk: {result\['risk\_level']}")

```



\### JavaScript

```javascript

const url = 'http://localhost:8000/predict';

const features = \[17.99, 10.38, ...];  // 30 features



fetch(url, {

&nbsp; method: 'POST',

&nbsp; headers: { 'Content-Type': 'application/json' },

&nbsp; body: JSON.stringify({ features })

})

.then(res => res.json())

.then(data => {

&nbsp; console.log('Prediction:', data.prediction\_label);

&nbsp; console.log('Risk:', data.risk\_level);

});

```



\### cURL

```bash

curl -X POST "http://localhost:8000/predict" \\

&nbsp;    -H "Content-Type: application/json" \\

&nbsp;    -d '{"features": \[17.99, 10.38, ...]}'

```



---



\## Interactive Documentation



Visit `/docs` for interactive Swagger UI where you can:

\- ✅ Test all endpoints

\- ✅ See request/response schemas

\- ✅ Try different inputs

\- ✅ Download OpenAPI spec



Visit `/redoc` for alternative documentation interface.



---



\## Rate Limiting



\*\*Current\*\*: No rate limiting



\*\*Recommended for production\*\*:

\- 100 requests per minute per IP

\- 1000 requests per hour per API key

\- Implement using middleware or API gateway



---



\## Best Practices



1\. \*\*Always validate input\*\* before sending

2\. \*\*Handle errors gracefully\*\* in your application

3\. \*\*Don't send PHI\*\* (Protected Health Information) without encryption

4\. \*\*Implement timeout\*\* (5-10 seconds recommended)

5\. \*\*Cache results\*\* when appropriate

6\. \*\*Log predictions\*\* for audit trails

7\. \*\*Monitor API health\*\* regularly



---



\## Support



\- Report issues: https://github.com/lixerbi/BreastCare-AI/issues

\- API questions: GitHub Discussions

