"""
API request/response schemas
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint"""
    features: List[float] = Field(
        ...,
        description="List of 30 feature values from tumor measurements",
        min_items=30,
        max_items=30
    )
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 30:
            raise ValueError('Exactly 30 features are required')
        if any(x < 0 for x in v):
            raise ValueError('Feature values must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
                    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
                    0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
                    184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint"""
    prediction: int = Field(..., description="Binary prediction (0=Benign, 1=Malignant)")
    prediction_label: str = Field(..., description="Human-readable prediction label")
    malignancy_probability: float = Field(..., description="Probability of malignancy (0-1)")
    benign_probability: float = Field(..., description="Probability of benign tumor (0-1)")
    risk_level: str = Field(..., description="Risk assessment level")
    confidence: float = Field(..., description="Model confidence (max probability)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "Malignant",
                "malignancy_probability": 0.92,
                "benign_probability": 0.08,
                "risk_level": "High Risk",
                "confidence": 0.92
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint"""
    samples: List[List[float]] = Field(
        ...,
        description="List of samples, each containing 30 feature values",
        min_items=1,
        max_items=100
    )
    
    @validator('samples')
    def validate_samples(cls, v):
        for i, sample in enumerate(v):
            if len(sample) != 30:
                raise ValueError(f'Sample {i} must have exactly 30 features')
            if any(x < 0 for x in sample):
                raise ValueError(f'Sample {i} contains negative values')
        return v


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint"""
    predictions: List[PredictionResponse]
    total_samples: int
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": 1,
                        "prediction_label": "Malignant",
                        "malignancy_probability": 0.92,
                        "benign_probability": 0.08,
                        "risk_level": "High Risk",
                        "confidence": 0.92
                    }
                ],
                "total_samples": 1
            }
        }


class HealthResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    version: str


class FeatureNamesResponse(BaseModel):
    """Response schema for feature names endpoint"""
    features: List[str]
    total_features: int