"""
Example client for interacting with the BreastCareAI API

This script demonstrates how to use the API programmatically.
"""
import requests
import json
from typing import List, Dict


class BreastCareAIClient:
    """Client for BreastCareAI API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """
        Check API health status
        
        Returns:
            Health status dictionary
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of required feature names
        
        Returns:
            List of feature names
        """
        url = f"{self.base_url}/features"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['features']
    
    def predict(self, features: List[float]) -> Dict:
        """
        Make a single prediction
        
        Args:
            features: List of 30 feature values
            
        Returns:
            Prediction result dictionary
        """
        url = f"{self.base_url}/predict"
        payload = {"features": features}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, samples: List[List[float]]) -> Dict:
        """
        Make predictions for multiple samples
        
        Args:
            samples: List of feature lists
            
        Returns:
            Batch prediction results
        """
        url = f"{self.base_url}/batch-predict"
        payload = {"samples": samples}
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    
    def predict_with_interpretation(self, features: List[float]) -> None:
        """
        Make a prediction and print human-readable interpretation
        
        Args:
            features: List of 30 feature values
        """
        result = self.predict(features)
        
        print("\n" + "="*60)
        print("BREAST CANCER DETECTION RESULT")
        print("="*60)
        
        print(f"\n🔍 Prediction: {result['prediction_label']}")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        
        print(f"\n📈 Probabilities:")
        print(f"   • Malignant: {result['malignancy_probability']:.2%}")
        print(f"   • Benign: {result['benign_probability']:.2%}")
        
        print("\n" + "="*60)
        
        # Interpretation
        if result['risk_level'] == "High Risk":
            print("⚠️  HIGH RISK detected. Immediate medical consultation recommended.")
        elif result['risk_level'] == "Medium Risk":
            print("⚠️  MEDIUM RISK detected. Medical follow-up advised.")
        else:
            print("✅ LOW RISK detected. Continue regular monitoring.")
        
        print("="*60 + "\n")


def example_usage():
    """Example usage of the API client"""
    
    # Initialize client
    client = BreastCareAIClient(base_url="http://localhost:8000")
    
    print("Testing BreastCareAI API...\n")
    
    # 1. Health check
    print("1. Checking API health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Version: {health['version']}\n")
    
    # 2. Get feature names
    print("2. Getting feature names...")
    features = client.get_feature_names()
    print(f"   Total features required: {len(features)}")
    print(f"   First 5 features: {features[:5]}\n")
    
    # 3. Single prediction (Malignant example)
    print("3. Making single prediction (Malignant sample)...")
    malignant_features = [
        17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
        0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
        0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
        184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
    client.predict_with_interpretation(malignant_features)
    
    # 4. Single prediction (Benign example)
    print("4. Making single prediction (Benign sample)...")
    benign_features = [
        11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115,
        0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47, 0.005857,
        0.009758, 0.01168, 0.007445, 0.02406, 0.001769, 12.98, 25.72,
        82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563
    ]
    client.predict_with_interpretation(benign_features)
    
    # 5. Batch prediction
    print("5. Making batch prediction...")
    samples = [malignant_features, benign_features]
    batch_result = client.batch_predict(samples)
    print(f"   Processed {batch_result['total_samples']} samples")
    for i, pred in enumerate(batch_result['predictions'], 1):
        print(f"   Sample {i}: {pred['prediction_label']} ({pred['risk_level']})")
    
    print("\n✅ All tests completed successfully!")


if __name__ == "__main__":
    try:
        example_usage()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure the server is running:")
        print("   uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error: {str(e)}")