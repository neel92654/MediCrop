"""
Test script for Plant Health Detection API
Demonstrates how to use the API endpoints
"""

import requests
import json
import sys

# API base URL
BASE_URL = "http://localhost:5000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_info():
    """Test info endpoint"""
    print("\n" + "="*60)
    print("Testing /info endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict(image_path):
    """Test prediction endpoint"""
    print("\n" + "="*60)
    print(f"Testing /predict endpoint with image: {image_path}")
    print("="*60)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        result = response.json()
        
        if result.get('success'):
            print("\n✅ Prediction successful!")
            print(f"Predicted Class: {result['prediction']['class']}")
            print(f"Category: {result['prediction']['category']}")
            print(f"Confidence: {result['prediction']['confidence_percent']}")
            print(f"Severity: {result['prediction']['severity']}")
            
            print("\nTop 5 Predictions:")
            for i, pred in enumerate(result['top_5_predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['confidence_percent']}")
        else:
            print(f"\n❌ Prediction failed: {result.get('error')}")
        
        return result.get('success', False)
        
    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("🌱 PLANT HEALTH DETECTION API - TEST SCRIPT")
    print("="*60)
    
    # Test health endpoint
    if not test_health():
        print("\n❌ API is not healthy. Make sure the server is running.")
        print("   Start server with: python app.py")
        return
    
    # Test info endpoint
    test_info()
    
    # Test prediction if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_predict(image_path)
    else:
        print("\n" + "="*60)
        print("📸 To test prediction, run:")
        print("   python test_api.py path/to/image.jpg")
        print("="*60)


if __name__ == '__main__':
    main()
