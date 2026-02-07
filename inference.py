"""
Example inference script for demonstrating model usage
Shows how to use the trained model for predictions
"""

import sys
import os
from utils import load_model, load_class_mapping, predict_single_image, display_prediction_results
import config


def main():
    """Main inference function"""
    
    # Check if image path provided
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        print("\nExample:")
        print("  python inference.py ../plants/New\\ Plant\\ Diseases\\ Dataset\\(Augmented\\)/valid/Tomato___Early_blight/image001.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        sys.exit(1)
    
    print("\n🔄 Loading model and class mapping...")
    
    # Load model and mapping
    try:
        model = load_model(config.MODEL_SAVE_PATH)
        class_mapping = load_class_mapping(config.CLASS_MAPPING_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n💡 Hint: Have you trained the model yet? Run: python train.py")
        sys.exit(1)
    
    print("✅ Model loaded successfully")
    
    # Make prediction
    print(f"\n🔍 Analyzing image: {os.path.basename(image_path)}")
    
    results = predict_single_image(model, image_path, class_mapping)
    
    # Display results
    display_prediction_results(image_path, results)
    
    # Additional recommendations based on severity
    print("💡 Recommendations:")
    if results['severity'] == "Mild/Minor":
        print("   ✓ Early detection - Monitor the plant regularly")
        print("   ✓ Apply preventive measures if available")
    elif results['severity'] == "Moderate":
        print("   ⚠ Take action - Apply appropriate treatment")
        print("   ⚠ Isolate affected plants if possible")
    else:
        print("   🚨 Seek expert consultation")
        print("   🚨 Consider removing severely affected plants")
        print("   🚨 Low confidence - verify diagnosis manually")
    
    print()


if __name__ == '__main__':
    main()
