"""
Flask API for Plant Health Detection System
Provides REST endpoints for disease and pest prediction
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from datetime import datetime

# Import custom modules
import config
from preprocessing import preprocess_single_image
from model import estimate_severity

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model and class mapping
model = None
class_mapping = None
class_names = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_mappings():
    """Load trained model and class mappings"""
    global model, class_mapping, class_names
    
    try:
        # Load model
        model_path = config.MODEL_SAVE_PATH
        if not os.path.exists(model_path):
            # Try .keras format as fallback
            model_path = model_path.replace('.h5', '.keras')
        
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        
        # Load class mapping
        with open(config.CLASS_MAPPING_PATH, 'r') as f:
            class_mapping = json.load(f)
        
        # Create class names list (sorted by index)
        class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
        
        print(f"✅ Loaded {len(class_names)} classes")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def predict_image(image_path):
    """
    Predict disease/pest from image
    
    Args:
        image_path: Path to image file
        
    Returns:
        dict: Prediction results
    """
    try:
        # Preprocess image
        img_array = preprocess_single_image(image_path, target_size=config.IMG_SIZE)
        
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get top prediction
        top_idx = np.argmax(predictions[0])
        top_confidence = float(predictions[0][top_idx])
        top_class = class_names[top_idx]
        
        # Get top 5 predictions
        top_5_idx = np.argsort(predictions[0])[-5:][::-1]
        top_5_predictions = [
            {
                'class': class_names[idx],
                'confidence': float(predictions[0][idx]),
                'confidence_percent': f"{float(predictions[0][idx]) * 100:.2f}%"
            }
            for idx in top_5_idx
        ]
        
        # Estimate severity
        severity = estimate_severity(top_confidence)
        
        # Determine category (plant disease vs pest)
        category = "pest" if "pest_" in top_class else "plant_disease"
        
        # Clean class name (remove pest_ prefix if present)
        display_class = top_class.replace("pest_", "").replace("_", " ").title()
        
        return {
            'success': True,
            'prediction': {
                'class': display_class,
                'raw_class': top_class,
                'category': category,
                'confidence': top_confidence,
                'confidence_percent': f"{top_confidence * 100:.2f}%",
                'severity': severity
            },
            'top_5_predictions': top_5_predictions,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# ==================== API ENDPOINTS ====================

@app.route('/', methods=['GET'])
def home():
    """API home page with documentation"""
    return jsonify({
        'service': 'Plant Health Detection API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'total_classes': len(class_names) if class_names else 0,
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'classes': '/classes',
            'info': '/info'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available classes"""
    if class_names is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'total_classes': len(class_names),
        'classes': class_names
    })


@app.route('/info', methods=['GET'])
def get_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model': {
            'name': 'Plant Health Detection - ResNet50',
            'version': '1.0.0',
            'architecture': 'ResNet50 Transfer Learning',
            'total_classes': len(class_names),
            'input_size': f"{config.IMG_HEIGHT}x{config.IMG_WIDTH}",
            'severity_thresholds': config.SEVERITY_THRESHOLDS
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease/pest from uploaded image
    
    Request:
        - Method: POST
        - Content-Type: multipart/form-data
        - Body: image file with key 'image'
        
    Response:
        {
            "success": true,
            "prediction": {
                "class": "Tomato Early Blight",
                "category": "plant_disease",
                "confidence": 0.95,
                "confidence_percent": "95.00%",
                "severity": "Mild/Minor"
            },
            "top_5_predictions": [...],
            "timestamp": "2026-02-07T09:25:00"
        }
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please contact administrator.'
        }), 503
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided. Please upload an image with key "image".'
        }), 400
    
    file = request.files['image']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename.'
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = predict_image(filepath)
        
        # Optionally delete uploaded file after prediction
        # os.remove(filepath)
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found. See / for available endpoints.'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error.'
    }), 500


# ==================== MAIN ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌱 PLANT HEALTH DETECTION API")
    print("="*60)
    
    # Load model on startup
    if load_model_and_mappings():
        print("\n✅ API ready to serve predictions")
        print(f"📊 Total classes: {len(class_names)}")
        print(f"🔧 Upload folder: {UPLOAD_FOLDER}")
        print("\n" + "="*60)
        print("Starting Flask server...")
        print("="*60 + "\n")
        
        # Run Flask app
        app.run(
            host='0.0.0.0',  # Accessible from network
            port=5000,
            debug=False  # Set to False in production
        )
    else:
        print("\n❌ Failed to load model. Please train the model first.")
        print("   Run: python train.py")
