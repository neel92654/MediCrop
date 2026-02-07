# Plant Health Detection API Documentation

## Quick Start

### 1. Install API Dependencies
```bash
cd "/Users/neelpatel/Documents/IIT gandhinagar/plant_health_detection"
../.venv/bin/pip install -r requirements-api.txt
```

### 2. Start the API Server
```bash
../.venv/bin/python app.py
```

Server will start at: `http://localhost:5000`

---

## API Endpoints

### GET `/` - API Home
Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Plant Health Detection API",
  "version": "1.0.0",
  "status": "running",
  "model_loaded": true,
  "total_classes": 50,
  "endpoints": {
    "health": "/health",
    "predict": "/predict (POST)",
    "classes": "/classes",
    "info": "/info"
  }
}
```

---

### GET `/health` - Health Check
Check if API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-02-07T09:25:00"
}
```

---

### GET `/classes` - Get All Classes
Returns list of all 50 classes (38 plant diseases + 12 pests).

**Response:**
```json
{
  "total_classes": 50,
  "classes": [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "pest_ants",
    "pest_bees",
    ...
  ]
}
```

---

### GET `/info` - Model Information
Returns detailed model information.

**Response:**
```json
{
  "model": {
    "name": "Plant Health Detection - ResNet50",
    "version": "1.0.0",
    "architecture": "ResNet50 Transfer Learning",
    "total_classes": 50,
    "input_size": "224x224",
    "severity_thresholds": {
      "high_confidence": 0.8,
      "medium_confidence": 0.5
    }
  }
}
```

---

### POST `/predict` - Predict Disease/Pest ⭐ **Main Endpoint**

Upload an image to get disease/pest prediction.

**Request:**
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body:** Image file with key `image`
- **Supported formats:** JPG, JPEG, PNG, BMP
- **Max file size:** 16MB

**Example Request (JavaScript):**
```javascript
const formData = new FormData();
formData.append('image', imageFile);

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

**Example Request (Python):**
```python
import requests

with open('leaf_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    
print(response.json())
```

**Example Request (cURL):**
```bash
curl -X POST -F "image=@leaf_image.jpg" http://localhost:5000/predict
```

**Success Response (200):**
```json
{
  "success": true,
  "prediction": {
    "class": "Tomato Early Blight",
    "raw_class": "Tomato___Early_blight",
    "category": "plant_disease",
    "confidence": 0.9534,
    "confidence_percent": "95.34%",
    "severity": "Mild/Minor"
  },
  "top_5_predictions": [
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.9534,
      "confidence_percent": "95.34%"
    },
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.0321,
      "confidence_percent": "3.21%"
    },
    ...
  ],
  "timestamp": "2026-02-07T09:25:00"
}
```

**Error Response (400 - No Image):**
```json
{
  "success": false,
  "error": "No image file provided. Please upload an image with key 'image'."
}
```

**Error Response (400 - Invalid File Type):**
```json
{
  "success": false,
  "error": "Invalid file type. Allowed types: png, jpg, jpeg, bmp"
}
```

**Error Response (413 - File Too Large):**
```json
{
  "success": false,
  "error": "File too large. Maximum size: 16MB"
}
```

---

## Response Field Explanations

### `prediction` Object
- **`class`**: Human-readable class name (e.g., "Tomato Early Blight")
- **`raw_class`**: Original class name from model (e.g., "Tomato___Early_blight")
- **`category`**: Either "plant_disease" or "pest"
- **`confidence`**: Confidence score (0.0 - 1.0)
- **`confidence_percent`**: Confidence as percentage string
- **`severity`**: Severity estimation based on confidence:
  - **"Mild/Minor"**: Confidence > 80%
  - **"Moderate"**: Confidence 50-80%
  - **"Severe/Uncertain"**: Confidence < 50%

### `top_5_predictions` Array
List of top 5 predictions with class names and confidence scores.

---

## Testing the API

### Method 1: Using Test Script
```bash
# Install requests library first
../.venv/bin/pip install requests

# Test without prediction
../.venv/bin/python test_api.py

# Test with prediction
../.venv/bin/python test_api.py path/to/leaf_image.jpg
```

### Method 2: Using cURL
```bash
# Health check
curl http://localhost:5000/health

# Get model info
curl http://localhost:5000/info

# Predict
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

### Method 3: Using Postman
1. Open Postman
2. Create POST request to `http://localhost:5000/predict`
3. Go to Body → form-data
4. Add key `image` with type `File`
5. Select an image file
6. Send request

---

## Frontend Integration Example

### React/Next.js
```javascript
async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  try {
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.success) {
      console.log('Predicted:', data.prediction.class);
      console.log('Confidence:', data.prediction.confidence_percent);
      console.log('Severity:', data.prediction.severity);
      return data;
    } else {
      console.error('Error:', data.error);
      return null;
    }
  } catch (error) {
    console.error('Network error:', error);
    return null;
  }
}

// Usage
const fileInput = document.getElementById('imageInput');
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const result = await predictDisease(file);
  // Update UI with result
});
```

### HTML + Vanilla JavaScript
```html
<!DOCTYPE html>
<html>
<head>
    <title>Plant Health Detection</title>
</head>
<body>
    <h1>Upload Leaf Image</h1>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="uploadAndPredict()">Predict</button>
    <div id="results"></div>

    <script>
        async function uploadAndPredict() {
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                const response = await fetch('http://localhost:5000/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('results').innerHTML = `
                        <h2>Prediction Results</h2>
                        <p><strong>Disease/Pest:</strong> ${data.prediction.class}</p>
                        <p><strong>Confidence:</strong> ${data.prediction.confidence_percent}</p>
                        <p><strong>Severity:</strong> ${data.prediction.severity}</p>
                    `;
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Network error: ' + error.message);
            }
        }
    </script>
</body>
</html>
```

---

## CORS Configuration

The API has CORS enabled by default, allowing requests from any origin. For production, you may want to restrict this:

```python
# In app.py, modify:
CORS(app, resources={r"/*": {"origins": "https://your-frontend-domain.com"}})
```

---

## Deployment Notes

### Running in Production

1. **Use Gunicorn** (production WSGI server):
```bash
../.venv/bin/pip install gunicorn
../.venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Or use Waitress** (cross-platform):
```bash
../.venv/bin/pip install waitress
../.venv/bin/waitress-serve --port=5000 app:app
```

### Environment Variables
Set these for production:
```bash
export FLASK_ENV=production
export MODEL_PATH=/path/to/model.keras
```

---

## Troubleshooting

### Error: "Model not loaded"
- Make sure you've trained the model first: `python train.py`
- Check that `models/plant_health_model.keras` exists

### Error: "ModuleNotFoundError: No module named 'flask'"
- Install API requirements: `pip install -r requirements-api.txt`

### Error: Port 5000 already in use
- Change port in `app.py`: `app.run(port=5001)`
- Or kill process using port 5000

---

## API Response Time

- **Average**: ~200-500ms per prediction (CPU)
- **With GPU**: ~50-100ms per prediction
- **Depends on**: Image size, server load, hardware

---

## Security Considerations

1. **File Upload Validation**: Only accepts image files (JPG, PNG, BMP)
2. **File Size Limit**: 16MB maximum
3. **CORS**: Enabled for development, restrict in production
4. **Uploads Folder**: Uploaded images are saved to `uploads/` directory

---

## Support

For issues or questions, check:
- API logs in terminal
- `test_api.py` for working examples
- Model must be trained before API can make predictions
