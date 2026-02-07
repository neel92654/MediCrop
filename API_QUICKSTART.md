# Backend API - Quick Reference for Frontend Dev

## API Base URL
```
http://localhost:5000
```

## Main Prediction Endpoint

**POST `/predict`**

### JavaScript Example
```javascript
const formData = new FormData();
formData.append('image', imageFile);  // imageFile from <input type="file">

fetch('http://localhost:5000/predict', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  if (data.success) {
    // Use these fields:
    console.log(data.prediction.class);          // "Tomato Early Blight"  
    console.log(data.prediction.confidence_percent); // "95.34%"
    console.log(data.prediction.severity);       // "Mild/Minor"
    console.log(data.prediction.category);       // "plant_disease" or "pest"
  }
});
```

### Response Format
```json
{
  "success": true,
  "prediction": {
    "class": "Tomato Early Blight",
    "confidence_percent": "95.34%",
    "severity": "Mild/Minor",
    "category": "plant_disease"
  },
  "top_5_predictions": [...],
  "timestamp": "2026-02-07T09:25:00"
}
```

## Other Endpoints

- `GET /health` - Check if API is running
- `GET /info` - Get model details
- `GET /classes` - List all 50 classes

## Start API Server
```bash
cd plant_health_detection
../.venv/bin/python app.py
```

## Test API
```bash
../.venv/bin/python test_api.py path/to/image.jpg
```

That's it! Full docs in `API_DOCUMENTATION.md`
