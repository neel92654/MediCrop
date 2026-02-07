# 🚀 Quick Setup Guide

## Installation Steps

### 1. Navigate to Project Directory
```bash
cd "/Users/neelpatel/Documents/IIT gandhinagar/plant_health_detection"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: This will install TensorFlow 2.x and all required libraries. Installation may take 5-10 minutes depending on your internet connection.

### 3. Verify Dataset
```bash
python data_loader.py
```

**Expected output**: Should show 50 classes (38 plant diseases + 12 pests) with 93,361 total images.

### 4. Test Training (Quick)
```bash
# Run 2 epochs to verify everything works
python train.py --epochs 2
```

### 5. Full Training
```bash
# Train for 30 epochs (recommended)
python train.py

# Or customize:
python train.py --epochs 50 --batch-size 64
```

**Expected time**: 2-4 hours on GPU, 8-12 hours on CPU

### 6. Test Inference
```bash
# After training completes
python inference.py "../plants/New Plant Diseases Dataset(Augmented)/valid/Tomato___Early_blight/0a2f8e1e-80f0-404e-acc6-f9bc78874d79___RS_Erly.B 7772.JPG"
```

## What Gets Created

After training, you'll have:

```
models/
├── plant_health_model.keras      # Main model file (for TensorFlow 2.x)
├── plant_health_model.h5         # Alternative format (compatibility)
├── class_mapping.json            # Class name to index mapping
├── class_indices.json            # Generator class indices
├── training_history.csv          # Training metrics log
├── training_history.png          # Accuracy/loss plots
└── tensorboard_logs/             # TensorBoard visualization files
```

## Troubleshooting

### Issue: TensorFlow Installation Fails
**Solution**: Try installing with specific version:
```bash
pip install tensorflow==2.13.0
```

### Issue: Out of Memory During Training
**Solution**: Reduce batch size in `config.py`:
```python
BATCH_SIZE = 16  # or even 8
```

### Issue: Model Training is Slow
**Solution**: 
- Verify GPU is available: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- If no GPU, consider reducing epochs or using cloud GPU (Google Colab, Kaggle)

## Next Steps

1. ✅ Install dependencies
2. ✅ Validate dataset
3. ✅ Train model
4. ✅ Test inference
5. 🚀 Deploy to Flask/FastAPI (see README.md for examples)
6. 📱 Build frontend interface
7. 🌍 Deploy to cloud (Heroku, AWS, GCP)

## Quick Commands Reference

```bash
# Validate dataset
python data_loader.py

# Quick test (2 epochs)
python train.py --epochs 2

# Full training
python train.py

# Inference on single image
python inference.py path/to/image.jpg

# View TensorBoard
tensorboard --logdir=models/tensorboard_logs
```

## Performance Expectations

- **Validation Accuracy**: Target 85-92%
- **Training Time**: 2-4 hours (GPU) / 8-12 hours (CPU)
- **Inference Speed**: 50-100ms per image
- **Model Size**: ~95MB

Good luck with your hackathon! 🌱🚀
