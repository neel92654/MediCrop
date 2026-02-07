# MediCrop
MediCrop is a web-based application that helps identify crop diseases and pest damage using images of plant leaves. The system uses a computer vision model trained on open agricultural datasets to classify plant health conditions, estimate severity, and provide basic, actionable recommendations.

# Plant Health Detection System

A deep learning-based agricultural health detection system that classifies plant diseases and pest infestations using transfer learning with ResNet50.

## Overview

This system provides:
- **50-class classification**: 38 plant disease classes + 12 pest categories
- **Transfer learning**: Leverages pretrained ResNet50 (ImageNet weights)
- **Severity estimation**: Confidence-based severity assessment (Mild/Moderate/Severe)
- **Ready for deployment**: Exportable model in `.keras` and `.h5` formats for Flask/FastAPI integration

## Dataset Structure

The system expects data in the following structure:

```
IIT gandhinagar/
├── plants/
│   └── New Plant Diseases Dataset(Augmented)/
│       ├── train/          # ~67,848 images, 38 classes
│       └── valid/          # ~16,973 images, 38 classes
└── pests/
    ├── ants/
    ├── bees/
    ├── beetle/
    └── ... (12 pest classes, ~5,494 images total)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Dataset

```bash
python data_loader.py
```

### 3. Train Model

```bash
# Full training (30 epochs)
python train.py

# Quick test (2 epochs)
python train.py --epochs 2

# Custom configuration
python train.py --epochs 50 --batch-size 64
```

### 4. Make Predictions

```python
from utils import load_model, load_class_mapping, predict_single_image

# Load trained model
model = load_model('models/plant_health_model.keras')
class_mapping = load_class_mapping('models/class_mapping.json')

# Predict on new image
results = predict_single_image(model, 'path/to/image.jpg', class_mapping)

print(f"Predicted: {results['predicted_class']}")
print(f"Confidence: {results['confidence']:.2%}")
print(f"Severity: {results['severity']}")
```

## Architecture

```
Input (224×224×3)
    ↓
ResNet50 Base (frozen, pretrained on ImageNet)
    ↓
Global Average Pooling
    ↓
Dense(512, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(50, Softmax) ← Output
```

**Key Components:**
- **Base Model**: ResNet50 (25M+ parameters, frozen during initial training)
- **Custom Head**: Dense layer (512 units) + Dropout + Output layer
- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Top-5 Accuracy

## Data Augmentation

Training augmentation includes:
- Rotation: ±20 degrees
- Width/Height shift: ±10%
- Brightness: ±20%
- Zoom: 0.8-1.2x
- Horizontal flip
- Normalization: [0, 1] rescaling

## Severity Estimation

Severity is estimated based on prediction confidence:

| Confidence | Severity Level | Meaning |
|-----------|----------------|---------|
| > 0.8 | **Mild/Minor** | High confidence, early detection |
| 0.5 - 0.8 | **Moderate** | Medium confidence, take action |
| < 0.5 | **Severe/Uncertain** | Low confidence, needs expert review |

## Project Structure

```
plant_health_detection/
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Dataset loading and validation
├── preprocessing.py       # Image preprocessing and augmentation
├── model.py              # Model architecture and severity estimation
├── utils.py              # Helper functions and visualization
├── train.py              # Main training script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── models/              # Saved models and outputs
    ├── plant_health_model.keras
    ├── plant_health_model.h5
    ├── class_mapping.json
    ├── class_indices.json
    ├── training_history.csv
    ├── training_history.png
    └── tensorboard_logs/
```

## Configuration

Edit `config.py` to customize:
- Image size and batch size
- Learning rate and epochs
- Data augmentation parameters
- Callback settings
- Model architecture (Dense units, dropout rate)
- Severity thresholds

## Training Features

**Callbacks:**
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if no improvement (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate when loss plateaus
- **CSVLogger**: Logs metrics to CSV
- **TensorBoard**: Visualization with TensorBoard

**Outputs:**
- Trained model (`.keras` and `.h5`)
- Training history plots
- Class mappings (JSON)
- Training logs (CSV)

## Testing Individual Modules

## Expected Performance

Based on PlantVillage benchmarks:
- **Training Accuracy**: 95-98% (with sufficient epochs)
- **Validation Accuracy**: 85-92% (target >85%)
- **Training Time**: ~2-4 hours on GPU (30 epochs)
- **Inference Time**: ~50-100ms per image

## Troubleshooting

**Issue: Out of Memory**
- Reduce `BATCH_SIZE` in `config.py` (try 16 or 8)
- Reduce image size (not recommended, may affect accuracy)

**Issue: Low Accuracy**
- Increase training epochs
- Enable fine-tuning (unfreeze base model layers)
- Verify dataset quality and balance

**Issue: Overfitting**
- Increase dropout rate
- Add more augmentation
- Collect more training data

## Technologies Used

- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **ResNet50**: Pretrained CNN architecture
- **NumPy & Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **OpenCV & Pillow**: Image processing
- **scikit-learn**: Metrics and evaluation

## Use Cases

- Agricultural crop monitoring systems
- Mobile applications for farmers
- Automated greenhouse monitoring
- Educational tools for agricultural students
- Research in plant pathology

---

**Built for sustainable agriculture**

