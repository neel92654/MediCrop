#!/bin/bash
# Installation script for macOS (Apple Silicon)

echo "🍎 Setting up Plant Health Detection System on macOS..."

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install numpy first (TensorFlow 2.13.0 requires numpy<=1.24.3)
echo "📐 Installing compatible numpy version..."
pip install "numpy>=1.22,<=1.24.3"

# Install TensorFlow for macOS (Apple Silicon)
echo "🧠 Installing TensorFlow for Apple Silicon..."
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.0

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install pandas>=2.0.0
pip install opencv-python>=4.8.0
pip install Pillow>=10.0.0
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
pip install scikit-learn>=1.3.0

echo "✅ Installation complete!"
echo ""
echo "🧪 Verifying installation..."
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import tensorflow as tf; print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"

echo ""
echo "🚀 Ready to train! Run:"
echo "   python train.py --epochs 2"
