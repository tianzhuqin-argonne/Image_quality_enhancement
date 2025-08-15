# Local Workstation Setup Guide

This guide will help you download and run the Sigray Machine Learning Platform on your local workstation for training and inference.

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **Python**: 3.8 or higher

### Recommended for Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/4070 or better)
- **RAM**: 32GB+
- **CPU**: 8+ cores
- **Storage**: SSD with 50GB+ free space

### GPU Support
- **NVIDIA**: CUDA 11.0+ with compatible drivers
- **Apple Silicon**: M1/M2 Macs with Metal Performance Shaders
- **CPU Only**: Supported but slower training

## 📥 Step 1: Download the Repository

### Option A: Using Git (Recommended)
```bash
# Clone the repository
git clone https://github.com/tianzhuqin-argonne/Image_quality_enhancement.git

# Navigate to the directory
cd Image_quality_enhancement
```

### Option B: Download ZIP
1. Go to https://github.com/tianzhuqin-argonne/Image_quality_enhancement
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal/command prompt in the extracted folder

## 🐍 Step 2: Set Up Python Environment

### Option A: Using Conda (Recommended)
```bash
# Create conda environment
conda create -n sigray-ml python=3.9
conda activate sigray-ml

# Install PyTorch (choose based on your system)
# For NVIDIA GPU:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For Apple Silicon Mac:
conda install pytorch torchvision torchaudio -c pytorch

# For CPU only:
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Option B: Using pip + venv
```bash
# Create virtual environment
python -m venv sigray-ml-env

# Activate environment
# On Windows:
sigray-ml-env\Scripts\activate
# On macOS/Linux:
source sigray-ml-env/bin/activate

# Install PyTorch (visit pytorch.org for your specific command)
# Example for NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Example for CPU only:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 📦 Step 3: Install the Platform

```bash
# Install the Sigray ML Platform
pip install -e .

# Verify installation
python -c "from src.inference.api import ImageEnhancementAPI; print('✅ Installation successful!')"
```

## 🧪 Step 4: Run Basic Tests

```bash
# Run a quick test to verify everything works
python -m pytest tests/test_data_models.py -v

# Check GPU availability (if you have one)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Step 5: Run Training Example

### Quick Training Example (5-10 minutes)
```bash
# Run the basic training example
python examples/basic_training_example.py
```

This will:
- Create synthetic training data
- Train a small U-Net model for a few epochs
- Save the trained model
- Show training progress and metrics

### Custom Training with Your Data
```bash
# Prepare your data structure:
# data/
# ├── inputs/
# │   ├── input_001.tif
# │   ├── input_002.tif
# │   └── ...
# └── targets/
#     ├── target_001.tif
#     ├── target_002.tif
#     └── ...

# Run training with CLI
python -m src.cli.train_cli \
  --input-dir data/inputs \
  --target-dir data/targets \
  --output-dir training_output \
  --epochs 50 \
  --batch-size 8 \
  --device auto
```

## 🔮 Step 6: Run Inference Example

### Quick Inference Example
```bash
# Run the basic inference example
python examples/basic_inference_example.py
```

This will:
- Load a trained model
- Create test image data
- Enhance the images
- Show quality metrics and processing time

### Inference with Your Images
```bash
# Single file enhancement
python -m src.cli.inference_cli \
  --model path/to/your/model.pth \
  --input your_image.tif \
  --output enhanced_image.tif \
  --calculate-metrics

# Batch processing
python -m src.cli.inference_cli \
  --model path/to/your/model.pth \
  --input-dir input_images/ \
  --output-dir enhanced_images/ \
  --batch-size 16 \
  --calculate-metrics
```

## 📊 Step 7: Monitor Training (Optional)

### Install TensorBoard
```bash
pip install tensorboard
```

### Start TensorBoard
```bash
# Start TensorBoard (run this in a separate terminal)
tensorboard --logdir training_output/logs

# Open browser to: http://localhost:6006
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. GPU Out of Memory
```bash
# Reduce batch size
python -m src.cli.train_cli --batch-size 4  # Instead of 16

# Or use CPU
python -m src.cli.train_cli --device cpu
```

#### 2. Import Errors
```bash
# Make sure you're in the right directory
pwd  # Should show the Image_quality_enhancement directory

# Reinstall the package
pip install -e .
```

#### 3. CUDA Issues (NVIDIA GPU)
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
# Then reinstall with correct CUDA version from pytorch.org
```

#### 4. Memory Issues
```bash
# Reduce memory usage
python -m src.cli.inference_cli --memory-limit 4.0  # 4GB limit
python -m src.cli.train_cli --num-workers 2  # Reduce data loading workers
```

#### 5. Slow Performance
```bash
# Check if GPU is being used
python -c "
import torch
from src.inference.api import ImageEnhancementAPI
api = ImageEnhancementAPI()
print(f'Device: {api.image_processor.device}')
"

# Force GPU usage (if available)
python -m src.cli.inference_cli --device cuda
```

## 📁 Understanding the Output

### Training Output Structure
```
training_output/
├── checkpoints/
│   ├── best_model.pth          # Best performing model
│   ├── checkpoint_epoch_10.pth # Regular checkpoints
│   └── final_model.pth         # Final model
├── logs/
│   └── tensorboard_logs/       # TensorBoard logs
├── data_statistics.json        # Training data statistics
└── training_state.json         # Training progress info
```

### Inference Output
```
enhanced_images/
├── enhanced_image_001.tif      # Enhanced images
├── enhanced_image_002.tif
└── metrics.json                # Quality metrics (if enabled)
```

## 🎯 Example Workflows

### Workflow 1: Quick Test Run
```bash
# 1. Download and setup (5 minutes)
git clone https://github.com/tianzhuqin-argonne/Image_quality_enhancement.git
cd Image_quality_enhancement
conda create -n sigray-ml python=3.9
conda activate sigray-ml
conda install pytorch torchvision torchaudio -c pytorch
pip install -e .

# 2. Run examples (10 minutes)
python examples/basic_training_example.py
python examples/basic_inference_example.py
```

### Workflow 2: Train with Your Data
```bash
# 1. Prepare your data
mkdir -p data/inputs data/targets
# Copy your TIFF files to these directories

# 2. Start training
python -m src.cli.train_cli \
  --input-dir data/inputs \
  --target-dir data/targets \
  --output-dir my_training \
  --epochs 100 \
  --batch-size 16 \
  --device auto \
  --tensorboard

# 3. Monitor progress
tensorboard --logdir my_training/logs

# 4. Use trained model for inference
python -m src.cli.inference_cli \
  --model my_training/checkpoints/best_model.pth \
  --input new_image.tif \
  --output enhanced_image.tif
```

### Workflow 3: Batch Processing
```bash
# Process many images at once
python -m src.cli.inference_cli \
  --model trained_model.pth \
  --input-dir batch_input/ \
  --output-dir batch_output/ \
  --batch-size 32 \
  --calculate-metrics \
  --save-metrics batch_metrics.json
```

## 📈 Performance Optimization

### For Training
```bash
# Use mixed precision (if supported)
python -m src.cli.train_cli --use-mixed-precision

# Increase batch size (if you have enough memory)
python -m src.cli.train_cli --batch-size 32

# Use more data loading workers
python -m src.cli.train_cli --num-workers 8
```

### For Inference
```bash
# Increase batch size for faster processing
python -m src.cli.inference_cli --batch-size 64

# Use larger patches for better quality (if you have memory)
python -m src.cli.inference_cli --patch-size 512 512

# Reduce overlap for faster processing (slight quality trade-off)
python -m src.cli.inference_cli --overlap 16
```

## 🔧 Advanced Configuration

### Custom Configuration Files
```bash
# Create config file
cat > my_config.json << EOF
{
  "patch_size": [256, 256],
  "overlap": 32,
  "batch_size": 16,
  "device": "auto",
  "memory_limit_gb": 8.0,
  "enable_quality_metrics": true
}
EOF

# Use config file
python -m src.cli.inference_cli --config-file my_config.json --input image.tif --output enhanced.tif
```

### Environment Variables
```bash
# Set default device
export SIGRAY_ML_DEVICE=cuda

# Set memory limit
export SIGRAY_ML_MEMORY_LIMIT=16.0

# Enable debug logging
export SIGRAY_ML_LOG_LEVEL=DEBUG
```

## 📚 Next Steps

1. **Experiment with Parameters**: Try different patch sizes, batch sizes, and learning rates
2. **Use Your Own Data**: Replace synthetic data with your real TIFF images
3. **Monitor Training**: Use TensorBoard to track training progress
4. **Scale Up**: Move to cloud platforms like AWS SageMaker for larger datasets
5. **Customize Models**: Modify the U-Net architecture for your specific needs

## 🆘 Getting Help

- **Documentation**: Check the main README.md for detailed API documentation
- **Examples**: Look in the `examples/` directory for more use cases
- **Issues**: Report bugs at https://github.com/tianzhuqin-argonne/Image_quality_enhancement/issues
- **Tests**: Run `pytest tests/` to verify your installation

Happy enhancing! 🚀