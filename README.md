# Sigray Machine Learning Platform

A comprehensive 3D image enhancement system for TIFF images using U-Net deep learning architecture. The platform provides both training and inference pipelines with automatic memory management, GPU acceleration, and CPU fallback support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](https://github.com/sigray/ml-platform)

> **Sigray Machine Learning Platform** - Advanced 3D image enhancement for scientific and industrial applications

## Features

### Training Pipeline
- **Data Preparation**: Automatic patch extraction and augmentation
- **Model Training**: U-Net fine-tuning with configurable loss functions
- **Progress Monitoring**: TensorBoard integration and metrics tracking
- **Checkpointing**: Automatic model saving and recovery
- **Early Stopping**: Prevents overfitting with configurable patience
- **Error Handling**: Comprehensive error recovery and logging

### Inference Pipeline
- **File Processing**: Direct TIFF file enhancement
- **Array Processing**: In-memory numpy array processing
- **Memory Management**: Automatic chunking for large volumes
- **Quality Metrics**: PSNR, MSE, MAE calculation
- **Progress Monitoring**: Real-time progress callbacks
- **Batch Processing**: Efficient multi-file processing

### Hardware Support
- **GPU Acceleration**: Automatic CUDA/MPS detection
- **CPU Fallback**: Seamless fallback when GPU unavailable
- **Memory Optimization**: Dynamic memory management
- **Device Flexibility**: Runtime device switching

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- TIFF file support

### Basic Installation
```bash
# Clone repository
git clone https://github.com/sigray/ml-platform.git
cd ml-platform

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Development Installation
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[dev,tensorboard,monitoring]"
```

## Quick Start

### Command Line Usage

#### Training
```bash
# Basic training
3d-enhance-train --input-dir data/inputs --target-dir data/targets --output-dir training_output

# Advanced training with GPU
3d-enhance-train --input-dir data/inputs --target-dir data/targets --output-dir training_output \
  --epochs 100 --batch-size 16 --device cuda --tensorboard
```

#### Inference
```bash
# Single file enhancement
3d-enhance-infer --model trained_model.pth --input image.tif --output enhanced.tif

# Batch processing
3d-enhance-infer --model trained_model.pth --input-dir inputs/ --output-dir outputs/ \
  --calculate-metrics --save-metrics metrics.json
```

### Programmatic Usage

#### Training
```python
from src.training.training_manager import TrainingManager
from src.core.config import TrainingConfig

# Configure training
config = TrainingConfig(
    epochs=50,
    batch_size=16,
    learning_rate=1e-4,
    device='auto'
)

# Initialize training manager
trainer = TrainingManager(config, output_dir="training_output")

# Prepare data and train
train_loader, val_loader = trainer.prepare_training_data(input_paths, target_paths)
best_model = trainer.fine_tune_model("pretrained_model.pth", train_loader, val_loader)
```

#### Inference
```python
from src.inference.api import ImageEnhancementAPI
from src.core.config import InferenceConfig

# Configure inference
config = InferenceConfig(
    patch_size=(256, 256),
    batch_size=32,
    device='auto',
    enable_quality_metrics=True
)

# Initialize API and load model
api = ImageEnhancementAPI(config)
api.load_model("trained_model.pth")

# Enhance image
result = api.enhance_3d_tiff("input.tif", "enhanced.tif")

if result.success:
    print(f"Enhancement completed in {result.processing_time:.2f}s")
    print(f"PSNR: {result.quality_metrics['psnr']:.2f}")
```

## Architecture

### System Components

```
3D Image Enhancement System
├── Core Components
│   ├── Data Models (SliceStack, Patch2D, TrainingPair2D)
│   ├── TIFF Handler (Loading, saving, validation)
│   ├── Patch Processor (Extraction, reconstruction)
│   └── Configuration (TrainingConfig, InferenceConfig)
├── Training Pipeline
│   ├── Data Preparation (Loading, augmentation)
│   ├── Training Manager (Fine-tuning, monitoring)
│   └── Model Management (Loading, validation, export)
├── Inference Pipeline
│   ├── Image Processor (Volume handling, preprocessing)
│   ├── Enhancement Processor (Model application)
│   └── API (High-level interface)
├── Command Line Interfaces
│   ├── Training CLI (3d-enhance-train)
│   └── Inference CLI (3d-enhance-infer)
└── Testing & Utilities
    ├── Synthetic Data Generation
    ├── Test Fixtures
    └── Error Handling
```

### Data Flow

1. **Training Flow**:
   ```
   TIFF Files → Patch Extraction → Data Augmentation → U-Net Training → Model Export
   ```

2. **Inference Flow**:
   ```
   Input TIFF → Preprocessing → Patch Processing → U-Net Enhancement → Reconstruction → Output TIFF
   ```

## Configuration

### Training Configuration
```json
{
  "patch_size": [256, 256],
  "overlap": 32,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "epochs": 100,
  "validation_split": 0.2,
  "optimizer": "adam",
  "loss_function": "mse",
  "device": "auto",
  "early_stopping_patience": 15,
  "use_augmentation": true
}
```

### Inference Configuration
```json
{
  "patch_size": [256, 256],
  "overlap": 32,
  "batch_size": 32,
  "device": "auto",
  "memory_limit_gb": 8.0,
  "enable_quality_metrics": true,
  "preserve_metadata": true
}
```

## Performance Guidelines

### Memory Requirements
- **Small volumes** (5-10 slices, 1024×1024): ~500MB RAM
- **Medium volumes** (10-20 slices, 2048×2048): ~2GB RAM
- **Large volumes** (20+ slices, 4096×4096): ~8GB+ RAM

### Processing Speed (approximate)
- **CPU**: 10-50 patches/second
- **GPU (CUDA)**: 100-500 patches/second
- **Apple Silicon (MPS)**: 50-200 patches/second

### Optimization Tips
1. **Use GPU** when available for 5-10x speedup
2. **Adjust batch size** based on available memory
3. **Use appropriate patch size**: larger patches = better quality, more memory
4. **Enable chunking** for very large volumes
5. **Disable augmentation** during inference for speed

## Integration Guide

### Existing Codebase Integration

#### Simple Integration
```python
from src.inference.api import ImageEnhancementAPI

# Initialize once
api = ImageEnhancementAPI()
api.load_model("your_model.pth")

# Use in your workflow
def enhance_image(input_path, output_path):
    result = api.enhance_3d_tiff(input_path, output_path)
    return result.success
```

#### Service Integration
```python
from src.inference.api import ImageEnhancementAPI
from src.core.config import InferenceConfig

class YourImageService:
    def __init__(self):
        config = InferenceConfig(device='auto', memory_limit_gb=4.0)
        self.enhancer = ImageEnhancementAPI(config)
        self.enhancer.load_model("path/to/model.pth")
    
    def process_image(self, image_data):
        result = self.enhancer.enhance_3d_array(image_data)
        return result.quality_metrics['enhanced_array'] if result.success else None
```

### Error Handling Best Practices
```python
from src.core.error_handling import ErrorHandler, with_error_handling

error_handler = ErrorHandler("logs/errors.log")

@with_error_handling(error_handler, "image_processing", recoverable=True)
def process_image(image_path):
    # Your processing code here
    pass
```

## Testing

### Run All Tests
```bash
# Run complete test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_training_*.py  # Training tests
pytest tests/test_inference_*.py  # Inference tests
pytest tests/test_end_to_end.py   # Integration tests
```

### Generate Test Data
```python
from src.testing.test_fixtures import TestDataFixtures

with TestDataFixtures() as fixtures:
    # Create synthetic dataset
    dataset = fixtures.create_test_dataset(num_pairs=10)
    
    # Get test volumes
    small_volume = fixtures.get_small_test_volume()
    medium_volume = fixtures.get_medium_test_volume()
```

## Troubleshooting

### Common Issues

#### GPU Out of Memory
```bash
# Reduce batch size
3d-enhance-infer --model model.pth --input image.tif --output enhanced.tif --batch-size 4

# Use CPU instead
3d-enhance-infer --model model.pth --input image.tif --output enhanced.tif --device cpu
```

#### Large File Processing
```bash
# Reduce memory limit to enable chunking
3d-enhance-infer --model model.pth --input large.tif --output enhanced.tif --memory-limit 2.0

# Use smaller patches
3d-enhance-infer --model model.pth --input large.tif --output enhanced.tif --patch-size 128 128
```

#### Model Compatibility Issues
- Ensure model was trained with compatible patch size
- Check model architecture matches expected U-Net structure
- Verify model file is not corrupted

### Performance Optimization

#### For Training
- Use GPU when available (`--device cuda`)
- Increase batch size if memory allows
- Enable data augmentation for better generalization
- Use mixed precision training (future feature)

#### For Inference
- Use GPU for faster processing
- Optimize batch size for your hardware
- Use appropriate patch size for quality/speed tradeoff
- Enable chunking for large volumes

## API Reference

### Core Classes
- `ImageEnhancementAPI`: High-level inference interface
- `TrainingManager`: Complete training pipeline management
- `SliceStack`: 3D volume data structure
- `TrainingConfig`: Training configuration
- `InferenceConfig`: Inference configuration

### Key Methods
- `api.enhance_3d_tiff()`: Enhance TIFF file
- `api.enhance_3d_array()`: Enhance numpy array
- `trainer.fine_tune_model()`: Train/fine-tune model
- `trainer.validate_model()`: Validate trained model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the examples in the `examples/` directory
- Run tests to verify your installation
- Check logs for detailed error information