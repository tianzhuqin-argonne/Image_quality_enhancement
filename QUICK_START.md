# 🚀 Quick Start Guide

Get up and running with the Sigray ML Platform in under 10 minutes!

## ⚡ Super Quick Setup (5 minutes)

```bash
# 1. Clone the repository
git clone https://github.com/tianzhuqin-argonne/Image_quality_enhancement.git
cd Image_quality_enhancement

# 2. Create environment and install
conda create -n sigray-ml python=3.9 -y
conda activate sigray-ml
conda install pytorch torchvision torchaudio -c pytorch -y
pip install -e .

# 3. Test installation
python -c "from src.inference.api import ImageEnhancementAPI; print('✅ Ready to go!')"
```

## 🎯 Run Your First Example (2 minutes)

### Training Example
```bash
python examples/basic_training_example.py
```
**What it does**: Creates synthetic data, trains a small model, shows progress

### Inference Example  
```bash
python examples/basic_inference_example.py
```
**What it does**: Loads a model, enhances test images, shows quality metrics

## 🔧 Use Your Own Data

### Training with Your TIFF Files
```bash
# 1. Organize your data
mkdir -p my_data/inputs my_data/targets
# Copy your input TIFF files to my_data/inputs/
# Copy your target TIFF files to my_data/targets/

# 2. Start training
python -m src.cli.train_cli \
  --input-dir my_data/inputs \
  --target-dir my_data/targets \
  --output-dir my_training \
  --epochs 50 \
  --batch-size 8 \
  --device auto
```

### Enhance Your Images
```bash
# Single image
python -m src.cli.inference_cli \
  --model my_training/checkpoints/best_model.pth \
  --input your_image.tif \
  --output enhanced_image.tif

# Multiple images
python -m src.cli.inference_cli \
  --model my_training/checkpoints/best_model.pth \
  --input-dir input_folder/ \
  --output-dir enhanced_folder/
```

## 🎛️ Key Parameters

### Training
- `--epochs 100`: Number of training epochs
- `--batch-size 16`: Batch size (reduce if out of memory)
- `--device cuda`: Use GPU (auto-detects by default)
- `--learning-rate 1e-4`: Learning rate

### Inference
- `--batch-size 32`: Processing batch size
- `--patch-size 256 256`: Patch size for processing
- `--calculate-metrics`: Show quality metrics
- `--device cuda`: Force GPU usage

## 🚨 Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
--batch-size 4

# Use CPU instead
--device cpu

# Reduce memory limit
--memory-limit 4.0
```

### Slow Performance?
```bash
# Check if GPU is being used
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Force GPU usage
--device cuda
```

### Import Errors?
```bash
# Reinstall
pip install -e .

# Check you're in the right directory
pwd  # Should show Image_quality_enhancement
```

## 📊 Monitor Training

```bash
# Install TensorBoard
pip install tensorboard

# Start monitoring (in separate terminal)
tensorboard --logdir my_training/logs

# Open browser to: http://localhost:6006
```

## 🎉 What's Next?

1. **Try the examples** to understand the workflow
2. **Use your own data** for real enhancement tasks  
3. **Experiment with parameters** to optimize for your use case
4. **Scale up** to AWS SageMaker for larger datasets
5. **Check the full documentation** in README.md

## 📞 Need Help?

- **Full Setup Guide**: See `LOCAL_SETUP_GUIDE.md`
- **Documentation**: Check `README.md`
- **Examples**: Browse the `examples/` folder
- **Issues**: https://github.com/tianzhuqin-argonne/Image_quality_enhancement/issues

Happy enhancing! 🎊