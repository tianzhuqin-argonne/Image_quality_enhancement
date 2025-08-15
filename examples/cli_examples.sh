#!/bin/bash
"""
Command-line interface examples for 3D image enhancement system.

This script demonstrates various ways to use the training and inference
command-line interfaces with different configurations and options.
"""

echo "=== 3D Image Enhancement CLI Examples ==="
echo

# Note: These are example commands. Adjust paths and parameters as needed.

echo "1. Basic Training Examples"
echo "========================="
echo

echo "# Basic training with default settings"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output"
echo

echo "# Training with custom parameters"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --epochs 50 \\"
echo "  --batch-size 8 \\"
echo "  --learning-rate 1e-4 \\"
echo "  --device cuda \\"
echo "  --patch-size 512 512"
echo

echo "# Training with configuration file"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --config-file configs/training_config.json"
echo

echo "# Resume training from checkpoint"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --resume-from training_output/checkpoints/best_model.pth"
echo

echo "# Training with data augmentation disabled"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --no-augmentation \\"
echo "  --epochs 30"
echo

echo
echo "2. Basic Inference Examples"
echo "==========================="
echo

echo "# Basic single file enhancement"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input data/test/input.tif \\"
echo "  --output data/test/enhanced.tif"
echo

echo "# Batch processing with quality metrics"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input-dir data/test/inputs \\"
echo "  --output-dir data/test/outputs \\"
echo "  --calculate-metrics \\"
echo "  --save-metrics results/metrics.json"
echo

echo "# High-quality processing with large patches"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input data/test/large_image.tif \\"
echo "  --output data/test/enhanced_large.tif \\"
echo "  --patch-size 512 512 \\"
echo "  --overlap 64 \\"
echo "  --device cuda \\"
echo "  --batch-size 4"
echo

echo "# Memory-constrained processing"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input data/test/huge_image.tif \\"
echo "  --output data/test/enhanced_huge.tif \\"
echo "  --memory-limit 2.0 \\"
echo "  --batch-size 1 \\"
echo "  --patch-size 256 256"
echo

echo "# Batch processing with file pattern and recursion"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input-dir data/test \\"
echo "  --output-dir data/enhanced \\"
echo "  --file-pattern '*.tiff' \\"
echo "  --recursive \\"
echo "  --max-files 100"
echo

echo
echo "3. Advanced Configuration Examples"
echo "================================="
echo

echo "# Save training configuration for reuse"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --epochs 100 \\"
echo "  --batch-size 16 \\"
echo "  --learning-rate 5e-5 \\"
echo "  --optimizer adamw \\"
echo "  --loss-function huber \\"
echo "  --save-config configs/my_training_config.json"
echo

echo "# Load and modify inference configuration"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input data/test/input.tif \\"
echo "  --output data/test/enhanced.tif \\"
echo "  --config-file configs/inference_config.json \\"
echo "  --batch-size 8 \\"
echo "  --device cuda"
echo

echo
echo "4. Monitoring and Debugging Examples"
echo "===================================="
echo

echo "# Training with detailed logging and TensorBoard"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --log-level DEBUG \\"
echo "  --tensorboard \\"
echo "  --checkpoint-interval 5"
echo

echo "# Inference with progress monitoring"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input-dir data/test/inputs \\"
echo "  --output-dir data/test/outputs \\"
echo "  --progress \\"
echo "  --log-level INFO \\"
echo "  --calculate-metrics"
echo

echo "# Quiet processing (errors only)"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input-dir data/test/inputs \\"
echo "  --output-dir data/test/outputs \\"
echo "  --quiet"
echo

echo
echo "5. Example Configuration Files"
echo "============================="
echo

echo "# Example training configuration (training_config.json):"
cat << 'EOF'
{
  "patch_size": [256, 256],
  "overlap": 32,
  "batch_size": 16,
  "learning_rate": 1e-4,
  "epochs": 100,
  "validation_split": 0.2,
  "weight_decay": 1e-5,
  "optimizer": "adam",
  "use_scheduler": true,
  "loss_function": "mse",
  "model_depth": 6,
  "base_channels": 64,
  "device": "auto",
  "num_workers": 4,
  "checkpoint_interval": 10,
  "early_stopping_patience": 15,
  "gradient_clip_value": 1.0,
  "use_augmentation": true,
  "augmentation_probability": 0.5
}
EOF
echo

echo "# Example inference configuration (inference_config.json):"
cat << 'EOF'
{
  "patch_size": [256, 256],
  "overlap": 32,
  "batch_size": 32,
  "device": "auto",
  "memory_limit_gb": 8.0,
  "enable_quality_metrics": true,
  "preserve_metadata": true,
  "output_format": "tiff",
  "compression": null
}
EOF
echo

echo
echo "6. Workflow Examples"
echo "==================="
echo

echo "# Complete workflow: train then infer"
echo "# Step 1: Train model"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --epochs 50 \\"
echo "  --device cuda"
echo
echo "# Step 2: Use trained model for inference"
echo "python -m src.cli.inference_cli \\"
echo "  --model training_output/final_model.pth \\"
echo "  --input-dir data/test/inputs \\"
echo "  --output-dir data/test/outputs \\"
echo "  --calculate-metrics \\"
echo "  --save-metrics results/test_metrics.json"
echo

echo
echo "7. Performance Optimization Examples"
echo "===================================="
echo

echo "# GPU training with large batches"
echo "python -m src.cli.train_cli \\"
echo "  --input-dir data/training/inputs \\"
echo "  --target-dir data/training/targets \\"
echo "  --output-dir training_output \\"
echo "  --device cuda \\"
echo "  --batch-size 32 \\"
echo "  --num-workers 8 \\"
echo "  --patch-size 512 512"
echo

echo "# CPU inference optimized for memory"
echo "python -m src.cli.inference_cli \\"
echo "  --model trained_models/best_model.pth \\"
echo "  --input data/test/large_image.tif \\"
echo "  --output data/test/enhanced.tif \\"
echo "  --device cpu \\"
echo "  --memory-limit 4.0 \\"
echo "  --batch-size 4 \\"
echo "  --patch-size 256 256"
echo

echo
echo "Note: Adjust file paths, model paths, and parameters according to your setup."
echo "For more options, use --help with either command:"
echo "  python -m src.cli.train_cli --help"
echo "  python -m src.cli.inference_cli --help"