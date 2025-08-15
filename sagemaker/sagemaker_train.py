#!/usr/bin/env python3
"""
SageMaker training script for Sigray ML Platform.

This script is designed to run as a SageMaker training job, handling
data loading from S3, model training, and artifact saving.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add the source directory to Python path
sys.path.append('/opt/ml/code')

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import Sigray ML Platform components
from src.training.training_manager import TrainingManager
from src.core.config import TrainingConfig
from src.core.error_handling import setup_global_error_handling

# SageMaker specific paths
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
SM_OUTPUT_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SageMaker Training for Sigray ML Platform')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--patch-size', type=str, default='256,256', help='Patch size (height,width)')
    parser.add_argument('--overlap', type=int, default=32, help='Patch overlap')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--loss-function', type=str, default='mse', choices=['mse', 'l1', 'huber'], help='Loss function')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Training control
    parser.add_argument('--early-stopping-patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--checkpoint-interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--use-augmentation', action='store_true', help='Enable data augmentation')
    
    # SageMaker specific
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR, help='Model directory')
    parser.add_argument('--output-dir', type=str, default=SM_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--training-dir', type=str, default=SM_CHANNEL_TRAINING, help='Training data directory')
    parser.add_argument('--validation-dir', type=str, default=SM_CHANNEL_VALIDATION, help='Validation data directory')
    
    return parser.parse_args()


def setup_distributed_training():
    """Set up distributed training if multiple GPUs are available."""
    if not TORCH_AVAILABLE:
        return False, 0, 1
    
    # Check if we're in a distributed environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size > 1:
        # Initialize distributed training
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
        logger.info(f"Initialized distributed training: rank {rank}/{world_size}")
        return True, rank, world_size
    
    return False, 0, 1


def find_training_files(data_dir):
    """Find training files in the data directory."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    # Look for input and target directories
    input_dir = data_path / 'input'
    target_dir = data_path / 'target'
    
    if input_dir.exists() and target_dir.exists():
        # Structured data with input/target subdirectories
        input_files = sorted(list(input_dir.glob('*.tif*')))
        target_files = sorted(list(target_dir.glob('*.tif*')))
        
        if len(input_files) != len(target_files):
            raise ValueError(f"Mismatch in number of input ({len(input_files)}) and target ({len(target_files)}) files")
        
        return [str(f) for f in input_files], [str(f) for f in target_files]
    
    else:
        # Look for paired files (input_*.tif, target_*.tif)
        all_files = list(data_path.glob('*.tif*'))
        input_files = [f for f in all_files if 'input' in f.name.lower()]
        target_files = [f for f in all_files if 'target' in f.name.lower()]
        
        if not input_files or not target_files:
            raise ValueError(f"No training files found in {data_dir}. Expected input_*.tif and target_*.tif files")
        
        input_files.sort()
        target_files.sort()
        
        return [str(f) for f in input_files], [str(f) for f in target_files]


def log_sagemaker_metrics(metrics_dict):
    """Log metrics in SageMaker format for CloudWatch."""
    for key, value in metrics_dict.items():
        print(f"METRICS: {key}={value}")


def save_model_artifacts(trainer, model_path, output_dir):
    """Save model artifacts for SageMaker."""
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Export model for inference
    inference_model_path = Path(output_dir) / 'model.pth'
    trainer.export_model_for_inference(model_path, str(inference_model_path))
    
    # Save training summary
    summary = trainer.get_training_summary()
    summary_path = Path(output_dir) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save configuration
    config_path = Path(output_dir) / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'patch_size': trainer.config.patch_size,
            'overlap': trainer.config.overlap,
            'batch_size': trainer.config.batch_size,
            'learning_rate': trainer.config.learning_rate,
            'epochs': trainer.config.epochs,
            'optimizer': trainer.config.optimizer,
            'loss_function': trainer.config.loss_function,
            'device': trainer.device
        }, f, indent=2)
    
    logger.info(f"Model artifacts saved to {output_dir}")


def main():
    """Main training function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up error handling
        setup_global_error_handling()
        
        # Set up distributed training
        is_distributed, rank, world_size = setup_distributed_training()
        
        logger.info("Starting SageMaker training job")
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"Distributed: {is_distributed}, Rank: {rank}, World Size: {world_size}")
        
        # Parse patch size
        patch_size = tuple(map(int, args.patch_size.split(',')))
        
        # Create training configuration
        config = TrainingConfig(
            patch_size=patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            validation_split=args.validation_split,
            optimizer=args.optimizer,
            loss_function=args.loss_function,
            device=args.device,
            num_workers=args.num_workers,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_interval=args.checkpoint_interval,
            use_augmentation=args.use_augmentation
        )
        
        # Initialize training manager
        trainer = TrainingManager(
            config=config,
            output_dir='/tmp/training_output',  # Temporary directory
            use_tensorboard=False  # Disable TensorBoard in SageMaker
        )
        
        logger.info(f"Using device: {trainer.device}")
        
        # Find training files
        logger.info(f"Looking for training data in: {args.training_dir}")
        input_files, target_files = find_training_files(args.training_dir)
        
        logger.info(f"Found {len(input_files)} training file pairs")
        
        # Prepare training data
        train_loader, val_loader = trainer.prepare_training_data(
            input_paths=input_files,
            target_paths=target_files
        )
        
        # Create a simple pre-trained model (in real scenario, you'd load an actual pre-trained model)
        # For now, we'll create a basic U-Net model
        from src.models.model_manager import UNetModelManager
        model_manager = UNetModelManager()
        
        # Create and save a basic model
        pretrained_model_path = '/tmp/pretrained_model.pth'
        model = model_manager.create_unet_model()
        torch.save({'model_state_dict': model.state_dict()}, pretrained_model_path)
        
        # Fine-tune model
        logger.info("Starting model fine-tuning...")
        best_model_path = trainer.fine_tune_model(
            pretrained_model_path,
            train_loader,
            val_loader
        )
        
        # Log final metrics
        if trainer.training_state and trainer.training_state.metrics_history:
            final_metrics = trainer.training_state.metrics_history[-1]
            log_sagemaker_metrics({
                'final_train_loss': final_metrics.train_loss,
                'final_val_loss': final_metrics.val_loss,
                'best_val_loss': trainer.training_state.best_val_loss,
                'total_epochs': trainer.training_state.epoch,
                'best_epoch': trainer.training_state.best_epoch
            })
        
        # Save model artifacts
        save_model_artifacts(trainer, best_model_path, args.model_dir)
        
        # Validate model
        logger.info("Validating trained model...")
        validation_metrics = trainer.validate_model(best_model_path, val_loader)
        
        # Log validation metrics
        log_sagemaker_metrics({
            f'validation_{k}': v for k, v in validation_metrics.items()
        })
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {args.model_dir}")
        logger.info(f"Final validation metrics: {validation_metrics}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Log error for SageMaker
        log_sagemaker_metrics({'training_error': 1})
        raise


if __name__ == '__main__':
    main()