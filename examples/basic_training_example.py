#!/usr/bin/env python3
"""
Basic training example for 3D image enhancement.

This example demonstrates how to train a U-Net model for 3D image enhancement
using the programmatic API with synthetic data.
"""

import logging
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.testing.test_fixtures import TestDataFixtures
    from src.training.training_manager import TrainingManager
    from src.core.config import TrainingConfig
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


def main():
    """Run basic training example."""
    logger.info("Starting basic training example")
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize test fixtures for synthetic data
        with TestDataFixtures(temp_dir) as fixtures:
            
            # Step 1: Create synthetic training dataset
            logger.info("Creating synthetic training dataset...")
            dataset_pairs = fixtures.create_test_dataset(
                num_pairs=8,  # Small dataset for quick example
                volume_size="small",  # 5x256x256 volumes
                degradation_level="moderate"  # Moderate degradation
            )
            
            input_paths = [pair[0] for pair in dataset_pairs]
            target_paths = [pair[1] for pair in dataset_pairs]
            
            logger.info(f"Created {len(dataset_pairs)} training pairs")
            
            # Step 2: Configure training parameters
            training_config = TrainingConfig(
                epochs=5,  # Few epochs for quick example
                batch_size=2,  # Small batch size for CPU
                learning_rate=1e-3,  # Higher learning rate for faster convergence
                device='cpu',  # Use CPU for compatibility
                checkpoint_interval=2,
                early_stopping_patience=3,
                use_augmentation=False  # Disable for faster training
            )
            
            logger.info("Training configuration:")
            logger.info(f"  Epochs: {training_config.epochs}")
            logger.info(f"  Batch size: {training_config.batch_size}")
            logger.info(f"  Learning rate: {training_config.learning_rate}")
            logger.info(f"  Device: {training_config.device}")
            
            # Step 3: Initialize training manager
            training_output_dir = Path(temp_dir) / "training_output"
            training_manager = TrainingManager(
                config=training_config,
                output_dir=str(training_output_dir),
                use_tensorboard=False  # Disable for simplicity
            )
            
            # Step 4: Prepare training data
            logger.info("Preparing training data...")
            train_loader, val_loader = training_manager.prepare_training_data(
                input_paths, target_paths
            )
            
            logger.info(f"Training batches: {len(train_loader)}")
            logger.info(f"Validation batches: {len(val_loader)}")
            
            # Step 5: Create initial model (in real scenario, use pre-trained)
            logger.info("Creating initial model...")
            initial_model_path = fixtures.save_mock_model("initial_model.pth")
            
            # Step 6: Train the model
            logger.info("Starting training...")
            best_model_path = training_manager.fine_tune_model(
                initial_model_path, train_loader, val_loader
            )
            
            logger.info(f"Training completed! Best model: {best_model_path}")
            
            # Step 7: Get training summary
            summary = training_manager.get_training_summary()
            
            logger.info("Training Summary:")
            logger.info(f"  Total epochs: {summary.get('total_epochs', 'N/A')}")
            logger.info(f"  Best epoch: {summary.get('best_epoch', 'N/A')}")
            logger.info(f"  Best validation loss: {summary.get('best_validation_loss', 'N/A'):.6f}")
            logger.info(f"  Total training time: {summary.get('total_training_time', 'N/A'):.1f}s")
            
            # Step 8: Export model for inference
            export_path = Path(temp_dir) / "final_model.pth"
            training_manager.export_model_for_inference(best_model_path, str(export_path))
            
            logger.info(f"Model exported for inference: {export_path}")
            
            # Step 9: Validate the trained model
            logger.info("Validating trained model...")
            validation_metrics = training_manager.validate_model(best_model_path, val_loader)
            
            logger.info("Validation Metrics:")
            logger.info(f"  Validation loss: {validation_metrics.get('validation_loss', 'N/A'):.6f}")
            logger.info(f"  PSNR: {validation_metrics.get('psnr', 'N/A'):.2f}")
            logger.info(f"  MSE: {validation_metrics.get('mse', 'N/A'):.6f}")
            logger.info(f"  MAE: {validation_metrics.get('mae', 'N/A'):.6f}")
            
            logger.info("Basic training example completed successfully!")
            
    except Exception as e:
        logger.error(f"Training example failed: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files")


if __name__ == '__main__':
    main()