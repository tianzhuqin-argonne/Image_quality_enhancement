#!/usr/bin/env python3
"""
Command-line interface for training 3D image enhancement models.

This script provides a comprehensive CLI for training U-Net models
on 3D TIFF datasets with configurable parameters and monitoring.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..training.training_manager import TrainingManager
from ..core.config import TrainingConfig
from ..core.error_handling import setup_global_error_handling


class TrainingCLI:
    """Command-line interface for training pipeline."""
    
    def __init__(self):
        """Initialize training CLI."""
        self.parser = self._create_parser()
        self.logger = logging.getLogger(__name__)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for training CLI."""
        parser = argparse.ArgumentParser(
            description="Train 3D image enhancement models using U-Net architecture",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic training with default settings
  python -m src.cli.train_cli --input-dir data/inputs --target-dir data/targets --output-dir training_output

  # Training with custom configuration
  python -m src.cli.train_cli --input-dir data/inputs --target-dir data/targets \\
    --output-dir training_output --epochs 50 --batch-size 8 --learning-rate 1e-4

  # Resume training from checkpoint
  python -m src.cli.train_cli --input-dir data/inputs --target-dir data/targets \\
    --output-dir training_output --resume-from training_output/checkpoints/best_model.pth

  # Training with GPU acceleration
  python -m src.cli.train_cli --input-dir data/inputs --target-dir data/targets \\
    --output-dir training_output --device cuda --batch-size 16
            """
        )
        
        # Required arguments
        required = parser.add_argument_group('required arguments')
        required.add_argument(
            '--input-dir', type=str, required=True,
            help='Directory containing input (low-quality) TIFF files'
        )
        required.add_argument(
            '--target-dir', type=str, required=True,
            help='Directory containing target (high-quality) TIFF files'
        )
        required.add_argument(
            '--output-dir', type=str, required=True,
            help='Directory to save training outputs (models, logs, etc.)'
        )
        
        # Model and training parameters
        training = parser.add_argument_group('training parameters')
        training.add_argument(
            '--epochs', type=int, default=100,
            help='Number of training epochs (default: 100)'
        )
        training.add_argument(
            '--batch-size', type=int, default=16,
            help='Training batch size (default: 16)'
        )
        training.add_argument(
            '--learning-rate', type=float, default=1e-4,
            help='Learning rate (default: 1e-4)'
        )
        training.add_argument(
            '--validation-split', type=float, default=0.2,
            help='Fraction of data to use for validation (default: 0.2)'
        )
        training.add_argument(
            '--optimizer', type=str, choices=['adam', 'adamw'], default='adam',
            help='Optimizer to use (default: adam)'
        )
        training.add_argument(
            '--loss-function', type=str, choices=['mse', 'l1', 'huber'], default='mse',
            help='Loss function to use (default: mse)'
        )
        training.add_argument(
            '--weight-decay', type=float, default=1e-5,
            help='Weight decay for regularization (default: 1e-5)'
        )
        
        # Model architecture
        model = parser.add_argument_group('model parameters')
        model.add_argument(
            '--patch-size', type=int, nargs=2, default=[256, 256], metavar=('H', 'W'),
            help='Patch size for training (height width) (default: 256 256)'
        )
        model.add_argument(
            '--overlap', type=int, default=32,
            help='Overlap between patches (default: 32)'
        )
        model.add_argument(
            '--model-depth', type=int, default=6,
            help='U-Net depth (number of levels) (default: 6)'
        )
        model.add_argument(
            '--base-channels', type=int, default=64,
            help='Base number of channels in U-Net (default: 64)'
        )
        
        # Hardware and performance
        hardware = parser.add_argument_group('hardware and performance')
        hardware.add_argument(
            '--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
            help='Device to use for training (default: auto)'
        )
        hardware.add_argument(
            '--num-workers', type=int, default=4,
            help='Number of data loading workers (default: 4)'
        )
        hardware.add_argument(
            '--gradient-clip-value', type=float, default=1.0,
            help='Gradient clipping value (default: 1.0)'
        )
        
        # Checkpointing and monitoring
        checkpoint = parser.add_argument_group('checkpointing and monitoring')
        checkpoint.add_argument(
            '--checkpoint-interval', type=int, default=10,
            help='Save checkpoint every N epochs (default: 10)'
        )
        checkpoint.add_argument(
            '--early-stopping-patience', type=int, default=15,
            help='Early stopping patience in epochs (default: 15)'
        )
        checkpoint.add_argument(
            '--resume-from', type=str,
            help='Resume training from checkpoint file'
        )
        checkpoint.add_argument(
            '--pretrained-model', type=str,
            help='Path to pre-trained model to fine-tune'
        )
        
        # Data augmentation
        augmentation = parser.add_argument_group('data augmentation')
        augmentation.add_argument(
            '--use-augmentation', action='store_true', default=True,
            help='Enable data augmentation (default: True)'
        )
        augmentation.add_argument(
            '--no-augmentation', action='store_false', dest='use_augmentation',
            help='Disable data augmentation'
        )
        augmentation.add_argument(
            '--augmentation-probability', type=float, default=0.5,
            help='Probability of applying augmentation (default: 0.5)'
        )
        
        # Logging and monitoring
        logging_group = parser.add_argument_group('logging and monitoring')
        logging_group.add_argument(
            '--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
            help='Logging level (default: INFO)'
        )
        logging_group.add_argument(
            '--tensorboard', action='store_true', default=True,
            help='Enable TensorBoard logging (default: True)'
        )
        logging_group.add_argument(
            '--no-tensorboard', action='store_false', dest='tensorboard',
            help='Disable TensorBoard logging'
        )
        logging_group.add_argument(
            '--config-file', type=str,
            help='Load configuration from JSON file'
        )
        logging_group.add_argument(
            '--save-config', type=str,
            help='Save current configuration to JSON file'
        )
        
        # Data selection
        data = parser.add_argument_group('data selection')
        data.add_argument(
            '--slice-indices', type=int, nargs='+',
            help='Specific slice indices to use for training'
        )
        data.add_argument(
            '--max-files', type=int,
            help='Maximum number of files to use for training'
        )
        
        return parser
    
    def _setup_logging(self, log_level: str, output_dir: str) -> None:
        """Setup logging configuration."""
        log_file = Path(output_dir) / "training.log"
        
        # Setup global error handling with log file
        setup_global_error_handling(str(log_file))
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger.info(f"Logging configured: level={log_level}, file={log_file}")
    
    def _load_config_from_file(self, config_file: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            sys.exit(1)
    
    def _save_config_to_file(self, config: TrainingConfig, config_file: str) -> None:
        """Save configuration to JSON file."""
        try:
            config_dict = {
                'patch_size': config.patch_size,
                'overlap': config.overlap,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'epochs': config.epochs,
                'validation_split': config.validation_split,
                'weight_decay': config.weight_decay,
                'optimizer': config.optimizer,
                'use_scheduler': config.use_scheduler,
                'loss_function': config.loss_function,
                'model_depth': config.model_depth,
                'base_channels': config.base_channels,
                'device': config.device,
                'num_workers': config.num_workers,
                'checkpoint_interval': config.checkpoint_interval,
                'early_stopping_patience': config.early_stopping_patience,
                'gradient_clip_value': config.gradient_clip_value,
                'use_augmentation': config.use_augmentation,
                'augmentation_probability': config.augmentation_probability
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def _create_training_config(self, args: argparse.Namespace) -> TrainingConfig:
        """Create training configuration from arguments."""
        # Load base config from file if specified
        if args.config_file:
            file_config = self._load_config_from_file(args.config_file)
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None and key in file_config:
                    file_config[key] = value
            
            # Convert to TrainingConfig
            config = TrainingConfig(
                patch_size=tuple(file_config.get('patch_size', [256, 256])),
                overlap=file_config.get('overlap', 32),
                batch_size=file_config.get('batch_size', 16),
                learning_rate=file_config.get('learning_rate', 1e-4),
                epochs=file_config.get('epochs', 100),
                validation_split=file_config.get('validation_split', 0.2),
                weight_decay=file_config.get('weight_decay', 1e-5),
                optimizer=file_config.get('optimizer', 'adam'),
                use_scheduler=file_config.get('use_scheduler', True),
                loss_function=file_config.get('loss_function', 'mse'),
                model_depth=file_config.get('model_depth', 6),
                base_channels=file_config.get('base_channels', 64),
                device=file_config.get('device', 'auto'),
                num_workers=file_config.get('num_workers', 4),
                checkpoint_interval=file_config.get('checkpoint_interval', 10),
                early_stopping_patience=file_config.get('early_stopping_patience', 15),
                gradient_clip_value=file_config.get('gradient_clip_value', 1.0),
                use_augmentation=file_config.get('use_augmentation', True),
                augmentation_probability=file_config.get('augmentation_probability', 0.5)
            )
        else:
            # Create config from command line arguments
            config = TrainingConfig(
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                validation_split=args.validation_split,
                weight_decay=args.weight_decay,
                optimizer=args.optimizer,
                use_scheduler=True,
                loss_function=args.loss_function,
                model_depth=args.model_depth,
                base_channels=args.base_channels,
                device=args.device,
                num_workers=args.num_workers,
                checkpoint_interval=args.checkpoint_interval,
                early_stopping_patience=args.early_stopping_patience,
                gradient_clip_value=args.gradient_clip_value,
                use_augmentation=args.use_augmentation,
                augmentation_probability=args.augmentation_probability
            )
        
        return config
    
    def _find_tiff_files(self, directory: str, max_files: Optional[int] = None) -> List[str]:
        """Find TIFF files in directory."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find TIFF files
        tiff_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tiff_files.extend(directory.glob(ext))
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {directory}")
        
        # Sort for consistent ordering
        tiff_files = sorted([str(f) for f in tiff_files])
        
        # Limit number of files if specified
        if max_files:
            tiff_files = tiff_files[:max_files]
        
        self.logger.info(f"Found {len(tiff_files)} TIFF files in {directory}")
        return tiff_files
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run training CLI."""
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)
            
            # Check PyTorch availability
            if not TORCH_AVAILABLE:
                print("Error: PyTorch is required for training but not available.")
                print("Please install PyTorch: pip install torch torchvision")
                return 1
            
            # Create output directory
            output_dir = Path(parsed_args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup logging
            self._setup_logging(parsed_args.log_level, str(output_dir))
            
            self.logger.info("Starting 3D image enhancement training")
            self.logger.info(f"PyTorch version: {torch.__version__}")
            self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Create training configuration
            config = self._create_training_config(parsed_args)
            
            # Save configuration if requested
            if parsed_args.save_config:
                self._save_config_to_file(config, parsed_args.save_config)
            
            # Find input and target files
            input_files = self._find_tiff_files(parsed_args.input_dir, parsed_args.max_files)
            target_files = self._find_tiff_files(parsed_args.target_dir, parsed_args.max_files)
            
            if len(input_files) != len(target_files):
                raise ValueError(
                    f"Number of input files ({len(input_files)}) must match "
                    f"number of target files ({len(target_files)})"
                )
            
            # Initialize training manager
            training_manager = TrainingManager(
                config=config,
                output_dir=str(output_dir),
                use_tensorboard=parsed_args.tensorboard
            )
            
            # Prepare training data
            self.logger.info("Preparing training data...")
            train_loader, val_loader = training_manager.prepare_training_data(
                input_files, target_files, parsed_args.slice_indices
            )
            
            # Determine model path
            if parsed_args.resume_from:
                model_path = parsed_args.resume_from
                self.logger.info(f"Resuming training from: {model_path}")
            elif parsed_args.pretrained_model:
                model_path = parsed_args.pretrained_model
                self.logger.info(f"Fine-tuning pre-trained model: {model_path}")
            else:
                # Use default pre-trained model or create new one
                self.logger.warning("No pre-trained model specified. Training from scratch may not work well.")
                self.logger.warning("Consider using --pretrained-model to specify a pre-trained model.")
                return 1
            
            # Start training
            self.logger.info("Starting model training...")
            best_model_path = training_manager.fine_tune_model(
                model_path, train_loader, val_loader
            )
            
            # Export model for inference
            export_path = output_dir / "final_model.pth"
            training_manager.export_model_for_inference(best_model_path, str(export_path))
            
            # Get training summary
            summary = training_manager.get_training_summary()
            
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Best model: {best_model_path}")
            self.logger.info(f"Exported model: {export_path}")
            self.logger.info(f"Total epochs: {summary.get('total_epochs', 'N/A')}")
            self.logger.info(f"Best validation loss: {summary.get('best_validation_loss', 'N/A'):.6f}")
            self.logger.info(f"Total training time: {summary.get('total_training_time', 'N/A'):.1f}s")
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return 1


def main():
    """Main entry point for training CLI."""
    cli = TrainingCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())