"""
Training Manager for 3D image enhancement system.

This module provides the TrainingManager class that orchestrates the complete
training process including data preparation, model fine-tuning, progress monitoring,
and checkpointing.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
    
    # TensorBoard is optional
    try:
        from torch.utils.tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None
        
except ImportError:
    TORCH_AVAILABLE = False
    TENSORBOARD_AVAILABLE = False

from .data_preparation import TrainingDataLoader, AugmentationConfig
from ..models.model_manager import UNetModelManager
from ..core.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    train_time: float
    val_time: float
    learning_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingState:
    """Current state of training."""
    epoch: int
    best_val_loss: float
    best_epoch: int
    patience_counter: int
    total_train_time: float
    metrics_history: List[TrainingMetrics]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'total_train_time': self.total_train_time,
            'metrics_history': [m.to_dict() for m in self.metrics_history]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingState':
        """Create from dictionary."""
        metrics_history = [
            TrainingMetrics(**m) for m in data.get('metrics_history', [])
        ]
        return cls(
            epoch=data['epoch'],
            best_val_loss=data['best_val_loss'],
            best_epoch=data['best_epoch'],
            patience_counter=data['patience_counter'],
            total_train_time=data['total_train_time'],
            metrics_history=metrics_history
        )


class TrainingManager:
    """
    Manages the complete training process for U-Net model fine-tuning.
    
    Features:
    - Data preparation with validation split
    - Model fine-tuning with configurable loss functions
    - Training progress monitoring and metrics tracking
    - Model checkpointing and early stopping
    - TensorBoard logging support
    - Comprehensive error handling
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str = "training_output",
        use_tensorboard: bool = True
    ):
        """
        Initialize training manager.
        
        Args:
            config: Training configuration
            output_dir: Directory for saving outputs
            use_tensorboard: Whether to use TensorBoard logging
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrainingManager")
        
        self.config = config
        self.output_dir = Path(output_dir)
        self.use_tensorboard = use_tensorboard
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Resolve device
        self.device = self._resolve_device(config.device)
        
        # Initialize components
        self.data_loader = TrainingDataLoader(
            patch_size=config.patch_size,
            overlap=config.overlap,
            device=self.device
        )
        self.model_manager = UNetModelManager()
        
        # Training state
        self.training_state: Optional[TrainingState] = None
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        # TensorBoard writer
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.logs_dir))
        elif self.use_tensorboard and not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available, logging disabled")
            self.use_tensorboard = False
        
        logger.info(f"Initialized TrainingManager with output_dir: {output_dir}")
    
    def _resolve_device(self, device_config: str) -> str:
        """Resolve device configuration to actual device string."""
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_config
    
    def prepare_training_data(
        self,
        input_paths: List[str],
        target_paths: List[str],
        slice_indices: Optional[List[int]] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare training and validation data loaders.
        
        Args:
            input_paths: Paths to input (low-quality) TIFF files
            target_paths: Paths to target (high-quality) TIFF files
            slice_indices: Optional list of slice indices to use
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Preparing training data...")
        
        try:
            # Load training pairs
            training_pairs = self.data_loader.load_training_data(
                input_paths, target_paths, slice_indices
            )
            
            if not training_pairs:
                raise ValueError("No training pairs loaded")
                
        except FileNotFoundError as e:
            logger.error(f"Training data files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            raise ValueError(f"Failed to prepare training data: {e}") from e
        
        # Get data statistics
        stats = self.data_loader.get_data_statistics(training_pairs)
        logger.info(f"Training data statistics: {stats}")
        
        # Save statistics
        stats_path = self.output_dir / "data_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Create augmentation config if enabled
        augmentation_config = None
        if self.config.use_augmentation:
            augmentation_config = AugmentationConfig(
                horizontal_flip=True,
                vertical_flip=True,
                rotation_90=True,
                gaussian_noise_std=0.01,
                brightness_range=(0.9, 1.1),
                contrast_range=(0.9, 1.1),
                apply_probability=0.5
            )
        
        # Create data loaders
        train_loader, val_loader = self.data_loader.create_data_loaders(
            training_pairs,
            validation_split=self.config.validation_split,
            batch_size=self.config.batch_size,
            augmentation_config=augmentation_config,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        logger.info(
            f"Created data loaders: train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}"
        )
        
        return train_loader, val_loader
    
    def fine_tune_model(
        self,
        pretrained_model_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> str:
        """
        Fine-tune the pre-trained U-Net model.
        
        Args:
            pretrained_model_path: Path to pre-trained model
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Path to the best trained model
        """
        logger.info(f"Starting fine-tuning with model: {pretrained_model_path}")
        
        try:
            # Load pre-trained model
            self.model = self.model_manager.load_pretrained_model(pretrained_model_path)
            self.model = self.model.to(self.device)
            
            # Validate architecture
            if not self.model_manager.validate_architecture(self.model):
                raise ValueError("Model architecture validation failed")
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Pre-trained model not found: {pretrained_model_path}")
        except Exception as e:
            logger.error(f"Failed to load pre-trained model: {e}")
            raise ValueError(f"Failed to load pre-trained model from {pretrained_model_path}: {e}") from e
        
        # Setup training components
        self._setup_training_components()
        
        # Initialize training state
        self.training_state = TrainingState(
            epoch=0,
            best_val_loss=float('inf'),
            best_epoch=0,
            patience_counter=0,
            total_train_time=0.0,
            metrics_history=[]
        )
        
        # Training loop
        try:
            for epoch in range(self.config.epochs):
                epoch_start_time = time.time()
                
                try:
                    # Training phase
                    train_loss, train_time = self._train_epoch(train_loader)
                    
                    # Validation phase
                    val_loss, val_time = self._validate_epoch(val_loader)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"GPU out of memory at epoch {epoch + 1}. Try reducing batch size.")
                        raise RuntimeError(f"GPU out of memory. Current batch size: {self.config.batch_size}. "
                                         f"Try reducing batch size or using CPU.") from e
                    else:
                        logger.error(f"Runtime error during training at epoch {epoch + 1}: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error during training at epoch {epoch + 1}: {e}")
                    raise
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step(val_loss)
                
                # Record metrics
                current_lr = self.optimizer.param_groups[0]['lr']
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_time=train_time,
                    val_time=val_time,
                    learning_rate=current_lr
                )
                
                self.training_state.epoch = epoch + 1
                self.training_state.total_train_time += (train_time + val_time)
                self.training_state.metrics_history.append(metrics)
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Check for improvement
                if val_loss < self.training_state.best_val_loss:
                    self.training_state.best_val_loss = val_loss
                    self.training_state.best_epoch = epoch + 1
                    self.training_state.patience_counter = 0
                    
                    # Save best model
                    best_model_path = self._save_checkpoint("best_model.pth", is_best=True)
                    logger.info(f"New best model saved: {best_model_path}")
                else:
                    self.training_state.patience_counter += 1
                
                # Save regular checkpoint
                if (epoch + 1) % self.config.checkpoint_interval == 0:
                    checkpoint_path = self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                # Early stopping check
                if (self.config.early_stopping_patience > 0 and 
                    self.training_state.patience_counter >= self.config.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs "
                        f"(patience: {self.config.early_stopping_patience})"
                    )
                    break
                
                # Log progress
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                    f"lr={current_lr:.2e}, time={train_time + val_time:.1f}s"
                )
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Save final state
            self._save_training_state()
            if self.writer:
                self.writer.close()
        
        # Return path to best model
        best_model_path = self.checkpoints_dir / "best_model.pth"
        if not best_model_path.exists():
            # Fallback to last checkpoint
            best_model_path = self._save_checkpoint("final_model.pth", is_best=True)
        
        logger.info(f"Training completed. Best model: {best_model_path}")
        return str(best_model_path)
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss function."""
        # Setup optimizer
        if self.config.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Setup learning rate scheduler
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        
        # Setup loss function
        if self.config.loss_function.lower() == 'mse':
            self.criterion = nn.MSELoss()
        elif self.config.loss_function.lower() == 'l1':
            self.criterion = nn.L1Loss()
        elif self.config.loss_function.lower() == 'huber':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
        
        logger.info(
            f"Training setup: optimizer={self.config.optimizer}, "
            f"loss={self.config.loss_function}, lr={self.config.learning_rate}"
        )
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.gradient_clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_value
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch progress
            if batch_idx % 50 == 0:
                logger.debug(
                    f"Train batch {batch_idx}/{len(train_loader)}: loss={loss.item():.6f}"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_time = time.time() - start_time
        
        return avg_loss, train_time
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_time = time.time() - start_time
        
        return avg_loss, val_time
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to TensorBoard and console."""
        if self.writer:
            self.writer.add_scalar('Loss/Train', metrics.train_loss, metrics.epoch)
            self.writer.add_scalar('Loss/Validation', metrics.val_loss, metrics.epoch)
            self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
            self.writer.add_scalar('Time/Train', metrics.train_time, metrics.epoch)
            self.writer.add_scalar('Time/Validation', metrics.val_time, metrics.epoch)
    
    def _save_checkpoint(self, filename: str, is_best: bool = False) -> str:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoints_dir / filename
        
        checkpoint = {
            'epoch': self.training_state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.training_state.to_dict(),
            'config': asdict(self.config)
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as best model
            best_path = self.checkpoints_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
        
        return str(checkpoint_path)
    
    def _save_training_state(self):
        """Save training state to JSON."""
        if self.training_state:
            state_path = self.output_dir / "training_state.json"
            with open(state_path, 'w') as f:
                json.dump(self.training_state.to_dict(), f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load training checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint loaded successfully
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            if self.model is None:
                raise ValueError("Model must be initialized before loading checkpoint")
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            if 'training_state' in checkpoint:
                self.training_state = TrainingState.from_dict(checkpoint['training_state'])
            
            logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def validate_model(
        self,
        model_path: str,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate a trained model and compute quality metrics.
        
        Args:
            model_path: Path to trained model
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info(f"Validating model: {model_path}")
        
        # Load model
        model = self.model_manager.load_pretrained_model(model_path)
        model = model.to(self.device)
        model.eval()
        
        # Validation metrics
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_psnr = 0.0
        num_batches = 0
        
        mse_criterion = nn.MSELoss()
        mae_criterion = nn.L1Loss()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                
                # Compute losses
                mse_loss = mse_criterion(outputs, targets)
                mae_loss = mae_criterion(outputs, targets)
                
                # Compute PSNR
                mse_value = mse_loss.item()
                if mse_value > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse_value))
                else:
                    psnr = float('inf')
                
                total_loss += self.criterion(outputs, targets).item()
                total_mse += mse_loss.item()
                total_mae += mae_loss.item()
                total_psnr += psnr
                num_batches += 1
        
        # Average metrics
        metrics = {
            'validation_loss': total_loss / num_batches,
            'mse': total_mse / num_batches,
            'mae': total_mae / num_batches,
            'psnr': total_psnr / num_batches,
            'num_samples': num_batches * val_loader.batch_size
        }
        
        logger.info(f"Validation metrics: {metrics}")
        return metrics
    
    def export_model_for_inference(
        self,
        model_path: str,
        export_path: str
    ) -> None:
        """
        Export trained model for inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            export_path: Path to save exported model
        """
        logger.info(f"Exporting model for inference: {model_path} -> {export_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract just the model state dict
        model_state = checkpoint['model_state_dict']
        
        # Save model for inference (without training-specific data)
        export_data = {
            'model_state_dict': model_state,
            'model_config': {
                'patch_size': self.config.patch_size,
                'architecture': 'unet_6_levels'
            },
            'training_info': {
                'best_epoch': checkpoint.get('training_state', {}).get('best_epoch', 'unknown'),
                'best_val_loss': checkpoint.get('training_state', {}).get('best_val_loss', 'unknown')
            }
        }
        
        # Create export directory
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(export_data, export_path)
        logger.info(f"Model exported successfully: {export_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training process.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_state:
            return {}
        
        return {
            'total_epochs': self.training_state.epoch,
            'best_epoch': self.training_state.best_epoch,
            'best_validation_loss': self.training_state.best_val_loss,
            'total_training_time': self.training_state.total_train_time,
            'final_learning_rate': (
                self.training_state.metrics_history[-1].learning_rate 
                if self.training_state.metrics_history else None
            ),
            'training_config': asdict(self.config)
        }