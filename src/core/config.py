"""
Configuration classes for training and inference pipelines.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline."""
    
    # Patch processing parameters
    patch_size: Tuple[int, int] = (256, 256)
    overlap: int = 32
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 100
    validation_split: float = 0.2
    weight_decay: float = 1e-5
    
    # Optimizer and scheduler
    optimizer: str = "adam"  # adam, adamw
    use_scheduler: bool = True
    
    # Loss function
    loss_function: str = "mse"  # mse, l1, huber
    
    # Model parameters
    model_depth: int = 6
    base_channels: int = 64
    
    # Hardware and performance
    device: str = "auto"  # auto, cpu, cuda
    num_workers: int = 4
    
    # Checkpointing and monitoring
    checkpoint_interval: int = 10
    early_stopping_patience: int = 15
    
    # Gradient clipping
    gradient_clip_value: float = 1.0
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_probability: float = 0.5


@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline."""
    
    # Patch processing parameters
    patch_size: Tuple[int, int] = (256, 256)
    overlap: int = 32
    
    # Processing parameters
    batch_size: int = 32
    device: str = "auto"  # auto, cpu, cuda
    memory_limit_gb: float = 8.0
    
    # Quality and performance
    enable_quality_metrics: bool = True
    preserve_metadata: bool = True
    
    # Output options
    output_format: str = "tiff"
    compression: Optional[str] = None


@dataclass
class EnhancementResult:
    """Result object containing enhancement processing information."""
    
    # Processing status
    success: bool
    processing_time: float
    
    # Data information
    input_shape: Tuple[int, int, int]  # (slices, height, width)
    output_path: Optional[str] = None
    
    # Quality metrics
    quality_metrics: Dict[str, float] = None
    
    # Error handling
    error_message: Optional[str] = None
    warnings: list = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.warnings is None:
            self.warnings = []
    
    def add_warning(self, message: str) -> None:
        """Add a warning message to the result."""
        self.warnings.append(message)
    
    def add_quality_metric(self, name: str, value: float) -> None:
        """Add a quality metric to the result."""
        self.quality_metrics[name] = value
    
    def is_successful(self) -> bool:
        """Check if the enhancement was successful."""
        return self.success and self.error_message is None