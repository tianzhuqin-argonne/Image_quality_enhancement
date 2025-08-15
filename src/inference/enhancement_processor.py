"""
Enhancement processor for applying trained models to 3D volumes.

This module provides efficient model application with memory optimization,
batch processing, and progress monitoring for production inference.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Callable
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..models.model_manager import UNetModelManager
from ..core.data_models import SliceStack
from ..core.config import InferenceConfig

logger = logging.getLogger(__name__)


class EnhancementProcessor:
    """
    Enhancement processor for applying trained U-Net models to 3D volumes.
    
    Handles model loading, batch processing, and memory-efficient inference
    for large 3D image volumes. Designed for production use with automatic
    resource management and progress monitoring.
    
    Features:
    - Memory-efficient batch processing
    - Progress monitoring and callbacks
    - Automatic model validation
    - CPU/GPU flexibility with fallback
    - Error handling and recovery
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize enhancement processor.
        
        Args:
            config: Inference configuration
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, enhancement processor disabled")
            raise ImportError("PyTorch is required for EnhancementProcessor")
        
        self.config = config
        
        # Resolve device
        self.device = self._resolve_device(config.device)
        logger.info(f"EnhancementProcessor initialized with device: {self.device}")
        
        # Initialize model manager
        self.model_manager = UNetModelManager()
        self.model: Optional[nn.Module] = None
        
        # Processing state
        self.is_model_loaded = False
        
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
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained model for inference.
        
        Args:
            model_path: Path to trained model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model is invalid or incompatible
        """
        logger.info(f"Loading model: {model_path}")
        
        try:
            # Load model
            self.model = self.model_manager.load_pretrained_model(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Validate architecture
            if not self.model_manager.validate_architecture(self.model):
                raise ValueError("Model architecture validation failed")
            
            self.is_model_loaded = True
            logger.info(f"Model loaded successfully on {self.device}")
            
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ValueError(f"Failed to load model from {model_path}: {e}") from e
    
    def enhance_patches(
        self,
        patches: torch.Tensor,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> torch.Tensor:
        """
        Enhance patches using the loaded model.
        
        Args:
            patches: Input patches tensor (N, H, W)
            batch_size: Batch size for processing (default from config)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Enhanced patches tensor
            
        Raises:
            RuntimeError: If model is not loaded
            RuntimeError: If GPU memory issues occur
        """
        if not self.is_model_loaded:
            raise RuntimeError("Model must be loaded before enhancement")
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        logger.info(f"Enhancing {len(patches)} patches with batch size {batch_size}")
        
        enhanced_patches = []
        num_batches = (len(patches) + batch_size - 1) // batch_size
        
        try:
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(patches))
                    
                    # Get batch
                    batch_patches = patches[start_idx:end_idx]
                    
                    # Add channel dimension for U-Net (B, C, H, W)
                    batch_input = batch_patches.unsqueeze(1).to(self.device)
                    
                    try:
                        # Process batch
                        batch_output = self.model(batch_input)
                        
                        # Remove channel dimension and move to CPU
                        batch_enhanced = batch_output.squeeze(1).cpu()
                        enhanced_patches.append(batch_enhanced)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.error(f"GPU out of memory at batch {batch_idx + 1}")
                            # Try to recover by reducing batch size
                            if batch_size > 1:
                                logger.info("Attempting recovery with smaller batch size")
                                return self.enhance_patches(
                                    patches, 
                                    batch_size=max(1, batch_size // 2),
                                    progress_callback=progress_callback
                                )
                            else:
                                raise RuntimeError("GPU out of memory even with batch size 1") from e
                        else:
                            raise
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(batch_idx + 1, num_batches)
                    
                    # Log progress
                    if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                        logger.debug(f"Processed batch {batch_idx + 1}/{num_batches}")
            
            # Concatenate all enhanced patches
            enhanced_tensor = torch.cat(enhanced_patches, dim=0)
            
            logger.info(f"Enhanced {len(enhanced_tensor)} patches successfully")
            return enhanced_tensor
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            raise
    
    def enhance_volume_patches(
        self,
        patches: torch.Tensor,
        patch_infos: List,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> torch.Tensor:
        """
        Enhance volume patches with progress monitoring.
        
        Args:
            patches: Input patches tensor
            patch_infos: Patch position information
            progress_callback: Optional callback for progress updates
            
        Returns:
            Enhanced patches tensor
        """
        start_time = time.time()
        
        def batch_progress(batch_num: int, total_batches: int):
            progress = batch_num / total_batches
            if progress_callback:
                progress_callback(f"Processing batch {batch_num}/{total_batches}", progress)
        
        # Enhance patches
        enhanced_patches = self.enhance_patches(
            patches, 
            progress_callback=batch_progress
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Volume enhancement completed in {processing_time:.2f} seconds")
        
        if progress_callback:
            progress_callback("Enhancement completed", 1.0)
        
        return enhanced_patches
    
    def validate_model_compatibility(self, patch_size: Tuple[int, int]) -> bool:
        """
        Validate that the loaded model is compatible with the given patch size.
        
        Args:
            patch_size: Expected patch size (height, width)
            
        Returns:
            True if model is compatible
        """
        if not self.is_model_loaded:
            return False
        
        try:
            # Test with dummy input
            dummy_input = torch.randn(1, 1, *patch_size).to(self.device)
            
            with torch.no_grad():
                output = self.model(dummy_input)
            
            # Check output shape
            expected_shape = (1, 1) + patch_size
            if output.shape != expected_shape:
                logger.warning(
                    f"Model output shape {output.shape} doesn't match expected {expected_shape}"
                )
                return False
            
            logger.info(f"Model compatibility validated for patch size {patch_size}")
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded:
            return {'loaded': False}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Get model size in MB
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)
        
        return {
            'loaded': True,
            'device': self.device,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'architecture_valid': self.model_manager.validate_architecture(self.model)
        }
    
    def estimate_processing_time(
        self, 
        num_patches: int,
        batch_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate processing time for a given number of patches.
        
        Args:
            num_patches: Number of patches to process
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with time estimates
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        # Rough estimates based on typical performance
        if self.device == "cuda":
            patches_per_second = 100  # GPU estimate
        elif self.device == "mps":
            patches_per_second = 50   # Apple Silicon estimate
        else:
            patches_per_second = 10   # CPU estimate
        
        num_batches = (num_patches + batch_size - 1) // batch_size
        estimated_seconds = num_patches / patches_per_second
        
        return {
            'num_patches': num_patches,
            'num_batches': num_batches,
            'batch_size': batch_size,
            'estimated_seconds': estimated_seconds,
            'estimated_minutes': estimated_seconds / 60,
            'patches_per_second': patches_per_second,
            'device': self.device
        }
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_model_loaded = False
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Enhancement processor cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()