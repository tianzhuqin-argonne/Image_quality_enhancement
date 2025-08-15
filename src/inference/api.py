"""
Image Enhancement API for 3D TIFF processing.

This module provides a high-level API for enhancing 3D TIFF images using
trained U-Net models. Designed for easy integration into existing codebases.
"""

import logging
from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path
import time
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .image_processor import ImageProcessor
from .enhancement_processor import EnhancementProcessor
from ..core.config import InferenceConfig, EnhancementResult
from ..core.data_models import SliceStack

logger = logging.getLogger(__name__)


class ImageEnhancementAPI:
    """
    High-level API for 3D image enhancement using trained U-Net models.
    
    Provides simple methods for enhancing 3D TIFF files and numpy arrays
    with automatic memory management, progress monitoring, and quality assessment.
    
    Features:
    - File-based and in-memory processing
    - Automatic memory management and chunking
    - Progress monitoring with callbacks
    - Quality metrics calculation
    - Runtime configuration updates
    - Comprehensive error handling
    - CPU/GPU flexibility
    
    Example:
        ```python
        from src.inference.api import ImageEnhancementAPI
        from src.core.config import InferenceConfig
        
        # Initialize API
        config = InferenceConfig(device='auto', memory_limit_gb=4.0)
        api = ImageEnhancementAPI(config)
        
        # Load model
        api.load_model('path/to/trained_model.pth')
        
        # Enhance TIFF file
        result = api.enhance_3d_tiff('input.tif', 'enhanced_output.tif')
        
        if result.success:
            print(f"Enhancement completed in {result.processing_time:.2f}s")
            print(f"Quality metrics: {result.quality_metrics}")
        ```
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the Image Enhancement API.
        
        Args:
            config: Optional inference configuration (uses defaults if None)
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, API will have limited functionality")
        
        self.config = config or InferenceConfig()
        
        # Initialize processors
        self.image_processor = ImageProcessor(self.config)
        self.enhancement_processor = None
        
        # State
        self.model_loaded = False
        self.model_path = None
        
        logger.info(f"ImageEnhancementAPI initialized with device: {self.image_processor.device}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model for enhancement.
        
        Args:
            model_path: Path to the trained model file
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not TORCH_AVAILABLE:
                logger.error("PyTorch not available, cannot load model")
                return False
            
            # Initialize enhancement processor if needed
            if self.enhancement_processor is None:
                self.enhancement_processor = EnhancementProcessor(self.config)
            
            # Load model
            self.enhancement_processor.load_model(model_path)
            
            # Validate compatibility
            if not self.enhancement_processor.validate_model_compatibility(self.config.patch_size):
                logger.error(f"Model incompatible with patch size {self.config.patch_size}")
                return False
            
            self.model_loaded = True
            self.model_path = model_path
            
            # Log model info
            model_info = self.enhancement_processor.get_model_info()
            logger.info(f"Model loaded: {model_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def enhance_3d_tiff(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        calculate_metrics: bool = None
    ) -> EnhancementResult:
        """
        Enhance a 3D TIFF file and save the result.
        
        Args:
            input_path: Path to input TIFF file
            output_path: Path to save enhanced TIFF file
            progress_callback: Optional callback for progress updates
            calculate_metrics: Whether to calculate quality metrics (default from config)
            
        Returns:
            EnhancementResult with processing information and metrics
        """
        start_time = time.time()
        result = EnhancementResult(
            success=False,
            processing_time=0.0,
            input_shape=(0, 0, 0),
            output_path=output_path
        )
        
        try:
            # Validate inputs
            if not Path(input_path).exists():
                result.error_message = f"Input file not found: {input_path}"
                return result
            
            if not self.model_loaded:
                result.error_message = "No model loaded. Call load_model() first."
                return result
            
            if progress_callback:
                progress_callback("Loading input volume", 0.1)
            
            # Load input volume
            logger.info(f"Loading input volume: {input_path}")
            input_volume = self.image_processor.load_3d_volume(input_path)
            result.input_shape = input_volume.shape
            
            # Check memory constraints
            if not self.image_processor.check_memory_constraints(input_volume.shape):
                result.add_warning("Volume may exceed memory limits, using chunked processing")
            
            if progress_callback:
                progress_callback("Preprocessing volume", 0.2)
            
            # Preprocess volume
            preprocessed_volume = self.image_processor.preprocess_volume(input_volume)
            
            # Enhance volume
            enhanced_volume = self._enhance_volume_with_chunking(
                preprocessed_volume, progress_callback
            )
            
            if progress_callback:
                progress_callback("Postprocessing volume", 0.9)
            
            # Postprocess volume
            final_volume = self.image_processor.postprocess_volume(enhanced_volume, preprocessed_volume)
            
            # Preserve metadata if configured
            if self.config.preserve_metadata:
                final_volume.metadata.update(input_volume.metadata)
                final_volume.metadata['enhancement_info'] = {
                    'model_path': self.model_path,
                    'processing_time': time.time() - start_time,
                    'patch_size': self.config.patch_size,
                    'device': self.image_processor.device
                }
            
            # Save enhanced volume
            self.image_processor.save_volume(final_volume, output_path)
            
            # Calculate quality metrics if requested
            if calculate_metrics is None:
                calculate_metrics = self.config.enable_quality_metrics
            
            if calculate_metrics:
                if progress_callback:
                    progress_callback("Calculating quality metrics", 0.95)
                
                metrics = self.image_processor.get_volume_quality_metrics(
                    final_volume, input_volume
                )
                result.quality_metrics = metrics
            
            # Success
            result.success = True
            result.processing_time = time.time() - start_time
            
            if progress_callback:
                progress_callback("Enhancement completed", 1.0)
            
            logger.info(f"Enhancement completed successfully in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Enhancement failed: {e}")
        
        return result
    
    def enhance_3d_array(
        self,
        input_array: np.ndarray,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        calculate_metrics: bool = None
    ) -> EnhancementResult:
        """
        Enhance a 3D numpy array in memory.
        
        Args:
            input_array: Input 3D numpy array (slices, height, width)
            progress_callback: Optional callback for progress updates
            calculate_metrics: Whether to calculate quality metrics (default from config)
            
        Returns:
            EnhancementResult with enhanced array and processing information
        """
        start_time = time.time()
        result = EnhancementResult(
            success=False,
            processing_time=0.0,
            input_shape=input_array.shape
        )
        
        try:
            # Validate inputs
            if input_array.ndim != 3:
                result.error_message = f"Input array must be 3D, got {input_array.ndim}D"
                return result
            
            if not self.model_loaded:
                result.error_message = "No model loaded. Call load_model() first."
                return result
            
            if progress_callback:
                progress_callback("Creating volume from array", 0.1)
            
            # Create SliceStack from array
            input_volume = SliceStack(input_array)
            
            # Check memory constraints
            if not self.image_processor.check_memory_constraints(input_volume.shape):
                result.add_warning("Volume may exceed memory limits, using chunked processing")
            
            if progress_callback:
                progress_callback("Preprocessing volume", 0.2)
            
            # Preprocess volume
            preprocessed_volume = self.image_processor.preprocess_volume(input_volume)
            
            # Enhance volume
            enhanced_volume = self._enhance_volume_with_chunking(
                preprocessed_volume, progress_callback
            )
            
            if progress_callback:
                progress_callback("Postprocessing volume", 0.9)
            
            # Postprocess volume
            final_volume = self.image_processor.postprocess_volume(enhanced_volume, preprocessed_volume)
            
            # Calculate quality metrics if requested
            if calculate_metrics is None:
                calculate_metrics = self.config.enable_quality_metrics
            
            if calculate_metrics:
                if progress_callback:
                    progress_callback("Calculating quality metrics", 0.95)
                
                metrics = self.image_processor.get_volume_quality_metrics(
                    final_volume, input_volume
                )
                result.quality_metrics = metrics
            
            # Success - store enhanced array in metadata
            result.success = True
            result.processing_time = time.time() - start_time
            result.quality_metrics['enhanced_array'] = final_volume.to_numpy()
            
            if progress_callback:
                progress_callback("Enhancement completed", 1.0)
            
            logger.info(f"Array enhancement completed successfully in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Array enhancement failed: {e}")
        
        return result
    
    def _enhance_volume_with_chunking(
        self,
        volume: SliceStack,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> SliceStack:
        """
        Enhance volume with automatic chunking for memory management.
        
        Args:
            volume: Input volume to enhance
            progress_callback: Optional progress callback
            
        Returns:
            Enhanced volume
        """
        # Check if chunking is needed
        if self.image_processor.check_memory_constraints(volume.shape):
            # Process entire volume at once
            return self._enhance_volume_direct(volume, progress_callback)
        else:
            # Process in chunks
            return self._enhance_volume_chunked(volume, progress_callback)
    
    def _enhance_volume_direct(
        self,
        volume: SliceStack,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> SliceStack:
        """Enhance entire volume directly."""
        if progress_callback:
            progress_callback("Extracting patches", 0.3)
        
        # Extract patches
        patches, patch_infos = self.image_processor.extract_patches_for_inference(volume)
        
        if progress_callback:
            progress_callback("Enhancing patches", 0.5)
        
        # Enhance patches
        enhanced_patches = self.enhancement_processor.enhance_volume_patches(
            patches, patch_infos, progress_callback
        )
        
        if progress_callback:
            progress_callback("Reconstructing volume", 0.8)
        
        # Reconstruct volume
        enhanced_volume = self.image_processor.reconstruct_volume_from_patches(
            enhanced_patches, patch_infos, volume.shape
        )
        
        return enhanced_volume
    
    def _enhance_volume_chunked(
        self,
        volume: SliceStack,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> SliceStack:
        """Enhance volume in chunks for memory efficiency."""
        logger.info("Using chunked processing for large volume")
        
        # Get processing chunks
        chunks = self.image_processor.get_processing_chunks(volume.shape)
        
        enhanced_slices = []
        
        for chunk_idx, (start_slice, end_slice) in enumerate(chunks):
            if progress_callback:
                chunk_progress = 0.3 + (chunk_idx / len(chunks)) * 0.5
                progress_callback(f"Processing chunk {chunk_idx + 1}/{len(chunks)}", chunk_progress)
            
            # Extract chunk
            chunk_slices = list(range(start_slice, end_slice))
            chunk_volume = SliceStack(volume.get_slices(chunk_slices))
            
            # Process chunk
            enhanced_chunk = self._enhance_volume_direct(chunk_volume)
            enhanced_slices.extend([enhanced_chunk.get_slice(i) for i in range(enhanced_chunk.num_slices)])
        
        # Combine enhanced slices
        if TORCH_AVAILABLE:
            # Convert to tensors if needed
            tensor_slices = []
            for slice_data in enhanced_slices:
                if isinstance(slice_data, torch.Tensor):
                    tensor_slices.append(slice_data)
                else:
                    tensor_slices.append(torch.from_numpy(slice_data))
            
            enhanced_data = torch.stack(tensor_slices, dim=0)
            enhanced_volume = SliceStack(enhanced_data.numpy())
        else:
            # Convert to numpy arrays if needed
            numpy_slices = []
            for slice_data in enhanced_slices:
                if isinstance(slice_data, np.ndarray):
                    numpy_slices.append(slice_data)
                else:
                    numpy_slices.append(slice_data.numpy())
            
            enhanced_data = np.stack(numpy_slices, axis=0)
            enhanced_volume = SliceStack(enhanced_data)
        
        return enhanced_volume
    
    def get_enhancement_metrics(self, result: EnhancementResult) -> Dict[str, Any]:
        """
        Get comprehensive enhancement metrics from a result.
        
        Args:
            result: Enhancement result to analyze
            
        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {
            'success': result.success,
            'processing_time': result.processing_time,
            'input_shape': result.input_shape,
            'output_path': result.output_path,
            'warnings': result.warnings,
            'error_message': result.error_message
        }
        
        # Add quality metrics if available
        if result.quality_metrics:
            metrics['quality_metrics'] = result.quality_metrics
        
        # Add processing efficiency metrics
        if result.success and result.input_shape[0] > 0:
            total_pixels = np.prod(result.input_shape)
            metrics['processing_efficiency'] = {
                'pixels_per_second': total_pixels / result.processing_time,
                'slices_per_second': result.input_shape[0] / result.processing_time,
                'megapixels_processed': total_pixels / 1e6
            }
        
        # Add model information if available
        if self.model_loaded and self.enhancement_processor:
            model_info = self.enhancement_processor.get_model_info()
            metrics['model_info'] = model_info
        
        return metrics
    
    def set_processing_config(self, **kwargs) -> None:
        """
        Update processing configuration at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
                     (batch_size, memory_limit_gb, device, etc.)
        """
        # Update config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
        
        # Reinitialize processors if needed
        if 'device' in kwargs or 'memory_limit_gb' in kwargs:
            logger.info("Reinitializing processors due to device/memory config change")
            self.image_processor = ImageProcessor(self.config)
            
            if self.model_loaded and self.enhancement_processor:
                # Reload model on new device
                model_path = self.model_path
                self.enhancement_processor = EnhancementProcessor(self.config)
                self.load_model(model_path)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for debugging and optimization.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'torch_available': TORCH_AVAILABLE,
            'device': self.image_processor.device,
            'config': {
                'patch_size': self.config.patch_size,
                'overlap': self.config.overlap,
                'batch_size': self.config.batch_size,
                'memory_limit_gb': self.config.memory_limit_gb,
                'enable_quality_metrics': self.config.enable_quality_metrics
            },
            'model_loaded': self.model_loaded,
            'model_path': self.model_path
        }
        
        if TORCH_AVAILABLE:
            info['torch_info'] = {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if hasattr(torch.backends, 'mps'):
                info['torch_info']['mps_available'] = torch.backends.mps.is_available()
        
        return info
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        if self.enhancement_processor:
            self.enhancement_processor.cleanup()
        
        self.model_loaded = False
        self.model_path = None
        
        logger.info("ImageEnhancementAPI cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()