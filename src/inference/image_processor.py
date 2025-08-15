"""
Image processing components for 3D volume handling in inference pipeline.

This module provides efficient processing of 3D image volumes with memory
optimization and batch processing capabilities.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.data_models import SliceStack
from ..core.tiff_handler import TIFFDataHandler
from ..core.patch_processor import PatchProcessor
from ..core.config import InferenceConfig

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Core image processor for 3D volume handling in inference pipeline.
    
    Handles loading, preprocessing, and memory-efficient processing of large
    3D image volumes for enhancement. Designed for production inference with
    automatic memory management and CPU/GPU flexibility.
    
    Features:
    - Memory-efficient batch processing
    - Automatic device detection with CPU fallback
    - Large volume handling with chunking
    - Preprocessing and normalization
    - Quality validation and metrics
    """
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize image processor.
        
        Args:
            config: Inference configuration
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU-only mode")
        
        self.config = config
        
        # Resolve device
        self.device = self._resolve_device(config.device)
        logger.info(f"ImageProcessor initialized with device: {self.device}")
        
        # Initialize components
        self.tiff_handler = TIFFDataHandler()
        self.patch_processor = PatchProcessor(
            patch_size=config.patch_size,
            overlap=config.overlap,
            device=self.device
        )
        
        # Memory management
        self.memory_limit_bytes = int(config.memory_limit_gb * 1024**3)
        
    def _resolve_device(self, device_config: str) -> str:
        """Resolve device configuration to actual device string."""
        if not TORCH_AVAILABLE:
            return "cpu"
            
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        else:
            return device_config
    
    def load_3d_volume(self, file_path: str) -> SliceStack:
        """
        Load 3D volume from TIFF file with validation.
        
        Args:
            file_path: Path to TIFF file
            
        Returns:
            SliceStack containing the loaded volume
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        logger.info(f"Loading 3D volume: {file_path}")
        
        try:
            # Load volume
            volume = SliceStack.from_tiff_handler(self.tiff_handler, file_path)
            
            # Validate dimensions
            if not self.tiff_handler.validate_dimensions(volume.to_numpy()):
                logger.warning(f"Volume dimensions may be outside optimal range: {volume.shape}")
            
            # Log volume statistics
            stats = volume.get_statistics()
            logger.info(f"Loaded volume: {stats}")
            
            return volume
            
        except Exception as e:
            logger.error(f"Failed to load volume from {file_path}: {e}")
            raise
    
    def preprocess_volume(self, volume: SliceStack) -> SliceStack:
        """
        Preprocess volume for inference.
        
        Args:
            volume: Input volume
            
        Returns:
            Preprocessed volume
        """
        logger.debug("Preprocessing volume for inference")
        
        # Convert to tensor for processing
        if TORCH_AVAILABLE:
            data = volume.to_tensor(self.device)
            
            # Normalize to [0, 1] range
            data_min = data.min()
            data_max = data.max()
            
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            
            # Convert back to SliceStack
            preprocessed_volume = SliceStack(data.cpu().numpy())
            
        else:
            # CPU-only preprocessing
            data = volume.to_numpy()
            
            # Normalize to [0, 1] range
            data_min = data.min()
            data_max = data.max()
            
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            
            preprocessed_volume = SliceStack(data)
        
        # Store normalization parameters for post-processing
        preprocessed_volume.metadata['normalization'] = {
            'min_value': float(data_min),
            'max_value': float(data_max)
        }
        
        return preprocessed_volume
    
    def postprocess_volume(self, volume: SliceStack, original_volume: SliceStack) -> SliceStack:
        """
        Postprocess enhanced volume to restore original value range.
        
        Args:
            volume: Enhanced volume
            original_volume: Original volume for reference
            
        Returns:
            Postprocessed volume
        """
        logger.debug("Postprocessing enhanced volume")
        
        # Get normalization parameters
        norm_params = original_volume.metadata.get('normalization', {})
        
        if norm_params:
            data_min = norm_params['min_value']
            data_max = norm_params['max_value']
            
            if TORCH_AVAILABLE:
                data = volume.to_tensor()
                # Denormalize
                data = data * (data_max - data_min) + data_min
                postprocessed_volume = SliceStack(data.cpu().numpy())
            else:
                data = volume.to_numpy()
                # Denormalize
                data = data * (data_max - data_min) + data_min
                postprocessed_volume = SliceStack(data)
        else:
            postprocessed_volume = volume
        
        # Preserve original metadata
        postprocessed_volume.metadata.update(original_volume.metadata)
        
        return postprocessed_volume
    
    def extract_patches_for_inference(
        self, 
        volume: SliceStack,
        slice_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List]:
        """
        Extract patches from volume for inference processing.
        
        Args:
            volume: Input volume
            slice_indices: Optional slice indices to process
            
        Returns:
            Tuple of (patches_tensor, patch_info_list)
        """
        logger.debug(f"Extracting patches for inference from volume shape: {volume.shape}")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for patch extraction")
        
        # Convert to tensor
        volume_tensor = volume.to_tensor(self.device)
        
        # Extract patches
        patches, patch_infos = self.patch_processor.extract_patches(
            volume_tensor, slice_indices
        )
        
        logger.info(f"Extracted {len(patches)} patches for inference")
        return patches, patch_infos
    
    def reconstruct_volume_from_patches(
        self,
        patches: torch.Tensor,
        patch_infos: List,
        original_shape: Tuple[int, int, int]
    ) -> SliceStack:
        """
        Reconstruct volume from enhanced patches.
        
        Args:
            patches: Enhanced patches tensor
            patch_infos: Patch position information
            original_shape: Original volume shape
            
        Returns:
            Reconstructed volume
        """
        logger.debug(f"Reconstructing volume from {len(patches)} patches")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for volume reconstruction")
        
        # Reconstruct volume
        reconstructed_tensor = self.patch_processor.reconstruct_volume(
            patches, patch_infos, original_shape
        )
        
        # Convert to SliceStack
        reconstructed_volume = SliceStack(reconstructed_tensor.cpu().numpy())
        
        logger.info(f"Reconstructed volume with shape: {reconstructed_volume.shape}")
        return reconstructed_volume
    
    def estimate_memory_usage(self, volume_shape: Tuple[int, int, int]) -> Dict[str, float]:
        """
        Estimate memory usage for processing a volume.
        
        Args:
            volume_shape: Shape of the volume to process
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        if TORCH_AVAILABLE:
            return self.patch_processor.get_memory_usage_estimate(volume_shape)
        else:
            # Simple CPU memory estimation
            num_slices, height, width = volume_shape
            volume_mb = (num_slices * height * width * 4) / (1024 ** 2)  # float32
            
            return {
                'volume_mb': volume_mb,
                'patches_mb': volume_mb * 2,  # Rough estimate
                'reconstruction_mb': volume_mb,
                'total_estimated_mb': volume_mb * 4,
                'patches_per_slice': 1,
                'total_patches': num_slices
            }
    
    def check_memory_constraints(self, volume_shape: Tuple[int, int, int]) -> bool:
        """
        Check if volume can be processed within memory constraints.
        
        Args:
            volume_shape: Shape of the volume to process
            
        Returns:
            True if processing is feasible within memory limits
        """
        memory_estimate = self.estimate_memory_usage(volume_shape)
        estimated_mb = memory_estimate['total_estimated_mb']
        limit_mb = self.memory_limit_bytes / (1024 ** 2)
        
        feasible = estimated_mb <= limit_mb
        
        if not feasible:
            logger.warning(
                f"Volume may exceed memory limit: {estimated_mb:.1f}MB estimated, "
                f"{limit_mb:.1f}MB limit"
            )
        
        return feasible
    
    def get_processing_chunks(
        self, 
        volume_shape: Tuple[int, int, int],
        max_slices_per_chunk: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Calculate optimal processing chunks for large volumes.
        
        Args:
            volume_shape: Shape of the volume
            max_slices_per_chunk: Maximum slices per chunk
            
        Returns:
            List of (start_slice, end_slice) tuples
        """
        num_slices = volume_shape[0]
        
        if max_slices_per_chunk is None:
            # Estimate based on memory constraints
            memory_per_slice = self.estimate_memory_usage((1,) + volume_shape[1:])['total_estimated_mb']
            limit_mb = self.memory_limit_bytes / (1024 ** 2)
            max_slices_per_chunk = max(1, int(limit_mb / memory_per_slice * 0.8))  # 80% safety margin
        
        chunks = []
        for start in range(0, num_slices, max_slices_per_chunk):
            end = min(start + max_slices_per_chunk, num_slices)
            chunks.append((start, end))
        
        logger.info(f"Processing volume in {len(chunks)} chunks, max {max_slices_per_chunk} slices per chunk")
        return chunks
    
    def save_volume(self, volume: SliceStack, output_path: str) -> None:
        """
        Save processed volume to TIFF file.
        
        Args:
            volume: Volume to save
            output_path: Output file path
        """
        logger.info(f"Saving volume to: {output_path}")
        
        try:
            volume.save_with_tiff_handler(self.tiff_handler, output_path)
            logger.info(f"Successfully saved volume: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save volume to {output_path}: {e}")
            raise
    
    def get_volume_quality_metrics(
        self, 
        enhanced_volume: SliceStack,
        original_volume: Optional[SliceStack] = None
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for enhanced volume.
        
        Args:
            enhanced_volume: Enhanced volume
            original_volume: Optional original volume for comparison
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Basic statistics
        stats = enhanced_volume.get_statistics()
        metrics.update({
            'mean_intensity': stats['mean_value'],
            'std_intensity': stats['std_value'],
            'min_intensity': stats['min_value'],
            'max_intensity': stats['max_value'],
            'dynamic_range': stats['max_value'] - stats['min_value']
        })
        
        # Comparison metrics if original is provided
        if original_volume is not None:
            if TORCH_AVAILABLE:
                enhanced_data = enhanced_volume.to_tensor()
                original_data = original_volume.to_tensor()
                
                # MSE and PSNR
                mse = torch.mean((enhanced_data - original_data) ** 2).item()
                if mse > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                else:
                    psnr = float('inf')
                
                metrics.update({
                    'mse': mse,
                    'psnr': psnr,
                    'mae': torch.mean(torch.abs(enhanced_data - original_data)).item()
                })
            else:
                enhanced_data = enhanced_volume.to_numpy()
                original_data = original_volume.to_numpy()
                
                # MSE and PSNR
                mse = np.mean((enhanced_data - original_data) ** 2)
                if mse > 0:
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse))
                else:
                    psnr = float('inf')
                
                metrics.update({
                    'mse': float(mse),
                    'psnr': float(psnr),
                    'mae': float(np.mean(np.abs(enhanced_data - original_data)))
                })
        
        return metrics