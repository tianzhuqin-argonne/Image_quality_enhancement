"""
PyTorch-based Patch Processing System for 3D Image Enhancement.

This module provides functionality for extracting and reconstructing patches from
large 3D image volumes, optimized for PyTorch tensor operations and GPU acceleration.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """Information about a patch's position and properties."""
    slice_idx: int
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    patch_size: Tuple[int, int]
    original_shape: Tuple[int, int]
    
    def __post_init__(self):
        """Validate patch information after initialization."""
        if self.end_row <= self.start_row or self.end_col <= self.start_col:
            raise ValueError("Invalid patch coordinates: end must be greater than start")
        
        actual_height = self.end_row - self.start_row
        actual_width = self.end_col - self.start_col
        expected_height, expected_width = self.patch_size
        
        if actual_height != expected_height or actual_width != expected_width:
            logger.warning(
                f"Patch size mismatch: expected {self.patch_size}, "
                f"got ({actual_height}, {actual_width})"
            )


class PatchProcessor:
    """
    PyTorch-based patch processor for 3D image volumes.
    
    This class handles the extraction and reconstruction of patches from large
    3D image volumes, with support for configurable patch sizes, overlap,
    and GPU acceleration through PyTorch tensors.
    
    Features:
    - Configurable patch sizes and overlap
    - PyTorch tensor support for GPU acceleration
    - Memory-efficient batch processing
    - Overlap blending for seamless reconstruction
    - Edge case handling for non-divisible dimensions
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (256, 256),
        overlap: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize the patch processor.
        
        Args:
            patch_size: Size of patches (height, width)
            overlap: Overlap between adjacent patches in pixels
            device: PyTorch device ('cpu', 'cuda', etc.)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PatchProcessor")
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = torch.device(device)
        
        # Validate parameters
        if overlap >= min(patch_size):
            raise ValueError("Overlap must be smaller than patch size")
        
        if overlap < 0:
            raise ValueError("Overlap must be non-negative")
        
        # Calculate effective stride (patch size minus overlap)
        self.stride = (patch_size[0] - overlap, patch_size[1] - overlap)
        
        logger.info(
            f"Initialized PatchProcessor: patch_size={patch_size}, "
            f"overlap={overlap}, stride={self.stride}, device={device}"
        )
    
    def calculate_patch_positions(
        self, 
        slice_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate patch positions for a given slice shape.
        
        Args:
            slice_shape: Shape of the slice (height, width)
            
        Returns:
            List of patch positions as (start_row, start_col, end_row, end_col)
        """
        height, width = slice_shape
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride
        
        positions = []
        
        # Calculate number of patches needed
        num_patches_h = max(1, (height - patch_height) // stride_height + 1)
        num_patches_w = max(1, (width - patch_width) // stride_width + 1)
        
        # Handle edge case where image is smaller than patch
        if height < patch_height or width < patch_width:
            logger.warning(
                f"Image size {slice_shape} is smaller than patch size {self.patch_size}. "
                "Using single patch with padding."
            )
            positions.append((0, 0, height, width))
            return positions
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Calculate patch start position
                start_row = i * stride_height
                start_col = j * stride_width
                
                # Calculate patch end position
                end_row = min(start_row + patch_height, height)
                end_col = min(start_col + patch_width, width)
                
                # Adjust start position if we're at the edge
                if end_row == height and end_row - start_row < patch_height:
                    start_row = max(0, height - patch_height)
                if end_col == width and end_col - start_col < patch_width:
                    start_col = max(0, width - patch_width)
                
                # Recalculate end positions after adjustment
                end_row = min(start_row + patch_height, height)
                end_col = min(start_col + patch_width, width)
                
                positions.append((start_row, start_col, end_row, end_col))
        
        logger.debug(
            f"Calculated {len(positions)} patch positions for shape {slice_shape}"
        )
        return positions
    
    def extract_patches(
        self, 
        volume: Union[np.ndarray, torch.Tensor],
        slice_indices: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, List[PatchInfo]]:
        """
        Extract patches from a 3D volume.
        
        Args:
            volume: 3D volume with shape (slices, height, width)
            slice_indices: Optional list of slice indices to process (default: all)
            
        Returns:
            Tuple of (patches_tensor, patch_info_list)
            - patches_tensor: Tensor with shape (num_patches, patch_height, patch_width)
            - patch_info_list: List of PatchInfo objects with position information
        """
        # Convert to PyTorch tensor if needed
        if isinstance(volume, np.ndarray):
            volume_tensor = torch.from_numpy(volume).to(self.device)
        else:
            volume_tensor = volume.to(self.device)
        
        if volume_tensor.dim() != 3:
            raise ValueError(f"Expected 3D volume, got {volume_tensor.dim()}D")
        
        num_slices, height, width = volume_tensor.shape
        
        # Determine which slices to process
        if slice_indices is None:
            slice_indices = list(range(num_slices))
        
        # Validate slice indices
        for idx in slice_indices:
            if idx < 0 or idx >= num_slices:
                raise ValueError(f"Slice index {idx} out of range [0, {num_slices-1}]")
        
        # Calculate patch positions
        patch_positions = self.calculate_patch_positions((height, width))
        
        # Extract patches
        patches = []
        patch_infos = []
        
        for slice_idx in slice_indices:
            slice_data = volume_tensor[slice_idx]
            
            for start_row, start_col, end_row, end_col in patch_positions:
                # Extract patch
                patch = slice_data[start_row:end_row, start_col:end_col]
                
                # Pad patch if necessary (for edge cases)
                if patch.shape != self.patch_size:
                    patch = self._pad_patch(patch, self.patch_size)
                
                patches.append(patch)
                
                # Create patch info
                patch_info = PatchInfo(
                    slice_idx=slice_idx,
                    start_row=start_row,
                    start_col=start_col,
                    end_row=end_row,
                    end_col=end_col,
                    patch_size=self.patch_size,
                    original_shape=(height, width)
                )
                patch_infos.append(patch_info)
        
        # Stack patches into a single tensor
        patches_tensor = torch.stack(patches, dim=0)
        
        logger.info(
            f"Extracted {len(patches)} patches from {len(slice_indices)} slices. "
            f"Patches tensor shape: {patches_tensor.shape}"
        )
        
        return patches_tensor, patch_infos
    
    def _pad_patch(self, patch: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Pad a patch to the target size.
        
        Args:
            patch: Input patch tensor
            target_size: Target size (height, width)
            
        Returns:
            Padded patch tensor
        """
        current_h, current_w = patch.shape
        target_h, target_w = target_size
        
        if current_h > target_h or current_w > target_w:
            raise ValueError("Patch is larger than target size")
        
        # Calculate padding
        pad_h = target_h - current_h
        pad_w = target_w - current_w
        
        # Pad with constant padding (zeros) to avoid compatibility issues
        # PyTorch padding format: (left, right, top, bottom)
        padding = (0, pad_w, 0, pad_h)
        padded_patch = F.pad(patch, padding, mode='constant', value=0)
        
        return padded_patch
    
    def reconstruct_slice(
        self,
        patches: torch.Tensor,
        patch_infos: List[PatchInfo],
        target_slice_idx: int,
        output_shape: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Reconstruct a slice from patches with overlap blending.
        
        Args:
            patches: Tensor of patches with shape (num_patches, patch_height, patch_width)
            patch_infos: List of PatchInfo objects corresponding to patches
            target_slice_idx: Index of the slice to reconstruct
            output_shape: Optional output shape (height, width)
            
        Returns:
            Reconstructed slice tensor
        """
        if len(patches) != len(patch_infos):
            raise ValueError("Number of patches must match number of patch infos")
        
        # Filter patches for the target slice
        slice_patches = []
        slice_infos = []
        
        for i, info in enumerate(patch_infos):
            if info.slice_idx == target_slice_idx:
                slice_patches.append(patches[i])
                slice_infos.append(info)
        
        if not slice_patches:
            raise ValueError(f"No patches found for slice {target_slice_idx}")
        
        # Determine output shape
        if output_shape is None:
            # Use the original shape from the first patch info
            output_shape = slice_infos[0].original_shape
        
        height, width = output_shape
        
        # Initialize output tensors
        reconstructed = torch.zeros(height, width, device=self.device, dtype=patches.dtype)
        weight_map = torch.zeros(height, width, device=self.device, dtype=torch.float32)
        
        # Create blending weights for overlap regions
        patch_height, patch_width = self.patch_size
        blend_weights = self._create_blend_weights(patch_height, patch_width)
        blend_weights = blend_weights.to(self.device)
        
        # Reconstruct slice by blending patches
        for patch, info in zip(slice_patches, slice_infos):
            start_row, start_col = info.start_row, info.start_col
            end_row, end_col = info.end_row, info.end_col
            
            # Handle patches that might have been padded
            actual_height = end_row - start_row
            actual_width = end_col - start_col
            
            # Extract the relevant portion of the patch (remove padding if any)
            patch_portion = patch[:actual_height, :actual_width]
            weight_portion = blend_weights[:actual_height, :actual_width]
            
            # Add to reconstruction with blending
            reconstructed[start_row:end_row, start_col:end_col] += (
                patch_portion * weight_portion
            )
            weight_map[start_row:end_row, start_col:end_col] += weight_portion
        
        # Normalize by weight map to handle overlaps
        # Avoid division by zero
        weight_map = torch.clamp(weight_map, min=1e-8)
        reconstructed = reconstructed / weight_map
        
        logger.debug(f"Reconstructed slice {target_slice_idx} with shape {reconstructed.shape}")
        
        return reconstructed
    
    def _create_blend_weights(self, height: int, width: int) -> torch.Tensor:
        """
        Create blending weights for smooth patch reconstruction.
        
        Args:
            height: Patch height
            width: Patch width
            
        Returns:
            Weight tensor with smooth transitions at edges
        """
        if self.overlap == 0:
            # No overlap, use uniform weights
            return torch.ones(height, width)
        
        # Create weight map with smooth transitions
        weights = torch.ones(height, width)
        
        # Apply smooth transitions at edges based on overlap
        fade_size = min(self.overlap // 2, height // 8, width // 8, 16)  # Limit fade size
        
        if fade_size > 1:
            # Create 1D fade profiles
            fade_profile = torch.linspace(0.1, 1.0, fade_size)  # Start from 0.1 instead of 0
            
            # Apply fade to edges more conservatively
            # Top edge
            weights[:fade_size, :] *= fade_profile.unsqueeze(1)
            # Bottom edge
            weights[-fade_size:, :] *= fade_profile.flip(0).unsqueeze(1)
            # Left edge
            weights[:, :fade_size] *= fade_profile.unsqueeze(0)
            # Right edge
            weights[:, -fade_size:] *= fade_profile.flip(0).unsqueeze(0)
        
        return weights
    
    def reconstruct_volume(
        self,
        patches: torch.Tensor,
        patch_infos: List[PatchInfo],
        output_shape: Optional[Tuple[int, int, int]] = None
    ) -> torch.Tensor:
        """
        Reconstruct a complete 3D volume from patches.
        
        Args:
            patches: Tensor of patches with shape (num_patches, patch_height, patch_width)
            patch_infos: List of PatchInfo objects corresponding to patches
            output_shape: Optional output shape (slices, height, width)
            
        Returns:
            Reconstructed 3D volume tensor
        """
        if not patch_infos:
            raise ValueError("No patch information provided")
        
        # Determine output shape
        if output_shape is None:
            # Infer from patch infos
            max_slice = max(info.slice_idx for info in patch_infos)
            original_shape = patch_infos[0].original_shape
            output_shape = (max_slice + 1, original_shape[0], original_shape[1])
        
        num_slices, height, width = output_shape
        
        # Get unique slice indices
        slice_indices = sorted(set(info.slice_idx for info in patch_infos))
        
        # Reconstruct each slice
        reconstructed_slices = []
        
        for slice_idx in range(num_slices):
            if slice_idx in slice_indices:
                # Reconstruct slice from patches
                reconstructed_slice = self.reconstruct_slice(
                    patches, patch_infos, slice_idx, (height, width)
                )
            else:
                # Create empty slice if no patches available
                reconstructed_slice = torch.zeros(
                    height, width, device=self.device, dtype=patches.dtype
                )
                logger.warning(f"No patches found for slice {slice_idx}, using zeros")
            
            reconstructed_slices.append(reconstructed_slice)
        
        # Stack slices into volume
        reconstructed_volume = torch.stack(reconstructed_slices, dim=0)
        
        logger.info(
            f"Reconstructed volume with shape {reconstructed_volume.shape} "
            f"from {len(patches)} patches"
        )
        
        return reconstructed_volume
    
    def get_memory_usage_estimate(
        self, 
        volume_shape: Tuple[int, int, int],
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, float]:
        """
        Estimate memory usage for patch processing.
        
        Args:
            volume_shape: Shape of the input volume (slices, height, width)
            dtype: Data type of tensors
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        num_slices, height, width = volume_shape
        
        # Calculate number of patches per slice
        positions = self.calculate_patch_positions((height, width))
        patches_per_slice = len(positions)
        total_patches = patches_per_slice * num_slices
        
        # Calculate memory usage
        bytes_per_element = torch.tensor(0, dtype=dtype).element_size()
        
        # Original volume
        volume_mb = (num_slices * height * width * bytes_per_element) / (1024 ** 2)
        
        # Patches tensor
        patch_elements = total_patches * self.patch_size[0] * self.patch_size[1]
        patches_mb = (patch_elements * bytes_per_element) / (1024 ** 2)
        
        # Reconstruction buffers (per slice)
        reconstruction_mb = (height * width * bytes_per_element * 2) / (1024 ** 2)  # data + weights
        
        return {
            'volume_mb': volume_mb,
            'patches_mb': patches_mb,
            'reconstruction_mb': reconstruction_mb,
            'total_estimated_mb': volume_mb + patches_mb + reconstruction_mb,
            'patches_per_slice': patches_per_slice,
            'total_patches': total_patches
        }