"""
Data models for 3D image enhancement system.

This module provides data structures for managing multi-slice image volumes
that are processed slice-by-slice using 2D U-Net architecture.

Key concepts:
- SliceStack: Manages a stack of 2D slices (what was called "3D volume")
- Patch2D: Represents a 2D patch extracted from a slice
- TrainingPair2D: Pairs of input/target patches for 2D U-Net training
"""

import logging
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Patch2D:
    """
    Represents a 2D patch extracted from an image slice.
    
    This class encapsulates a 2D patch with its position information,
    making it suitable for 2D U-Net processing while maintaining
    spatial context for reconstruction.
    """
    data: Union[np.ndarray, 'torch.Tensor']
    slice_idx: int
    start_row: int
    start_col: int
    patch_size: Tuple[int, int]
    original_slice_shape: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate patch data after initialization."""
        if isinstance(self.data, np.ndarray):
            if self.data.ndim != 2:
                raise ValueError(f"Patch data must be 2D, got {self.data.ndim}D")
        elif TORCH_AVAILABLE and isinstance(self.data, torch.Tensor):
            if self.data.dim() != 2:
                raise ValueError(f"Patch tensor must be 2D, got {self.data.dim()}D")
        else:
            raise ValueError("Patch data must be numpy array or PyTorch tensor")
        
        # Validate coordinates
        if self.start_row < 0 or self.start_col < 0:
            raise ValueError("Patch coordinates must be non-negative")
        
        if self.slice_idx < 0:
            raise ValueError("Slice index must be non-negative")
    
    @property
    def end_row(self) -> int:
        """End row coordinate of the patch."""
        return self.start_row + self.patch_size[0]
    
    @property
    def end_col(self) -> int:
        """End column coordinate of the patch."""
        return self.start_col + self.patch_size[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of the patch data."""
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        else:
            return tuple(self.data.shape)
    
    @property
    def dtype(self):
        """Data type of the patch."""
        return self.data.dtype
    
    def to_tensor(self, device: str = 'cpu') -> 'torch.Tensor':
        """
        Convert patch data to PyTorch tensor.
        
        Args:
            device: Target device for the tensor
            
        Returns:
            PyTorch tensor representation of the patch
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        if isinstance(self.data, torch.Tensor):
            return self.data.to(device)
        else:
            return torch.from_numpy(self.data).to(device)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert patch data to numpy array.
        
        Returns:
            Numpy array representation of the patch
        """
        if isinstance(self.data, np.ndarray):
            return self.data
        else:
            return self.data.detach().cpu().numpy()
    
    def normalize(self, method: str = 'minmax') -> 'Patch2D':
        """
        Normalize patch data.
        
        Args:
            method: Normalization method ('minmax', 'zscore')
            
        Returns:
            New Patch2D with normalized data
        """
        if isinstance(self.data, np.ndarray):
            data = self.data.copy()
        else:
            data = self.data.clone()
        
        if method == 'minmax':
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        elif method == 'zscore':
            data_mean = data.mean()
            data_std = data.std()
            if data_std > 0:
                data = (data - data_mean) / data_std
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return Patch2D(
            data=data,
            slice_idx=self.slice_idx,
            start_row=self.start_row,
            start_col=self.start_col,
            patch_size=self.patch_size,
            original_slice_shape=self.original_slice_shape,
            metadata=self.metadata.copy()
        )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the patch."""
        self.metadata[key] = value
    
    def get_position_info(self) -> Dict[str, Any]:
        """Get position information as a dictionary."""
        return {
            'slice_idx': self.slice_idx,
            'start_row': self.start_row,
            'start_col': self.start_col,
            'end_row': self.end_row,
            'end_col': self.end_col,
            'patch_size': self.patch_size,
            'original_slice_shape': self.original_slice_shape
        }


@dataclass
class TrainingPair2D:
    """
    Represents a pair of input and target 2D patches for training.
    
    This class is designed for 2D U-Net training where we have
    corresponding input (low quality) and target (high quality) patches.
    """
    input_patch: Patch2D
    target_patch: Patch2D
    augmentation_applied: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate training pair after initialization."""
        # Check that patches have same position and size
        if self.input_patch.slice_idx != self.target_patch.slice_idx:
            raise ValueError("Input and target patches must be from the same slice")
        
        if self.input_patch.start_row != self.target_patch.start_row:
            raise ValueError("Input and target patches must have same position")
        
        if self.input_patch.start_col != self.target_patch.start_col:
            raise ValueError("Input and target patches must have same position")
        
        if self.input_patch.patch_size != self.target_patch.patch_size:
            raise ValueError("Input and target patches must have same size")
    
    @property
    def slice_idx(self) -> int:
        """Slice index of the training pair."""
        return self.input_patch.slice_idx
    
    @property
    def patch_size(self) -> Tuple[int, int]:
        """Size of the patches."""
        return self.input_patch.patch_size
    
    @property
    def position(self) -> Tuple[int, int]:
        """Position of the patches (start_row, start_col)."""
        return (self.input_patch.start_row, self.input_patch.start_col)
    
    def to_tensors(self, device: str = 'cpu') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """
        Convert both patches to PyTorch tensors.
        
        Args:
            device: Target device for tensors
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        input_tensor = self.input_patch.to_tensor(device)
        target_tensor = self.target_patch.to_tensor(device)
        return input_tensor, target_tensor
    
    def apply_augmentation(self, augmentation_name: str) -> 'TrainingPair2D':
        """
        Apply augmentation to both patches.
        
        Args:
            augmentation_name: Name of the augmentation applied
            
        Returns:
            New TrainingPair2D with augmentation applied
        """
        # This is a placeholder - actual augmentation would be implemented
        # based on specific requirements (rotation, flip, noise, etc.)
        
        augmented_input = Patch2D(
            data=self.input_patch.data,  # Would apply actual augmentation here
            slice_idx=self.input_patch.slice_idx,
            start_row=self.input_patch.start_row,
            start_col=self.input_patch.start_col,
            patch_size=self.input_patch.patch_size,
            original_slice_shape=self.input_patch.original_slice_shape,
            metadata=self.input_patch.metadata.copy()
        )
        
        augmented_target = Patch2D(
            data=self.target_patch.data,  # Would apply same augmentation here
            slice_idx=self.target_patch.slice_idx,
            start_row=self.target_patch.start_row,
            start_col=self.target_patch.start_col,
            patch_size=self.target_patch.patch_size,
            original_slice_shape=self.target_patch.original_slice_shape,
            metadata=self.target_patch.metadata.copy()
        )
        
        new_augmentations = self.augmentation_applied + [augmentation_name]
        
        return TrainingPair2D(
            input_patch=augmented_input,
            target_patch=augmented_target,
            augmentation_applied=new_augmentations
        )


class SliceStack:
    """
    Manages a stack of 2D image slices for processing with 2D U-Net.
    
    This class provides a convenient interface for working with multi-slice
    image data (like 3D TIFF files) while emphasizing that processing
    happens slice-by-slice using 2D operations.
    
    Features:
    - Slice-by-slice access and modification
    - PyTorch tensor and numpy array support
    - Metadata management
    - Validation and type checking
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, 'torch.Tensor'],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SliceStack with multi-slice data.
        
        Args:
            data: 3D array/tensor with shape (num_slices, height, width)
            metadata: Optional metadata dictionary
        """
        if isinstance(data, np.ndarray):
            if data.ndim != 3:
                raise ValueError(f"Data must be 3D (slices, height, width), got {data.ndim}D")
            self._data = data
            self._is_tensor = False
        elif TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            if data.dim() != 3:
                raise ValueError(f"Data must be 3D (slices, height, width), got {data.dim()}D")
            self._data = data
            self._is_tensor = True
        else:
            raise ValueError("Data must be numpy array or PyTorch tensor")
        
        self.metadata = metadata or {}
        
        logger.info(f"Created SliceStack with shape {self.shape}, dtype {self.dtype}")
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the slice stack (num_slices, height, width)."""
        return tuple(self._data.shape)
    
    @property
    def num_slices(self) -> int:
        """Number of slices in the stack."""
        return self._data.shape[0]
    
    @property
    def slice_shape(self) -> Tuple[int, int]:
        """Shape of individual slices (height, width)."""
        return (self._data.shape[1], self._data.shape[2])
    
    @property
    def dtype(self):
        """Data type of the slice stack."""
        return self._data.dtype
    
    @property
    def device(self) -> str:
        """Device of the data (for PyTorch tensors)."""
        if self._is_tensor:
            return str(self._data.device)
        else:
            return 'cpu'
    
    def get_slice(self, slice_idx: int) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Get a single 2D slice.
        
        Args:
            slice_idx: Index of the slice to retrieve
            
        Returns:
            2D array/tensor representing the slice
        """
        if slice_idx < 0 or slice_idx >= self.num_slices:
            raise IndexError(f"Slice index {slice_idx} out of range [0, {self.num_slices-1}]")
        
        return self._data[slice_idx]
    
    def set_slice(self, slice_idx: int, slice_data: Union[np.ndarray, 'torch.Tensor']) -> None:
        """
        Set a single 2D slice.
        
        Args:
            slice_idx: Index of the slice to set
            slice_data: 2D array/tensor to set as the slice
        """
        if slice_idx < 0 or slice_idx >= self.num_slices:
            raise IndexError(f"Slice index {slice_idx} out of range [0, {self.num_slices-1}]")
        
        # Validate slice data
        if isinstance(slice_data, np.ndarray):
            if slice_data.ndim != 2:
                raise ValueError("Slice data must be 2D")
            if slice_data.shape != self.slice_shape:
                raise ValueError(f"Slice shape {slice_data.shape} doesn't match expected {self.slice_shape}")
        elif TORCH_AVAILABLE and isinstance(slice_data, torch.Tensor):
            if slice_data.dim() != 2:
                raise ValueError("Slice tensor must be 2D")
            if tuple(slice_data.shape) != self.slice_shape:
                raise ValueError(f"Slice shape {slice_data.shape} doesn't match expected {self.slice_shape}")
        
        self._data[slice_idx] = slice_data
    
    def get_slices(self, slice_indices: List[int]) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Get multiple slices.
        
        Args:
            slice_indices: List of slice indices to retrieve
            
        Returns:
            3D array/tensor with selected slices
        """
        for idx in slice_indices:
            if idx < 0 or idx >= self.num_slices:
                raise IndexError(f"Slice index {idx} out of range [0, {self.num_slices-1}]")
        
        if self._is_tensor:
            return self._data[slice_indices]
        else:
            return self._data[slice_indices]
    
    def iter_slices(self) -> Iterator[Tuple[int, Union[np.ndarray, 'torch.Tensor']]]:
        """
        Iterate over slices with their indices.
        
        Yields:
            Tuple of (slice_index, slice_data)
        """
        for i in range(self.num_slices):
            yield i, self.get_slice(i)
    
    def to_tensor(self, device: str = 'cpu') -> 'torch.Tensor':
        """
        Convert to PyTorch tensor.
        
        Args:
            device: Target device for the tensor
            
        Returns:
            PyTorch tensor representation
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
        
        if self._is_tensor:
            return self._data.to(device)
        else:
            return torch.from_numpy(self._data).to(device)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert to numpy array.
        
        Returns:
            Numpy array representation
        """
        if self._is_tensor:
            return self._data.detach().cpu().numpy()
        else:
            return self._data
    
    def validate_slice_dimensions(self, expected_shape: Optional[Tuple[int, int]] = None) -> bool:
        """
        Validate slice dimensions.
        
        Args:
            expected_shape: Expected shape for each slice (height, width)
            
        Returns:
            True if all slices have valid dimensions
        """
        if expected_shape is None:
            # Just check that all slices have the same shape
            expected_shape = self.slice_shape
        
        for i in range(self.num_slices):
            slice_data = self.get_slice(i)
            if isinstance(slice_data, np.ndarray):
                slice_shape = slice_data.shape
            else:
                slice_shape = tuple(slice_data.shape)
            
            if slice_shape != expected_shape:
                logger.warning(f"Slice {i} has shape {slice_shape}, expected {expected_shape}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the slice stack.
        
        Returns:
            Dictionary with statistics
        """
        if self._is_tensor:
            data_min = self._data.min().item()
            data_max = self._data.max().item()
            data_mean = self._data.mean().item()
            data_std = self._data.std().item()
        else:
            data_min = float(self._data.min())
            data_max = float(self._data.max())
            data_mean = float(self._data.mean())
            data_std = float(self._data.std())
        
        return {
            'shape': self.shape,
            'num_slices': self.num_slices,
            'slice_shape': self.slice_shape,
            'dtype': str(self.dtype),
            'device': self.device,
            'min_value': data_min,
            'max_value': data_max,
            'mean_value': data_mean,
            'std_value': data_std,
            'total_pixels': int(np.prod(self.shape))
        }
    
    def extract_patches_from_slice(
        self,
        slice_idx: int,
        patch_size: Tuple[int, int],
        stride: Optional[Tuple[int, int]] = None
    ) -> List[Patch2D]:
        """
        Extract 2D patches from a specific slice.
        
        Args:
            slice_idx: Index of the slice to extract patches from
            patch_size: Size of patches (height, width)
            stride: Stride for patch extraction (default: same as patch_size)
            
        Returns:
            List of Patch2D objects
        """
        if stride is None:
            stride = patch_size
        
        slice_data = self.get_slice(slice_idx)
        height, width = self.slice_shape
        patch_h, patch_w = patch_size
        stride_h, stride_w = stride
        
        patches = []
        
        for start_row in range(0, height - patch_h + 1, stride_h):
            for start_col in range(0, width - patch_w + 1, stride_w):
                end_row = min(start_row + patch_h, height)
                end_col = min(start_col + patch_w, width)
                
                # Extract patch data
                if self._is_tensor:
                    patch_data = slice_data[start_row:end_row, start_col:end_col]
                else:
                    patch_data = slice_data[start_row:end_row, start_col:end_col].copy()
                
                # Create Patch2D object
                patch = Patch2D(
                    data=patch_data,
                    slice_idx=slice_idx,
                    start_row=start_row,
                    start_col=start_col,
                    patch_size=(end_row - start_row, end_col - start_col),
                    original_slice_shape=self.slice_shape
                )
                
                patches.append(patch)
        
        logger.debug(f"Extracted {len(patches)} patches from slice {slice_idx}")
        return patches
    
    def reconstruct_slice_from_patches(
        self,
        patches: List[Patch2D],
        slice_idx: int,
        blend_overlaps: bool = True
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Reconstruct a slice from patches.
        
        Args:
            patches: List of Patch2D objects for the slice
            slice_idx: Index of the slice to reconstruct
            blend_overlaps: Whether to blend overlapping regions
            
        Returns:
            Reconstructed 2D slice
        """
        # Filter patches for this slice
        slice_patches = [p for p in patches if p.slice_idx == slice_idx]
        
        if not slice_patches:
            raise ValueError(f"No patches found for slice {slice_idx}")
        
        # Initialize reconstruction arrays
        if self._is_tensor:
            reconstructed = torch.zeros(self.slice_shape, dtype=self.dtype, device=self.device)
            if blend_overlaps:
                weight_map = torch.zeros(self.slice_shape, dtype=torch.float32, device=self.device)
        else:
            reconstructed = np.zeros(self.slice_shape, dtype=self.dtype)
            if blend_overlaps:
                weight_map = np.zeros(self.slice_shape, dtype=np.float32)
        
        # Reconstruct from patches
        for patch in slice_patches:
            start_row, start_col = patch.start_row, patch.start_col
            end_row, end_col = patch.end_row, patch.end_col
            
            if blend_overlaps:
                # Simple uniform weighting (could be improved with distance-based weighting)
                weight = 1.0
                reconstructed[start_row:end_row, start_col:end_col] += patch.data * weight
                weight_map[start_row:end_row, start_col:end_col] += weight
            else:
                reconstructed[start_row:end_row, start_col:end_col] = patch.data
        
        # Normalize by weights if blending
        if blend_overlaps:
            if self._is_tensor:
                weight_map = torch.clamp(weight_map, min=1e-8)
            else:
                weight_map = np.clip(weight_map, 1e-8, None)
            reconstructed = reconstructed / weight_map
        
        return reconstructed
    
    @classmethod
    def from_tiff_handler(cls, tiff_handler, file_path: str) -> 'SliceStack':
        """
        Create SliceStack from TIFF file using TIFFDataHandler.
        
        Args:
            tiff_handler: TIFFDataHandler instance
            file_path: Path to TIFF file
            
        Returns:
            SliceStack instance
        """
        data = tiff_handler.load_3d_tiff(file_path)
        metadata = tiff_handler.get_metadata(file_path)
        
        return cls(data=data, metadata=metadata)
    
    def save_with_tiff_handler(self, tiff_handler, file_path: str) -> None:
        """
        Save SliceStack to TIFF file using TIFFDataHandler.
        
        Args:
            tiff_handler: TIFFDataHandler instance
            file_path: Path to save TIFF file
        """
        data = self.to_numpy()  # Convert to numpy for TIFF saving
        tiff_handler.save_3d_tiff(data, file_path)