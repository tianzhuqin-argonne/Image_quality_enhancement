"""
TIFF Data Handler for 3D image processing.

This module provides functionality for loading, validating, and saving 3D TIFF datasets
with specific focus on handling large grayscale image volumes.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import tifffile
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)


class TIFFDataHandler:
    """
    Handles loading, validation, and saving of 3D TIFF datasets with PyTorch integration.
    
    This class provides methods for working with multi-slice TIFF files,
    specifically designed for processing large 3D grayscale image volumes.
    Supports flexible dimension validation (typically 1000-5000 pixels per slice)
    and seamless integration with PyTorch tensors for deep learning workflows.
    
    Features:
    - Flexible dimension validation (configurable min/max dimensions)
    - PyTorch tensor support for deep learning workflows
    - Robust error handling for corrupted files
    - Metadata extraction and preservation
    - Memory-efficient processing of large datasets
    """
    
    def __init__(self, min_dimension: int = 1000, max_dimension: int = 5000):
        """
        Initialize the TIFF data handler.
        
        Args:
            min_dimension: Minimum acceptable dimension for height/width (default: 1000)
            max_dimension: Maximum acceptable dimension for height/width (default: 5000)
        """
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.supported_dtypes = [np.uint8, np.uint16, np.float32, np.float64]
    
    def load_3d_tiff(self, file_path: str) -> np.ndarray:
        """
        Load a 3D TIFF file containing multiple slices.
        
        Args:
            file_path: Path to the TIFF file to load
            
        Returns:
            numpy.ndarray: 3D array with shape (slices, height, width)
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid or corrupted
            IOError: If there's an error reading the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {file_path}")
        
        if not file_path.suffix.lower() in ['.tif', '.tiff']:
            raise ValueError(f"File must be a TIFF file, got: {file_path.suffix}")
        
        try:
            logger.info(f"Loading 3D TIFF file: {file_path}")
            
            # Load the TIFF file using tifffile
            with tifffile.TiffFile(file_path) as tif:
                # Read all pages/slices
                data = tif.asarray()
                
                # Ensure we have a 3D array
                if data.ndim == 2:
                    # Single slice, add slice dimension
                    data = data[np.newaxis, ...]
                elif data.ndim != 3:
                    raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
                
                # Ensure grayscale (single channel)
                if data.ndim == 4 and data.shape[-1] == 1:
                    data = data.squeeze(-1)
                elif data.ndim == 4:
                    raise ValueError(f"Expected grayscale data, got {data.shape[-1]} channels")
                
                logger.info(f"Loaded TIFF with shape: {data.shape}, dtype: {data.dtype}")
                return data
                
        except tifffile.TiffFileError as e:
            raise ValueError(f"Corrupted or invalid TIFF file: {e}")
        except Exception as e:
            raise IOError(f"Error reading TIFF file: {e}")
    
    def validate_dimensions(self, data: Union[np.ndarray, 'torch.Tensor']) -> bool:
        """
        Validate that the data meets the dimension requirements.
        
        Args:
            data: 3D numpy array or PyTorch tensor to validate
            
        Returns:
            bool: True if dimensions are valid, False otherwise
            
        Raises:
            ValueError: If data is not a 3D array/tensor
        """
        # Handle both numpy arrays and PyTorch tensors
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            shape = data.shape
            dtype = data.dtype
            data_type = "torch.Tensor"
        elif isinstance(data, np.ndarray):
            shape = data.shape
            dtype = data.dtype
            data_type = "numpy.ndarray"
        else:
            raise ValueError("Data must be a numpy array or PyTorch tensor")
        
        if len(shape) != 3:
            raise ValueError(f"Expected 3D data, got {len(shape)}D")
        
        slices, height, width = shape
        
        # Check if dimensions are within acceptable range (flexible validation)
        height_valid = self.min_dimension <= height <= self.max_dimension
        width_valid = self.min_dimension <= width <= self.max_dimension
        dimensions_valid = height_valid and width_valid
        
        if not dimensions_valid:
            logger.warning(
                f"Dimension out of range: expected {self.min_dimension}-{self.max_dimension} pixels, "
                f"got {height}×{width}"
            )
        
        # Check data type (more flexible for PyTorch tensors)
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # PyTorch tensors can be converted to supported numpy dtypes
            dtype_valid = True
        else:
            dtype_valid = dtype in self.supported_dtypes
            if not dtype_valid:
                logger.warning(f"Unsupported data type: {dtype}")
        
        # Log validation results
        logger.info(
            f"Validation results - Type: {data_type}, Shape: {shape}, "
            f"Dimensions valid: {dimensions_valid}, "
            f"Data type valid: {dtype_valid}"
        )
        
        return dimensions_valid and dtype_valid
    
    def save_3d_tiff(self, data: Union[np.ndarray, 'torch.Tensor'], output_path: str) -> None:
        """
        Save a 3D numpy array or PyTorch tensor as a multi-slice TIFF file.
        
        Args:
            data: 3D numpy array or PyTorch tensor with shape (slices, height, width)
            output_path: Path where the TIFF file will be saved
            
        Raises:
            ValueError: If data format is invalid
            IOError: If there's an error writing the file
        """
        # Convert PyTorch tensor to numpy array if needed
        if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
            # Move to CPU and convert to numpy
            data = data.detach().cpu().numpy()
            logger.info("Converted PyTorch tensor to numpy array for saving")
        elif not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array or PyTorch tensor")
        
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")
        
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving 3D TIFF to: {output_path}")
            logger.info(f"Data shape: {data.shape}, dtype: {data.dtype}")
            
            # Save using tifffile without compression (to avoid imagecodecs dependency)
            tifffile.imwrite(
                output_path,
                data,
                metadata={'axes': 'ZYX'}  # Z=slices, Y=height, X=width
            )
            
            logger.info(f"Successfully saved TIFF file: {output_path}")
            
        except Exception as e:
            raise IOError(f"Error saving TIFF file: {e}")
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata information from a TIFF file.
        
        Args:
            file_path: Path to the TIFF file
            
        Returns:
            Dict containing metadata information
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is corrupted or invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {file_path}")
        
        try:
            metadata = {}
            
            with tifffile.TiffFile(file_path) as tif:
                # Basic file information
                metadata['file_path'] = str(file_path)
                metadata['file_size_bytes'] = file_path.stat().st_size
                metadata['num_pages'] = len(tif.pages)
                
                # Get information from first page
                if tif.pages:
                    first_page = tif.pages[0]
                    metadata['image_width'] = first_page.imagewidth
                    metadata['image_height'] = first_page.imagelength
                    metadata['bits_per_sample'] = first_page.bitspersample
                    metadata['samples_per_pixel'] = first_page.samplesperpixel
                    metadata['photometric'] = first_page.photometric.name
                    metadata['compression'] = first_page.compression.name
                    
                    # Data type information
                    sample_format = getattr(first_page, 'sampleformat', None)
                    if sample_format:
                        # Handle both enum and integer values
                        if hasattr(sample_format, 'name'):
                            metadata['sample_format'] = sample_format.name
                        else:
                            metadata['sample_format'] = str(sample_format)
                
                # Try to get additional metadata
                if hasattr(tif, 'imagej_metadata') and tif.imagej_metadata:
                    metadata['imagej_metadata'] = tif.imagej_metadata
                
                if hasattr(tif, 'ome_metadata') and tif.ome_metadata:
                    metadata['ome_metadata'] = tif.ome_metadata
                
                # Calculate total data shape if possible
                if tif.pages:
                    try:
                        shape = tif.asarray().shape
                        metadata['data_shape'] = shape
                        metadata['total_pixels'] = np.prod(shape)
                    except Exception as e:
                        logger.warning(f"Could not determine data shape: {e}")
                        metadata['data_shape'] = None
            
            logger.info(f"Extracted metadata for: {file_path}")
            return metadata
            
        except tifffile.TiffFileError as e:
            raise ValueError(f"Corrupted or invalid TIFF file: {e}")
        except Exception as e:
            raise IOError(f"Error reading TIFF metadata: {e}")
    
    def to_torch_tensor(self, data: np.ndarray, device: str = 'cpu') -> 'torch.Tensor':
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            data: 3D numpy array to convert
            device: Target device ('cpu', 'cuda', etc.)
            
        Returns:
            torch.Tensor: PyTorch tensor on specified device
            
        Raises:
            ImportError: If PyTorch is not available
            ValueError: If data format is invalid
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch.")
        
        if not isinstance(data, np.ndarray):
            raise ValueError("Data must be a numpy array")
        
        if data.ndim != 3:
            raise ValueError(f"Expected 3D data, got {data.ndim}D")
        
        # Convert to float32 for PyTorch compatibility
        if data.dtype not in [np.float32, np.float64]:
            # Normalize integer types to [0, 1] range
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            elif data.dtype == np.uint16:
                data = data.astype(np.float32) / 65535.0
            else:
                data = data.astype(np.float32)
        
        tensor = torch.from_numpy(data).to(device)
        logger.info(f"Converted numpy array to PyTorch tensor on {device}")
        return tensor
    
    def from_torch_tensor(self, tensor: 'torch.Tensor', target_dtype: np.dtype = np.float32) -> np.ndarray:
        """
        Convert PyTorch tensor to numpy array.
        
        Args:
            tensor: PyTorch tensor to convert
            target_dtype: Target numpy dtype
            
        Returns:
            numpy.ndarray: Numpy array
            
        Raises:
            ImportError: If PyTorch is not available
            ValueError: If tensor format is invalid
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install torch.")
        
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")
        
        if tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {tensor.dim()}D")
        
        # Move to CPU and convert to numpy
        data = tensor.detach().cpu().numpy()
        
        # Convert to target dtype if needed
        if data.dtype != target_dtype:
            if target_dtype in [np.uint8, np.uint16]:
                # Denormalize from [0, 1] range if needed
                if data.max() <= 1.0 and data.min() >= 0.0:
                    if target_dtype == np.uint8:
                        data = (data * 255.0).astype(target_dtype)
                    elif target_dtype == np.uint16:
                        data = (data * 65535.0).astype(target_dtype)
                else:
                    data = data.astype(target_dtype)
            else:
                data = data.astype(target_dtype)
        
        logger.info(f"Converted PyTorch tensor to numpy array with dtype {target_dtype}")
        return data