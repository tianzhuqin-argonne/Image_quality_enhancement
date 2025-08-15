"""
Training data preparation system for 3D image enhancement.

This module provides functionality for loading training data, extracting patches,
applying data augmentation, and creating PyTorch DataLoaders for training.
"""

import logging
import random
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
from pathlib import Path
import numpy as np
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.data_models import SliceStack, Patch2D, TrainingPair2D
from ..core.tiff_handler import TIFFDataHandler
from ..core.patch_processor import PatchProcessor

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation."""
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_90: bool = True
    gaussian_noise_std: float = 0.01
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    apply_probability: float = 0.5


class DataAugmentation:
    """
    Data augmentation for training patches.
    
    Provides various augmentation techniques suitable for grayscale image patches
    while ensuring that both input and target patches receive identical transformations.
    """
    
    def __init__(self, config: AugmentationConfig):
        """
        Initialize data augmentation.
        
        Args:
            config: Augmentation configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for data augmentation")
        
        self.config = config
        logger.info(f"Initialized DataAugmentation with config: {config}")
    
    def apply_augmentation(self, training_pair: TrainingPair2D) -> TrainingPair2D:
        """
        Apply random augmentation to a training pair.
        
        Args:
            training_pair: Input training pair
            
        Returns:
            Augmented training pair
        """
        if random.random() > self.config.apply_probability:
            return training_pair
        
        # Convert patches to tensors for augmentation
        input_tensor = training_pair.input_patch.to_tensor()
        target_tensor = training_pair.target_patch.to_tensor()
        
        augmentations_applied = []
        
        # Geometric augmentations (applied to both input and target)
        if self.config.horizontal_flip and random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[1])
            target_tensor = torch.flip(target_tensor, dims=[1])
            augmentations_applied.append("horizontal_flip")
        
        if self.config.vertical_flip and random.random() < 0.5:
            input_tensor = torch.flip(input_tensor, dims=[0])
            target_tensor = torch.flip(target_tensor, dims=[0])
            augmentations_applied.append("vertical_flip")
        
        if self.config.rotation_90 and random.random() < 0.5:
            k = random.randint(1, 3)  # 90, 180, or 270 degrees
            input_tensor = torch.rot90(input_tensor, k=k, dims=[0, 1])
            target_tensor = torch.rot90(target_tensor, k=k, dims=[0, 1])
            augmentations_applied.append(f"rotation_90_{k}")
        
        # Photometric augmentations (applied only to input)
        if self.config.gaussian_noise_std > 0 and random.random() < 0.5:
            noise = torch.randn_like(input_tensor) * self.config.gaussian_noise_std
            input_tensor = torch.clamp(input_tensor + noise, 0, 1)
            augmentations_applied.append("gaussian_noise")
        
        if self.config.brightness_range != (1.0, 1.0) and random.random() < 0.5:
            brightness_factor = random.uniform(*self.config.brightness_range)
            input_tensor = torch.clamp(input_tensor * brightness_factor, 0, 1)
            augmentations_applied.append(f"brightness_{brightness_factor:.2f}")
        
        if self.config.contrast_range != (1.0, 1.0) and random.random() < 0.5:
            contrast_factor = random.uniform(*self.config.contrast_range)
            mean_val = input_tensor.mean()
            input_tensor = torch.clamp(
                (input_tensor - mean_val) * contrast_factor + mean_val, 0, 1
            )
            augmentations_applied.append(f"contrast_{contrast_factor:.2f}")
        
        # Create augmented patches
        augmented_input = Patch2D(
            data=input_tensor,
            slice_idx=training_pair.input_patch.slice_idx,
            start_row=training_pair.input_patch.start_row,
            start_col=training_pair.input_patch.start_col,
            patch_size=training_pair.input_patch.patch_size,
            original_slice_shape=training_pair.input_patch.original_slice_shape,
            metadata=training_pair.input_patch.metadata.copy()
        )
        
        augmented_target = Patch2D(
            data=target_tensor,
            slice_idx=training_pair.target_patch.slice_idx,
            start_row=training_pair.target_patch.start_row,
            start_col=training_pair.target_patch.start_col,
            patch_size=training_pair.target_patch.patch_size,
            original_slice_shape=training_pair.target_patch.original_slice_shape,
            metadata=training_pair.target_patch.metadata.copy()
        )
        
        return TrainingPair2D(
            input_patch=augmented_input,
            target_patch=augmented_target,
            augmentation_applied=training_pair.augmentation_applied + augmentations_applied
        )


class TrainingDataset(Dataset):
    """
    PyTorch Dataset for training patch pairs.
    
    Handles loading and augmentation of training patch pairs for U-Net training.
    """
    
    def __init__(
        self,
        training_pairs: List[TrainingPair2D],
        augmentation: Optional[DataAugmentation] = None,
        device: str = 'cpu'
    ):
        """
        Initialize training dataset.
        
        Args:
            training_pairs: List of training patch pairs
            augmentation: Optional data augmentation
            device: Target device for tensors
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrainingDataset")
        
        self.training_pairs = training_pairs
        self.augmentation = augmentation
        self.device = device
        
        logger.info(f"Created TrainingDataset with {len(training_pairs)} pairs")
    
    def __len__(self) -> int:
        """Return the number of training pairs."""
        return len(self.training_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training pair as tensors.
        
        Args:
            idx: Index of the training pair
            
        Returns:
            Tuple of (input_tensor, target_tensor)
        """
        training_pair = self.training_pairs[idx]
        
        # Apply augmentation if configured
        if self.augmentation is not None:
            training_pair = self.augmentation.apply_augmentation(training_pair)
        
        # Convert to tensors
        input_tensor, target_tensor = training_pair.to_tensors(self.device)
        
        # Add channel dimension for U-Net (batch_size, channels, height, width)
        input_tensor = input_tensor.unsqueeze(0)  # Add channel dimension
        target_tensor = target_tensor.unsqueeze(0)  # Add channel dimension
        
        return input_tensor, target_tensor


class TrainingDataLoader:
    """
    Training data preparation system.
    
    Handles loading TIFF datasets, extracting patches, creating training pairs,
    and preparing PyTorch DataLoaders for training.
    """
    
    def __init__(
        self,
        patch_size: Tuple[int, int] = (256, 256),
        overlap: int = 32,
        device: str = 'cpu'
    ):
        """
        Initialize training data loader.
        
        Args:
            patch_size: Size of patches for training
            overlap: Overlap between patches
            device: PyTorch device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TrainingDataLoader")
        
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = device
        
        # Initialize components
        self.tiff_handler = TIFFDataHandler()
        self.patch_processor = PatchProcessor(
            patch_size=patch_size,
            overlap=overlap,
            device=device
        )
        
        logger.info(f"Initialized TrainingDataLoader with patch_size={patch_size}")
    
    def load_training_data(
        self,
        input_paths: List[str],
        target_paths: List[str],
        slice_indices: Optional[List[int]] = None
    ) -> List[TrainingPair2D]:
        """
        Load training data from TIFF files and create training pairs.
        
        Args:
            input_paths: Paths to input (low-quality) TIFF files
            target_paths: Paths to target (high-quality) TIFF files
            slice_indices: Optional list of slice indices to use
            
        Returns:
            List of training patch pairs
        """
        if len(input_paths) != len(target_paths):
            raise ValueError("Number of input and target paths must match")
        
        training_pairs = []
        
        for input_path, target_path in zip(input_paths, target_paths):
            logger.info(f"Loading training pair: {input_path} -> {target_path}")
            
            # Load input and target volumes
            input_volume = SliceStack.from_tiff_handler(self.tiff_handler, input_path)
            target_volume = SliceStack.from_tiff_handler(self.tiff_handler, target_path)
            
            # Validate dimensions match
            if input_volume.shape != target_volume.shape:
                raise ValueError(
                    f"Input and target volumes must have same shape. "
                    f"Got {input_volume.shape} vs {target_volume.shape}"
                )
            
            # Extract patches from both volumes
            pairs = self._extract_training_pairs(
                input_volume, target_volume, slice_indices
            )
            training_pairs.extend(pairs)
        
        logger.info(f"Loaded {len(training_pairs)} training pairs total")
        return training_pairs
    
    def _extract_training_pairs(
        self,
        input_volume: SliceStack,
        target_volume: SliceStack,
        slice_indices: Optional[List[int]] = None
    ) -> List[TrainingPair2D]:
        """
        Extract training pairs from input and target volumes.
        
        Args:
            input_volume: Input volume
            target_volume: Target volume
            slice_indices: Optional slice indices to process
            
        Returns:
            List of training pairs
        """
        if slice_indices is None:
            slice_indices = list(range(input_volume.num_slices))
        
        training_pairs = []
        
        for slice_idx in slice_indices:
            if slice_idx >= input_volume.num_slices:
                logger.warning(f"Slice index {slice_idx} out of range, skipping")
                continue
            
            # Extract patches from both slices
            input_patches = input_volume.extract_patches_from_slice(
                slice_idx, self.patch_size, 
                stride=(self.patch_size[0] - self.overlap, self.patch_size[1] - self.overlap)
            )
            target_patches = target_volume.extract_patches_from_slice(
                slice_idx, self.patch_size,
                stride=(self.patch_size[0] - self.overlap, self.patch_size[1] - self.overlap)
            )
            
            # Create training pairs
            if len(input_patches) != len(target_patches):
                logger.warning(
                    f"Patch count mismatch for slice {slice_idx}: "
                    f"{len(input_patches)} vs {len(target_patches)}"
                )
                continue
            
            for input_patch, target_patch in zip(input_patches, target_patches):
                # Validate patch positions match
                if (input_patch.start_row != target_patch.start_row or
                    input_patch.start_col != target_patch.start_col):
                    logger.warning(f"Patch position mismatch in slice {slice_idx}")
                    continue
                
                training_pair = TrainingPair2D(
                    input_patch=input_patch,
                    target_patch=target_patch
                )
                training_pairs.append(training_pair)
        
        return training_pairs
    
    def create_data_loaders(
        self,
        training_pairs: List[TrainingPair2D],
        validation_split: float = 0.2,
        batch_size: int = 16,
        augmentation_config: Optional[AugmentationConfig] = None,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create training and validation DataLoaders.
        
        Args:
            training_pairs: List of training pairs
            validation_split: Fraction of data to use for validation
            batch_size: Batch size for training
            augmentation_config: Optional augmentation configuration
            shuffle: Whether to shuffle training data
            num_workers: Number of worker processes for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if not training_pairs:
            raise ValueError("No training pairs provided")
        
        # Split data into training and validation
        num_val = int(len(training_pairs) * validation_split)
        num_train = len(training_pairs) - num_val
        
        # Shuffle before splitting if requested
        if shuffle:
            random.shuffle(training_pairs)
        
        train_pairs = training_pairs[:num_train]
        val_pairs = training_pairs[num_train:]
        
        logger.info(f"Split data: {num_train} training, {num_val} validation pairs")
        
        # Create augmentation if configured
        augmentation = None
        if augmentation_config is not None:
            augmentation = DataAugmentation(augmentation_config)
        
        # Create datasets
        train_dataset = TrainingDataset(
            train_pairs, augmentation=augmentation, device=self.device
        )
        val_dataset = TrainingDataset(
            val_pairs, augmentation=None, device=self.device  # No augmentation for validation
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(self.device != 'cpu')
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device != 'cpu')
        )
        
        logger.info(
            f"Created DataLoaders: train_batches={len(train_loader)}, "
            f"val_batches={len(val_loader)}, batch_size={batch_size}"
        )
        
        return train_loader, val_loader
    
    def get_data_statistics(self, training_pairs: List[TrainingPair2D]) -> Dict[str, Any]:
        """
        Get statistics about the training data.
        
        Args:
            training_pairs: List of training pairs
            
        Returns:
            Dictionary with data statistics
        """
        if not training_pairs:
            return {}
        
        # Collect statistics
        input_values = []
        target_values = []
        patch_sizes = []
        slice_indices = set()
        
        for pair in training_pairs[:1000]:  # Sample first 1000 pairs for efficiency
            input_data = pair.input_patch.to_numpy()
            target_data = pair.target_patch.to_numpy()
            
            input_values.extend(input_data.flatten())
            target_values.extend(target_data.flatten())
            patch_sizes.append(pair.patch_size)
            slice_indices.add(pair.slice_idx)
        
        input_values = np.array(input_values)
        target_values = np.array(target_values)
        
        return {
            'num_pairs': len(training_pairs),
            'unique_slices': len(slice_indices),
            'patch_sizes': list(set(patch_sizes)),
            'input_stats': {
                'min': float(input_values.min()),
                'max': float(input_values.max()),
                'mean': float(input_values.mean()),
                'std': float(input_values.std())
            },
            'target_stats': {
                'min': float(target_values.min()),
                'max': float(target_values.max()),
                'mean': float(target_values.mean()),
                'std': float(target_values.std())
            }
        }