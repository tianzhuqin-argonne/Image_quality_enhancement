"""
Unit tests for training data preparation system.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.training.data_preparation import (
    TrainingDataLoader, TrainingDataset, DataAugmentation, AugmentationConfig
)
from src.core.data_models import SliceStack, Patch2D, TrainingPair2D
from src.core.tiff_handler import TIFFDataHandler


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDataAugmentation:
    """Test data augmentation functionality."""
    
    def test_augmentation_config_defaults(self):
        """Test default augmentation configuration."""
        config = AugmentationConfig()
        assert config.horizontal_flip is True
        assert config.vertical_flip is True
        assert config.rotation_90 is True
        assert config.gaussian_noise_std == 0.01
        assert config.brightness_range == (0.8, 1.2)
        assert config.contrast_range == (0.8, 1.2)
        assert config.apply_probability == 0.5
    
    def test_data_augmentation_init(self):
        """Test DataAugmentation initialization."""
        config = AugmentationConfig()
        augmentation = DataAugmentation(config)
        assert augmentation.config == config
    
    def test_apply_augmentation_no_change(self):
        """Test augmentation when probability is 0."""
        config = AugmentationConfig(apply_probability=0.0)
        augmentation = DataAugmentation(config)
        
        # Create test training pair
        input_data = torch.rand(64, 64)
        target_data = torch.rand(64, 64)
        
        input_patch = Patch2D(
            data=input_data, slice_idx=0, start_row=0, start_col=0,
            patch_size=(64, 64), original_slice_shape=(256, 256)
        )
        target_patch = Patch2D(
            data=target_data, slice_idx=0, start_row=0, start_col=0,
            patch_size=(64, 64), original_slice_shape=(256, 256)
        )
        
        training_pair = TrainingPair2D(input_patch, target_patch)
        
        # Apply augmentation (should not change anything)
        augmented = augmentation.apply_augmentation(training_pair)
        
        # Check that data is unchanged
        assert torch.equal(augmented.input_patch.data, input_data)
        assert torch.equal(augmented.target_patch.data, target_data)
    
    def test_apply_augmentation_with_changes(self):
        """Test augmentation when probability is 1."""
        config = AugmentationConfig(
            apply_probability=1.0,
            horizontal_flip=True,
            vertical_flip=False,
            rotation_90=False,
            gaussian_noise_std=0.0,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0)
        )
        augmentation = DataAugmentation(config)
        
        # Create test training pair with distinctive pattern
        input_data = torch.zeros(4, 4)
        input_data[0, :] = 1.0  # Top row is 1, rest is 0
        target_data = input_data.clone()
        
        input_patch = Patch2D(
            data=input_data, slice_idx=0, start_row=0, start_col=0,
            patch_size=(4, 4), original_slice_shape=(256, 256)
        )
        target_patch = Patch2D(
            data=target_data, slice_idx=0, start_row=0, start_col=0,
            patch_size=(4, 4), original_slice_shape=(256, 256)
        )
        
        training_pair = TrainingPair2D(input_patch, target_patch)
        
        # Apply augmentation multiple times to test randomness
        augmented_results = []
        for _ in range(10):
            augmented = augmentation.apply_augmentation(training_pair)
            augmented_results.append(augmented)
        
        # At least some should be different due to random horizontal flip
        original_sum = input_data.sum().item()
        augmented_sums = [result.input_patch.data.sum().item() for result in augmented_results]
        
        # All should have same sum (just flipped)
        assert all(abs(s - original_sum) < 1e-6 for s in augmented_sums)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingDataset:
    """Test PyTorch Dataset for training."""
    
    def test_training_dataset_init(self):
        """Test TrainingDataset initialization."""
        # Create test training pairs
        training_pairs = self._create_test_training_pairs(5)
        dataset = TrainingDataset(training_pairs)
        
        assert len(dataset) == 5
        assert dataset.training_pairs == training_pairs
        assert dataset.augmentation is None
        assert dataset.device == 'cpu'
    
    def test_training_dataset_getitem(self):
        """Test getting items from TrainingDataset."""
        training_pairs = self._create_test_training_pairs(3)
        dataset = TrainingDataset(training_pairs)
        
        # Get first item
        input_tensor, target_tensor = dataset[0]
        
        # Check tensor properties
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 64, 64)  # (channels, height, width)
        assert target_tensor.shape == (1, 64, 64)
        assert input_tensor.device.type == 'cpu'
        assert target_tensor.device.type == 'cpu'
    
    def test_training_dataset_with_augmentation(self):
        """Test TrainingDataset with augmentation."""
        training_pairs = self._create_test_training_pairs(2)
        config = AugmentationConfig(apply_probability=1.0)
        augmentation = DataAugmentation(config)
        dataset = TrainingDataset(training_pairs, augmentation=augmentation)
        
        # Get item (should apply augmentation)
        input_tensor, target_tensor = dataset[0]
        
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
    
    def _create_test_training_pairs(self, count: int) -> list:
        """Create test training pairs."""
        pairs = []
        for i in range(count):
            input_data = torch.rand(64, 64)
            target_data = torch.rand(64, 64)
            
            input_patch = Patch2D(
                data=input_data, slice_idx=i, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            target_patch = Patch2D(
                data=target_data, slice_idx=i, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            
            pairs.append(TrainingPair2D(input_patch, target_patch))
        
        return pairs


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingDataLoader:
    """Test training data loader functionality."""
    
    def test_training_data_loader_init(self):
        """Test TrainingDataLoader initialization."""
        loader = TrainingDataLoader(patch_size=(128, 128), overlap=16)
        
        assert loader.patch_size == (128, 128)
        assert loader.overlap == 16
        assert loader.device == 'cpu'
        assert isinstance(loader.tiff_handler, TIFFDataHandler)
    
    def test_load_training_data_validation(self):
        """Test validation in load_training_data."""
        loader = TrainingDataLoader()
        
        # Test mismatched path counts
        with pytest.raises(ValueError, match="Number of input and target paths must match"):
            loader.load_training_data(['path1'], ['path1', 'path2'])
    
    @patch('src.core.data_models.SliceStack.from_tiff_handler')
    def test_extract_training_pairs(self, mock_from_tiff):
        """Test extracting training pairs from volumes."""
        loader = TrainingDataLoader(patch_size=(64, 64), overlap=0)
        
        # Create mock volumes
        input_data = np.random.rand(2, 128, 128).astype(np.float32)
        target_data = np.random.rand(2, 128, 128).astype(np.float32)
        
        input_volume = SliceStack(input_data)
        target_volume = SliceStack(target_data)
        
        # Extract training pairs
        pairs = loader._extract_training_pairs(input_volume, target_volume, [0, 1])
        
        # Should have 4 patches per slice (2x2 grid), 2 slices = 8 pairs
        assert len(pairs) == 8
        
        # Check first pair
        first_pair = pairs[0]
        assert isinstance(first_pair, TrainingPair2D)
        assert first_pair.input_patch.patch_size == (64, 64)
        assert first_pair.target_patch.patch_size == (64, 64)
        assert first_pair.slice_idx == 0
    
    def test_create_data_loaders(self):
        """Test creating PyTorch DataLoaders."""
        loader = TrainingDataLoader()
        
        # Create test training pairs
        training_pairs = self._create_test_training_pairs(10)
        
        # Create data loaders
        train_loader, val_loader = loader.create_data_loaders(
            training_pairs, validation_split=0.2, batch_size=2
        )
        
        # Check loader properties
        assert len(train_loader.dataset) == 8  # 80% of 10
        assert len(val_loader.dataset) == 2   # 20% of 10
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        
        # Test getting a batch
        batch_input, batch_target = next(iter(train_loader))
        assert batch_input.shape == (2, 1, 64, 64)  # (batch, channels, height, width)
        assert batch_target.shape == (2, 1, 64, 64)
    
    def test_create_data_loaders_with_augmentation(self):
        """Test creating DataLoaders with augmentation."""
        loader = TrainingDataLoader()
        training_pairs = self._create_test_training_pairs(4)
        
        config = AugmentationConfig(apply_probability=0.5)
        train_loader, val_loader = loader.create_data_loaders(
            training_pairs, 
            validation_split=0.5, 
            batch_size=1,
            augmentation_config=config
        )
        
        # Training dataset should have augmentation, validation should not
        assert train_loader.dataset.augmentation is not None
        assert val_loader.dataset.augmentation is None
    
    def test_get_data_statistics(self):
        """Test getting data statistics."""
        loader = TrainingDataLoader()
        training_pairs = self._create_test_training_pairs(5)
        
        stats = loader.get_data_statistics(training_pairs)
        
        assert stats['num_pairs'] == 5
        assert 'unique_slices' in stats
        assert 'patch_sizes' in stats
        assert 'input_stats' in stats
        assert 'target_stats' in stats
        
        # Check input stats structure
        input_stats = stats['input_stats']
        assert 'min' in input_stats
        assert 'max' in input_stats
        assert 'mean' in input_stats
        assert 'std' in input_stats
    
    def test_get_data_statistics_empty(self):
        """Test getting statistics with empty data."""
        loader = TrainingDataLoader()
        stats = loader.get_data_statistics([])
        assert stats == {}
    
    def _create_test_training_pairs(self, count: int) -> list:
        """Create test training pairs."""
        pairs = []
        for i in range(count):
            input_data = np.random.rand(64, 64).astype(np.float32)
            target_data = np.random.rand(64, 64).astype(np.float32)
            
            input_patch = Patch2D(
                data=input_data, slice_idx=i % 3, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            target_patch = Patch2D(
                data=target_data, slice_idx=i % 3, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            
            pairs.append(TrainingPair2D(input_patch, target_patch))
        
        return pairs


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_import_error_without_torch():
    """Test that appropriate errors are raised when PyTorch is not available."""
    with pytest.raises(ImportError, match="PyTorch is required"):
        TrainingDataLoader()
    
    with pytest.raises(ImportError, match="PyTorch is required"):
        config = AugmentationConfig()
        DataAugmentation(config)
    
    with pytest.raises(ImportError, match="PyTorch is required"):
        TrainingDataset([])