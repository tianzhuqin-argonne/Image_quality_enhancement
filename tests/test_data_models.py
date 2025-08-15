"""
Unit tests for data models (SliceStack, Patch2D, TrainingPair2D).
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.data_models import SliceStack, Patch2D, TrainingPair2D


class TestPatch2D:
    """Test cases for Patch2D class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.patch_data = np.random.rand(64, 64).astype(np.float32)
        self.patch = Patch2D(
            data=self.patch_data,
            slice_idx=5,
            start_row=100,
            start_col=200,
            patch_size=(64, 64),
            original_slice_shape=(1000, 1000)
        )
    
    def test_patch2d_creation(self):
        """Test Patch2D creation and basic properties."""
        assert self.patch.slice_idx == 5
        assert self.patch.start_row == 100
        assert self.patch.start_col == 200
        assert self.patch.patch_size == (64, 64)
        assert self.patch.original_slice_shape == (1000, 1000)
        assert self.patch.shape == (64, 64)
        assert self.patch.end_row == 164
        assert self.patch.end_col == 264
    
    def test_patch2d_invalid_data(self):
        """Test Patch2D with invalid data."""
        # 3D data should fail
        with pytest.raises(ValueError, match="Patch data must be 2D"):
            Patch2D(
                data=np.random.rand(10, 64, 64),
                slice_idx=0,
                start_row=0,
                start_col=0,
                patch_size=(64, 64),
                original_slice_shape=(1000, 1000)
            )
        
        # Invalid coordinates
        with pytest.raises(ValueError, match="Patch coordinates must be non-negative"):
            Patch2D(
                data=self.patch_data,
                slice_idx=0,
                start_row=-1,
                start_col=0,
                patch_size=(64, 64),
                original_slice_shape=(1000, 1000)
            )
    
    def test_patch2d_to_numpy(self):
        """Test conversion to numpy array."""
        numpy_data = self.patch.to_numpy()
        assert isinstance(numpy_data, np.ndarray)
        np.testing.assert_array_equal(numpy_data, self.patch_data)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_patch2d_to_tensor(self):
        """Test conversion to PyTorch tensor."""
        tensor_data = self.patch.to_tensor()
        assert isinstance(tensor_data, torch.Tensor)
        assert tensor_data.shape == (64, 64)
        assert tensor_data.device.type == 'cpu'
        
        # Test with different device
        if torch.cuda.is_available():
            cuda_tensor = self.patch.to_tensor('cuda')
            assert cuda_tensor.device.type == 'cuda'
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_patch2d_with_tensor_data(self):
        """Test Patch2D with PyTorch tensor data."""
        tensor_data = torch.randn(32, 32)
        patch = Patch2D(
            data=tensor_data,
            slice_idx=0,
            start_row=0,
            start_col=0,
            patch_size=(32, 32),
            original_slice_shape=(512, 512)
        )
        
        assert patch.shape == (32, 32)
        assert isinstance(patch.to_tensor(), torch.Tensor)
        assert isinstance(patch.to_numpy(), np.ndarray)
    
    def test_patch2d_normalize(self):
        """Test patch normalization."""
        # Create patch with known values
        data = np.array([[0, 1], [2, 3]], dtype=np.float32)
        patch = Patch2D(
            data=data,
            slice_idx=0,
            start_row=0,
            start_col=0,
            patch_size=(2, 2),
            original_slice_shape=(10, 10)
        )
        
        # Test minmax normalization
        normalized = patch.normalize('minmax')
        expected = np.array([[0, 1/3], [2/3, 1]], dtype=np.float32)
        np.testing.assert_array_almost_equal(normalized.data, expected)
        
        # Test zscore normalization
        zscore_normalized = patch.normalize('zscore')
        assert isinstance(zscore_normalized, Patch2D)
        assert zscore_normalized.data.mean() == pytest.approx(0, abs=1e-6)
    
    def test_patch2d_metadata(self):
        """Test patch metadata handling."""
        self.patch.add_metadata('quality_score', 0.95)
        self.patch.add_metadata('processing_time', 1.23)
        
        assert self.patch.metadata['quality_score'] == 0.95
        assert self.patch.metadata['processing_time'] == 1.23
    
    def test_patch2d_position_info(self):
        """Test position information extraction."""
        pos_info = self.patch.get_position_info()
        
        expected_keys = [
            'slice_idx', 'start_row', 'start_col', 'end_row', 'end_col',
            'patch_size', 'original_slice_shape'
        ]
        
        for key in expected_keys:
            assert key in pos_info
        
        assert pos_info['slice_idx'] == 5
        assert pos_info['start_row'] == 100
        assert pos_info['end_row'] == 164


class TestTrainingPair2D:
    """Test cases for TrainingPair2D class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        input_data = np.random.rand(32, 32).astype(np.float32)
        target_data = np.random.rand(32, 32).astype(np.float32)
        
        self.input_patch = Patch2D(
            data=input_data,
            slice_idx=3,
            start_row=50,
            start_col=75,
            patch_size=(32, 32),
            original_slice_shape=(512, 512)
        )
        
        self.target_patch = Patch2D(
            data=target_data,
            slice_idx=3,
            start_row=50,
            start_col=75,
            patch_size=(32, 32),
            original_slice_shape=(512, 512)
        )
        
        self.training_pair = TrainingPair2D(
            input_patch=self.input_patch,
            target_patch=self.target_patch
        )
    
    def test_training_pair_creation(self):
        """Test TrainingPair2D creation and properties."""
        assert self.training_pair.slice_idx == 3
        assert self.training_pair.patch_size == (32, 32)
        assert self.training_pair.position == (50, 75)
        assert len(self.training_pair.augmentation_applied) == 0
    
    def test_training_pair_mismatched_patches(self):
        """Test TrainingPair2D with mismatched patches."""
        # Different slice indices
        wrong_target = Patch2D(
            data=np.random.rand(32, 32),
            slice_idx=4,  # Different slice
            start_row=50,
            start_col=75,
            patch_size=(32, 32),
            original_slice_shape=(512, 512)
        )
        
        with pytest.raises(ValueError, match="same slice"):
            TrainingPair2D(
                input_patch=self.input_patch,
                target_patch=wrong_target
            )
        
        # Different positions
        wrong_position = Patch2D(
            data=np.random.rand(32, 32),
            slice_idx=3,
            start_row=60,  # Different position
            start_col=75,
            patch_size=(32, 32),
            original_slice_shape=(512, 512)
        )
        
        with pytest.raises(ValueError, match="same position"):
            TrainingPair2D(
                input_patch=self.input_patch,
                target_patch=wrong_position
            )
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_training_pair_to_tensors(self):
        """Test conversion to PyTorch tensors."""
        input_tensor, target_tensor = self.training_pair.to_tensors()
        
        assert isinstance(input_tensor, torch.Tensor)
        assert isinstance(target_tensor, torch.Tensor)
        assert input_tensor.shape == (32, 32)
        assert target_tensor.shape == (32, 32)
        assert input_tensor.device.type == 'cpu'
        assert target_tensor.device.type == 'cpu'
    
    def test_training_pair_augmentation(self):
        """Test augmentation application."""
        augmented = self.training_pair.apply_augmentation('horizontal_flip')
        
        assert isinstance(augmented, TrainingPair2D)
        assert 'horizontal_flip' in augmented.augmentation_applied
        assert len(augmented.augmentation_applied) == 1
        
        # Apply another augmentation
        double_augmented = augmented.apply_augmentation('rotation_90')
        assert len(double_augmented.augmentation_applied) == 2
        assert 'horizontal_flip' in double_augmented.augmentation_applied
        assert 'rotation_90' in double_augmented.augmentation_applied


class TestSliceStack:
    """Test cases for SliceStack class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_3d = np.random.rand(10, 256, 256).astype(np.float32)
        self.slice_stack = SliceStack(self.data_3d)
    
    def test_slice_stack_creation(self):
        """Test SliceStack creation and basic properties."""
        assert self.slice_stack.shape == (10, 256, 256)
        assert self.slice_stack.num_slices == 10
        assert self.slice_stack.slice_shape == (256, 256)
        assert self.slice_stack.dtype == np.float32
        assert self.slice_stack.device == 'cpu'
    
    def test_slice_stack_invalid_data(self):
        """Test SliceStack with invalid data."""
        # 2D data should fail
        with pytest.raises(ValueError, match="Data must be 3D"):
            SliceStack(np.random.rand(256, 256))
        
        # 4D data should fail
        with pytest.raises(ValueError, match="Data must be 3D"):
            SliceStack(np.random.rand(10, 256, 256, 3))
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_slice_stack_with_tensor(self):
        """Test SliceStack with PyTorch tensor."""
        tensor_data = torch.randn(5, 128, 128)
        stack = SliceStack(tensor_data)
        
        assert stack.shape == (5, 128, 128)
        assert stack.device == 'cpu'
        assert isinstance(stack.to_tensor(), torch.Tensor)
        assert isinstance(stack.to_numpy(), np.ndarray)
    
    def test_slice_access(self):
        """Test slice access methods."""
        # Get single slice
        slice_0 = self.slice_stack.get_slice(0)
        assert slice_0.shape == (256, 256)
        np.testing.assert_array_equal(slice_0, self.data_3d[0])
        
        # Get multiple slices
        slices = self.slice_stack.get_slices([0, 2, 4])
        assert slices.shape == (3, 256, 256)
        np.testing.assert_array_equal(slices[0], self.data_3d[0])
        np.testing.assert_array_equal(slices[1], self.data_3d[2])
        
        # Invalid slice index
        with pytest.raises(IndexError):
            self.slice_stack.get_slice(15)
    
    def test_slice_modification(self):
        """Test slice modification."""
        new_slice = np.ones((256, 256), dtype=np.float32)
        self.slice_stack.set_slice(0, new_slice)
        
        retrieved_slice = self.slice_stack.get_slice(0)
        np.testing.assert_array_equal(retrieved_slice, new_slice)
        
        # Invalid slice shape
        with pytest.raises(ValueError, match="Slice shape"):
            self.slice_stack.set_slice(0, np.ones((128, 128)))
    
    def test_slice_iteration(self):
        """Test slice iteration."""
        slice_count = 0
        for idx, slice_data in self.slice_stack.iter_slices():
            assert isinstance(idx, int)
            assert slice_data.shape == (256, 256)
            assert 0 <= idx < self.slice_stack.num_slices
            slice_count += 1
        
        assert slice_count == self.slice_stack.num_slices
    
    def test_slice_validation(self):
        """Test slice dimension validation."""
        # All slices should have same shape
        assert self.slice_stack.validate_slice_dimensions() is True
        
        # Test with specific expected shape
        assert self.slice_stack.validate_slice_dimensions((256, 256)) is True
        assert self.slice_stack.validate_slice_dimensions((128, 128)) is False
    
    def test_statistics(self):
        """Test statistics calculation."""
        stats = self.slice_stack.get_statistics()
        
        required_keys = [
            'shape', 'num_slices', 'slice_shape', 'dtype', 'device',
            'min_value', 'max_value', 'mean_value', 'std_value', 'total_pixels'
        ]
        
        for key in required_keys:
            assert key in stats
        
        assert stats['shape'] == (10, 256, 256)
        assert stats['num_slices'] == 10
        assert stats['slice_shape'] == (256, 256)
        assert stats['total_pixels'] == 10 * 256 * 256
    
    def test_patch_extraction(self):
        """Test patch extraction from slices."""
        patches = self.slice_stack.extract_patches_from_slice(
            slice_idx=0,
            patch_size=(64, 64),
            stride=(32, 32)
        )
        
        assert len(patches) > 0
        assert all(isinstance(p, Patch2D) for p in patches)
        assert all(p.slice_idx == 0 for p in patches)
        assert all(p.patch_size == (64, 64) for p in patches)
        
        # Check patch positions
        first_patch = patches[0]
        assert first_patch.start_row == 0
        assert first_patch.start_col == 0
    
    def test_slice_reconstruction(self):
        """Test slice reconstruction from patches."""
        # Extract patches
        patches = self.slice_stack.extract_patches_from_slice(
            slice_idx=0,
            patch_size=(64, 64),
            stride=(64, 64)  # No overlap for simpler testing
        )
        
        # Reconstruct slice
        reconstructed = self.slice_stack.reconstruct_slice_from_patches(
            patches=patches,
            slice_idx=0,
            blend_overlaps=False
        )
        
        assert reconstructed.shape == (256, 256)
        
        # With no overlap, reconstruction should be close to original
        # (some edge regions might not be covered by patches)
        original_slice = self.slice_stack.get_slice(0)
        
        # Check that covered regions match
        for patch in patches:
            start_row, start_col = patch.start_row, patch.start_col
            end_row, end_col = patch.end_row, patch.end_col
            
            original_region = original_slice[start_row:end_row, start_col:end_col]
            reconstructed_region = reconstructed[start_row:end_row, start_col:end_col]
            
            np.testing.assert_array_almost_equal(original_region, reconstructed_region)
    
    def test_metadata_handling(self):
        """Test metadata handling."""
        metadata = {'source': 'test_data', 'quality': 'high'}
        stack = SliceStack(self.data_3d, metadata=metadata)
        
        assert stack.metadata['source'] == 'test_data'
        assert stack.metadata['quality'] == 'high'
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_tensor_conversion(self):
        """Test tensor conversion methods."""
        # Convert to tensor
        tensor = self.slice_stack.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (10, 256, 256)
        
        # Convert back to numpy
        numpy_data = self.slice_stack.to_numpy()
        assert isinstance(numpy_data, np.ndarray)
        np.testing.assert_array_equal(numpy_data, self.data_3d)
        
        # Test device placement
        if torch.cuda.is_available():
            cuda_tensor = self.slice_stack.to_tensor('cuda')
            assert cuda_tensor.device.type == 'cuda'


class TestDataModelIntegration:
    """Integration tests for data models."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.volume_data = np.random.rand(5, 128, 128).astype(np.float32)
        self.slice_stack = SliceStack(self.volume_data)
    
    def test_patch_extraction_and_reconstruction_workflow(self):
        """Test complete patch extraction and reconstruction workflow."""
        slice_idx = 2
        patch_size = (32, 32)
        stride = (16, 16)  # 50% overlap
        
        # Extract patches
        patches = self.slice_stack.extract_patches_from_slice(
            slice_idx=slice_idx,
            patch_size=patch_size,
            stride=stride
        )
        
        assert len(patches) > 0
        
        # Simulate processing (just add small noise)
        processed_patches = []
        for patch in patches:
            processed_data = patch.data + np.random.normal(0, 0.01, patch.data.shape)
            processed_patch = Patch2D(
                data=processed_data,
                slice_idx=patch.slice_idx,
                start_row=patch.start_row,
                start_col=patch.start_col,
                patch_size=patch.patch_size,
                original_slice_shape=patch.original_slice_shape
            )
            processed_patches.append(processed_patch)
        
        # Reconstruct slice
        reconstructed = self.slice_stack.reconstruct_slice_from_patches(
            patches=processed_patches,
            slice_idx=slice_idx,
            blend_overlaps=True
        )
        
        assert reconstructed.shape == self.slice_stack.slice_shape
        
        # Update slice stack
        self.slice_stack.set_slice(slice_idx, reconstructed)
        
        # Verify update
        updated_slice = self.slice_stack.get_slice(slice_idx)
        np.testing.assert_array_equal(updated_slice, reconstructed)
    
    def test_training_pair_creation_workflow(self):
        """Test training pair creation workflow."""
        # Create input and target slice stacks
        input_stack = SliceStack(np.random.rand(3, 64, 64).astype(np.float32))
        target_stack = SliceStack(np.random.rand(3, 64, 64).astype(np.float32))
        
        training_pairs = []
        
        for slice_idx in range(input_stack.num_slices):
            # Extract patches from both stacks
            input_patches = input_stack.extract_patches_from_slice(
                slice_idx=slice_idx,
                patch_size=(32, 32),
                stride=(32, 32)
            )
            
            target_patches = target_stack.extract_patches_from_slice(
                slice_idx=slice_idx,
                patch_size=(32, 32),
                stride=(32, 32)
            )
            
            # Create training pairs
            assert len(input_patches) == len(target_patches)
            
            for input_patch, target_patch in zip(input_patches, target_patches):
                training_pair = TrainingPair2D(
                    input_patch=input_patch,
                    target_patch=target_patch
                )
                training_pairs.append(training_pair)
        
        assert len(training_pairs) > 0
        
        # Test that all training pairs are valid
        for pair in training_pairs:
            assert isinstance(pair, TrainingPair2D)
            assert pair.input_patch.slice_idx == pair.target_patch.slice_idx
            assert pair.input_patch.patch_size == pair.target_patch.patch_size
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_integration(self):
        """Test PyTorch integration across all data models."""
        # Create tensor-based slice stack
        tensor_data = torch.randn(3, 64, 64)
        tensor_stack = SliceStack(tensor_data)
        
        # Extract patches
        patches = tensor_stack.extract_patches_from_slice(
            slice_idx=0,
            patch_size=(32, 32),
            stride=(32, 32)
        )
        
        # Convert patches to tensors
        tensor_patches = [patch.to_tensor() for patch in patches]
        
        assert all(isinstance(t, torch.Tensor) for t in tensor_patches)
        assert all(t.shape == (32, 32) for t in tensor_patches)
        
        # Test training pair tensor conversion
        if len(patches) >= 1:
            # Create a target patch with same position as input patch
            input_patch = patches[0]
            target_data = torch.randn_like(input_patch.data)
            target_patch = Patch2D(
                data=target_data,
                slice_idx=input_patch.slice_idx,
                start_row=input_patch.start_row,
                start_col=input_patch.start_col,
                patch_size=input_patch.patch_size,
                original_slice_shape=input_patch.original_slice_shape
            )
            
            training_pair = TrainingPair2D(
                input_patch=input_patch,
                target_patch=target_patch
            )
            
            input_tensor, target_tensor = training_pair.to_tensors()
            assert isinstance(input_tensor, torch.Tensor)
            assert isinstance(target_tensor, torch.Tensor)


if __name__ == "__main__":
    pytest.main([__file__])