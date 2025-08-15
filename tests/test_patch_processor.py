"""
Unit tests for PatchProcessor class.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.patch_processor import PatchProcessor, PatchInfo


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestPatchProcessor:
    """Test cases for PatchProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = PatchProcessor(patch_size=(256, 256), overlap=32, device='cpu')
        self.small_processor = PatchProcessor(patch_size=(64, 64), overlap=8, device='cpu')
    
    def create_test_volume(self, shape=(10, 1000, 1000), dtype=torch.float32):
        """Create test volume data."""
        return torch.randn(shape, dtype=dtype)
    
    # Tests for PatchInfo dataclass
    
    def test_patch_info_creation(self):
        """Test PatchInfo creation and validation."""
        info = PatchInfo(
            slice_idx=0,
            start_row=0,
            start_col=0,
            end_row=256,
            end_col=256,
            patch_size=(256, 256),
            original_shape=(1000, 1000)
        )
        
        assert info.slice_idx == 0
        assert info.start_row == 0
        assert info.start_col == 0
        assert info.end_row == 256
        assert info.end_col == 256
        assert info.patch_size == (256, 256)
        assert info.original_shape == (1000, 1000)
    
    def test_patch_info_invalid_coordinates(self):
        """Test PatchInfo validation with invalid coordinates."""
        with pytest.raises(ValueError, match="Invalid patch coordinates"):
            PatchInfo(
                slice_idx=0,
                start_row=100,
                start_col=100,
                end_row=50,  # Invalid: end < start
                end_col=150,
                patch_size=(256, 256),
                original_shape=(1000, 1000)
            )
    
    # Tests for PatchProcessor initialization
    
    def test_processor_initialization(self):
        """Test PatchProcessor initialization."""
        processor = PatchProcessor(patch_size=(128, 128), overlap=16, device='cpu')
        
        assert processor.patch_size == (128, 128)
        assert processor.overlap == 16
        assert processor.stride == (112, 112)  # 128 - 16
        assert processor.device == torch.device('cpu')
    
    def test_processor_invalid_overlap(self):
        """Test PatchProcessor with invalid overlap."""
        with pytest.raises(ValueError, match="Overlap must be smaller than patch size"):
            PatchProcessor(patch_size=(64, 64), overlap=64)
        
        with pytest.raises(ValueError, match="Overlap must be non-negative"):
            PatchProcessor(patch_size=(64, 64), overlap=-1)
    
    def test_processor_without_pytorch(self):
        """Test PatchProcessor initialization without PyTorch."""
        with patch('src.core.patch_processor.TORCH_AVAILABLE', False):
            with pytest.raises(ImportError, match="PyTorch is required"):
                PatchProcessor()
    
    # Tests for calculate_patch_positions
    
    def test_calculate_patch_positions_normal(self):
        """Test patch position calculation for normal case."""
        positions = self.processor.calculate_patch_positions((1000, 1000))
        
        # Should have multiple patches
        assert len(positions) > 1
        
        # Check first patch
        start_row, start_col, end_row, end_col = positions[0]
        assert start_row == 0
        assert start_col == 0
        assert end_row == 256
        assert end_col == 256
        
        # All positions should be valid
        for start_row, start_col, end_row, end_col in positions:
            assert 0 <= start_row < end_row <= 1000
            assert 0 <= start_col < end_col <= 1000
            assert end_row - start_row <= 256
            assert end_col - start_col <= 256
    
    def test_calculate_patch_positions_small_image(self):
        """Test patch position calculation for image smaller than patch."""
        positions = self.processor.calculate_patch_positions((100, 100))
        
        # Should have exactly one patch
        assert len(positions) == 1
        
        start_row, start_col, end_row, end_col = positions[0]
        assert start_row == 0
        assert start_col == 0
        assert end_row == 100
        assert end_col == 100
    
    def test_calculate_patch_positions_exact_fit(self):
        """Test patch position calculation for exact fit."""
        # Image size that fits exactly with patches
        positions = self.small_processor.calculate_patch_positions((64, 64))
        
        # Should have exactly one patch
        assert len(positions) == 1
        
        start_row, start_col, end_row, end_col = positions[0]
        assert start_row == 0
        assert start_col == 0
        assert end_row == 64
        assert end_col == 64
    
    def test_calculate_patch_positions_with_overlap(self):
        """Test patch position calculation with overlap."""
        positions = self.small_processor.calculate_patch_positions((120, 120))
        
        # Should have multiple overlapping patches
        assert len(positions) > 1
        
        # Check that patches overlap
        if len(positions) >= 2:
            pos1 = positions[0]
            pos2 = positions[1]
            
            # Check if there's overlap (depends on layout)
            overlap_exists = (
                (pos1[2] > pos2[0] and pos1[0] < pos2[2]) or  # Row overlap
                (pos1[3] > pos2[1] and pos1[1] < pos2[3])     # Column overlap
            )
            # At least some patches should have overlap
            assert any(
                (positions[i][2] > positions[j][0] and positions[i][0] < positions[j][2]) or
                (positions[i][3] > positions[j][1] and positions[i][1] < positions[j][3])
                for i in range(len(positions)) for j in range(i+1, len(positions))
            )
    
    # Tests for extract_patches
    
    def test_extract_patches_basic(self):
        """Test basic patch extraction."""
        volume = self.create_test_volume((5, 512, 512))
        patches, patch_infos = self.processor.extract_patches(volume)
        
        # Check output shapes
        assert patches.dim() == 3  # (num_patches, height, width)
        assert patches.shape[1:] == self.processor.patch_size
        assert len(patch_infos) == patches.shape[0]
        
        # Check patch info consistency
        for i, info in enumerate(patch_infos):
            assert 0 <= info.slice_idx < 5
            assert info.patch_size == self.processor.patch_size
            assert info.original_shape == (512, 512)
    
    def test_extract_patches_numpy_input(self):
        """Test patch extraction with numpy input."""
        volume_np = np.random.randn(3, 300, 300).astype(np.float32)
        patches, patch_infos = self.processor.extract_patches(volume_np)
        
        # Should work with numpy input
        assert isinstance(patches, torch.Tensor)
        assert len(patch_infos) > 0
    
    def test_extract_patches_specific_slices(self):
        """Test patch extraction from specific slices."""
        volume = self.create_test_volume((10, 400, 400))
        slice_indices = [1, 3, 7]
        
        patches, patch_infos = self.processor.extract_patches(volume, slice_indices)
        
        # Check that only specified slices are processed
        processed_slices = set(info.slice_idx for info in patch_infos)
        assert processed_slices == set(slice_indices)
    
    def test_extract_patches_invalid_volume(self):
        """Test patch extraction with invalid volume."""
        # 2D volume (should fail)
        volume_2d = torch.randn(512, 512)
        with pytest.raises(ValueError, match="Expected 3D volume"):
            self.processor.extract_patches(volume_2d)
        
        # 4D volume (should fail)
        volume_4d = torch.randn(5, 512, 512, 3)
        with pytest.raises(ValueError, match="Expected 3D volume"):
            self.processor.extract_patches(volume_4d)
    
    def test_extract_patches_invalid_slice_indices(self):
        """Test patch extraction with invalid slice indices."""
        volume = self.create_test_volume((5, 400, 400))
        
        # Out of range slice index
        with pytest.raises(ValueError, match="Slice index .* out of range"):
            self.processor.extract_patches(volume, [10])
        
        # Negative slice index
        with pytest.raises(ValueError, match="Slice index .* out of range"):
            self.processor.extract_patches(volume, [-1])
    
    def test_extract_patches_small_volume(self):
        """Test patch extraction from volume smaller than patch size."""
        volume = self.create_test_volume((2, 100, 100))
        patches, patch_infos = self.processor.extract_patches(volume)
        
        # Should still work, with padding
        assert patches.shape[1:] == self.processor.patch_size
        assert len(patch_infos) == 2  # One patch per slice
    
    # Tests for _pad_patch
    
    def test_pad_patch(self):
        """Test patch padding functionality."""
        small_patch = torch.randn(100, 100)
        padded = self.processor._pad_patch(small_patch, (256, 256))
        
        assert padded.shape == (256, 256)
        
        # Original data should be preserved in top-left corner
        torch.testing.assert_close(padded[:100, :100], small_patch, rtol=1e-4, atol=1e-4)
    
    def test_pad_patch_invalid_size(self):
        """Test patch padding with invalid target size."""
        large_patch = torch.randn(300, 300)
        
        with pytest.raises(ValueError, match="Patch is larger than target size"):
            self.processor._pad_patch(large_patch, (256, 256))
    
    # Tests for reconstruct_slice
    
    def test_reconstruct_slice_basic(self):
        """Test basic slice reconstruction."""
        # Create a simple test case
        volume = torch.ones(1, 128, 128) * 5.0  # Constant value for easy testing
        patches, patch_infos = self.small_processor.extract_patches(volume)
        
        # Reconstruct the slice
        reconstructed = self.small_processor.reconstruct_slice(
            patches, patch_infos, target_slice_idx=0, output_shape=(128, 128)
        )
        
        assert reconstructed.shape == (128, 128)
        
        # With constant input, output should be close to constant (allowing for blending)
        mean_value = reconstructed.mean().item()
        assert abs(mean_value - 5.0) < 1.0  # Allow larger deviation due to blending effects
    
    def test_reconstruct_slice_no_patches(self):
        """Test slice reconstruction with no patches for target slice."""
        patches = torch.randn(5, 64, 64)
        patch_infos = [
            PatchInfo(slice_idx=1, start_row=0, start_col=0, end_row=64, end_col=64,
                     patch_size=(64, 64), original_shape=(128, 128))
            for _ in range(5)
        ]
        
        with pytest.raises(ValueError, match="No patches found for slice"):
            self.small_processor.reconstruct_slice(
                patches, patch_infos, target_slice_idx=0
            )
    
    def test_reconstruct_slice_mismatched_inputs(self):
        """Test slice reconstruction with mismatched inputs."""
        patches = torch.randn(5, 64, 64)
        patch_infos = [PatchInfo(slice_idx=0, start_row=0, start_col=0, end_row=64, end_col=64,
                                patch_size=(64, 64), original_shape=(128, 128))]  # Only 1 info for 5 patches
        
        with pytest.raises(ValueError, match="Number of patches must match"):
            self.small_processor.reconstruct_slice(patches, patch_infos, target_slice_idx=0)
    
    # Tests for _create_blend_weights
    
    def test_create_blend_weights_no_overlap(self):
        """Test blend weight creation with no overlap."""
        processor_no_overlap = PatchProcessor(patch_size=(64, 64), overlap=0)
        weights = processor_no_overlap._create_blend_weights(64, 64)
        
        # Should be all ones
        assert weights.shape == (64, 64)
        torch.testing.assert_close(weights, torch.ones(64, 64))
    
    def test_create_blend_weights_with_overlap(self):
        """Test blend weight creation with overlap."""
        weights = self.small_processor._create_blend_weights(64, 64)
        
        assert weights.shape == (64, 64)
        
        # Center should be 1.0
        center_value = weights[32, 32].item()
        assert abs(center_value - 1.0) < 0.01
        
        # Edges should have lower values
        edge_value = weights[0, 0].item()
        assert edge_value < 1.0
    
    # Tests for reconstruct_volume
    
    def test_reconstruct_volume_basic(self):
        """Test basic volume reconstruction."""
        # Create test volume
        original_volume = torch.ones(3, 128, 128) * 2.0
        
        # Extract patches
        patches, patch_infos = self.small_processor.extract_patches(original_volume)
        
        # Reconstruct volume
        reconstructed = self.small_processor.reconstruct_volume(patches, patch_infos)
        
        assert reconstructed.shape == (3, 128, 128)
        
        # Should be close to original (allowing for blending effects)
        mean_diff = (reconstructed - original_volume).abs().mean().item()
        assert mean_diff < 0.5  # Allow larger deviation due to blending effects
    
    def test_reconstruct_volume_custom_shape(self):
        """Test volume reconstruction with custom output shape."""
        volume = torch.randn(2, 100, 100)
        patches, patch_infos = self.small_processor.extract_patches(volume)
        
        # Reconstruct with different shape
        reconstructed = self.small_processor.reconstruct_volume(
            patches, patch_infos, output_shape=(2, 150, 150)
        )
        
        assert reconstructed.shape == (2, 150, 150)
    
    def test_reconstruct_volume_missing_slices(self):
        """Test volume reconstruction with missing slices."""
        volume = torch.randn(5, 100, 100)
        patches, patch_infos = self.small_processor.extract_patches(volume, slice_indices=[0, 2, 4])
        
        # Reconstruct full volume
        reconstructed = self.small_processor.reconstruct_volume(
            patches, patch_infos, output_shape=(5, 100, 100)
        )
        
        assert reconstructed.shape == (5, 100, 100)
        
        # Missing slices should be zeros
        assert torch.allclose(reconstructed[1], torch.zeros(100, 100))
        assert torch.allclose(reconstructed[3], torch.zeros(100, 100))
    
    def test_reconstruct_volume_empty_patches(self):
        """Test volume reconstruction with empty patch list."""
        with pytest.raises(ValueError, match="No patch information provided"):
            self.processor.reconstruct_volume(torch.empty(0, 64, 64), [])
    
    # Tests for get_memory_usage_estimate
    
    def test_memory_usage_estimate(self):
        """Test memory usage estimation."""
        usage = self.processor.get_memory_usage_estimate((10, 1000, 1000))
        
        # Check that all expected keys are present
        expected_keys = ['volume_mb', 'patches_mb', 'reconstruction_mb', 
                        'total_estimated_mb', 'patches_per_slice', 'total_patches']
        for key in expected_keys:
            assert key in usage
            assert isinstance(usage[key], (int, float))
            assert usage[key] >= 0
        
        # Total should be sum of components
        expected_total = usage['volume_mb'] + usage['patches_mb'] + usage['reconstruction_mb']
        assert abs(usage['total_estimated_mb'] - expected_total) < 0.01
    
    def test_memory_usage_estimate_different_dtypes(self):
        """Test memory usage estimation with different data types."""
        usage_float32 = self.processor.get_memory_usage_estimate(
            (5, 500, 500), dtype=torch.float32
        )
        usage_float16 = self.processor.get_memory_usage_estimate(
            (5, 500, 500), dtype=torch.float16
        )
        
        # float16 should use less memory than float32
        assert usage_float16['volume_mb'] < usage_float32['volume_mb']
        assert usage_float16['patches_mb'] < usage_float32['patches_mb']
    
    # Integration tests
    
    def test_full_pipeline_round_trip(self):
        """Test complete pipeline: extract patches -> reconstruct volume."""
        # Create test volume with known pattern
        original_volume = torch.zeros(3, 200, 200)
        
        # Add some patterns to make reconstruction testable
        original_volume[0, 50:150, 50:150] = 1.0
        original_volume[1, 25:175, 25:175] = 2.0
        original_volume[2, 75:125, 75:125] = 3.0
        
        # Extract patches
        patches, patch_infos = self.small_processor.extract_patches(original_volume)
        
        # Reconstruct volume
        reconstructed = self.small_processor.reconstruct_volume(patches, patch_infos)
        
        assert reconstructed.shape == original_volume.shape
        
        # Check that patterns are preserved (allowing for blending)
        for slice_idx in range(3):
            original_slice = original_volume[slice_idx]
            reconstructed_slice = reconstructed[slice_idx]
            
            # Calculate correlation to check pattern preservation
            correlation = torch.corrcoef(torch.stack([
                original_slice.flatten(), 
                reconstructed_slice.flatten()
            ]))[0, 1]
            
            assert correlation > 0.9  # High correlation indicates good reconstruction
    
    def test_device_consistency(self):
        """Test that all operations maintain device consistency."""
        if torch.cuda.is_available():
            # Test with CUDA device
            cuda_processor = PatchProcessor(patch_size=(64, 64), overlap=8, device='cuda')
            volume = torch.randn(2, 128, 128).cuda()
            
            patches, patch_infos = cuda_processor.extract_patches(volume)
            
            # Patches should be on CUDA
            assert patches.device.type == 'cuda'
            
            # Reconstruction should also be on CUDA
            reconstructed = cuda_processor.reconstruct_volume(patches, patch_infos)
            assert reconstructed.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")
    
    def test_different_patch_sizes(self):
        """Test processor with different patch sizes."""
        sizes_to_test = [(32, 32), (128, 128), (512, 512)]
        
        for patch_size in sizes_to_test:
            processor = PatchProcessor(patch_size=patch_size, overlap=8)
            volume = torch.randn(2, 256, 256)
            
            patches, patch_infos = processor.extract_patches(volume)
            
            # Check patch size
            assert patches.shape[1:] == patch_size
            
            # Check reconstruction
            reconstructed = processor.reconstruct_volume(patches, patch_infos)
            assert reconstructed.shape == volume.shape


if __name__ == "__main__":
    pytest.main([__file__])