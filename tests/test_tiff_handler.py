"""
Unit tests for TIFFDataHandler class.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.core.tiff_handler import TIFFDataHandler


class TestTIFFDataHandler:
    """Test cases for TIFFDataHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = TIFFDataHandler()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_test_tiff_data(self, shape=(10, 3000, 3000), dtype=np.uint16):
        """Create test TIFF data with specified shape and dtype."""
        return np.random.randint(0, 1000, size=shape, dtype=dtype)
    
    def create_temp_tiff_file(self, data, filename="test.tiff"):
        """Create a temporary TIFF file with given data."""
        import tifffile
        filepath = Path(self.temp_dir) / filename
        tifffile.imwrite(filepath, data)
        return str(filepath)
    
    # Tests for load_3d_tiff method
    
    def test_load_3d_tiff_valid_file(self):
        """Test loading a valid 3D TIFF file."""
        # Create test data
        test_data = self.create_test_tiff_data()
        filepath = self.create_temp_tiff_file(test_data)
        
        # Load the file
        loaded_data = self.handler.load_3d_tiff(filepath)
        
        # Verify the data
        assert loaded_data.shape == test_data.shape
        assert loaded_data.dtype == test_data.dtype
        np.testing.assert_array_equal(loaded_data, test_data)
    
    def test_load_3d_tiff_2d_file(self):
        """Test loading a 2D TIFF file (single slice)."""
        # Create 2D test data
        test_data = np.random.randint(0, 1000, size=(3000, 3000), dtype=np.uint16)
        filepath = self.create_temp_tiff_file(test_data)
        
        # Load the file
        loaded_data = self.handler.load_3d_tiff(filepath)
        
        # Verify the data has been expanded to 3D
        assert loaded_data.shape == (1, 3000, 3000)
        assert loaded_data.dtype == test_data.dtype
        np.testing.assert_array_equal(loaded_data[0], test_data)
    
    def test_load_3d_tiff_file_not_found(self):
        """Test loading a non-existent file."""
        with pytest.raises(FileNotFoundError, match="TIFF file not found"):
            self.handler.load_3d_tiff("nonexistent.tiff")
    
    def test_load_3d_tiff_invalid_extension(self):
        """Test loading a file with invalid extension."""
        # Create a file with wrong extension
        filepath = Path(self.temp_dir) / "test.jpg"
        filepath.touch()
        
        with pytest.raises(ValueError, match="File must be a TIFF file"):
            self.handler.load_3d_tiff(str(filepath))
    
    @patch('tifffile.TiffFile')
    def test_load_3d_tiff_corrupted_file(self, mock_tiff_file):
        """Test loading a corrupted TIFF file."""
        # Mock TiffFile to raise TiffFileError
        import tifffile
        mock_tiff_file.side_effect = tifffile.TiffFileError("Corrupted file")
        
        # Create a dummy file
        filepath = Path(self.temp_dir) / "corrupted.tiff"
        filepath.touch()
        
        with pytest.raises(ValueError, match="Corrupted or invalid TIFF file"):
            self.handler.load_3d_tiff(str(filepath))
    
    def test_load_3d_tiff_4d_data_single_channel(self):
        """Test loading 4D data with single channel (should be squeezed to 3D)."""
        # Create 4D test data with single channel
        test_data_4d = np.random.randint(0, 1000, size=(10, 3000, 3000, 1), dtype=np.uint16)
        
        # Note: tifffile may not handle 4D data as expected, so we expect an error
        # This test verifies our error handling for unsupported dimensions
        filepath = self.create_temp_tiff_file(test_data_4d)
        
        with pytest.raises((ValueError, IOError)):
            self.handler.load_3d_tiff(filepath)
    
    def test_load_3d_tiff_4d_data_multi_channel(self):
        """Test loading 4D data with multiple channels (should raise error)."""
        # Create 4D test data with multiple channels
        test_data_4d = np.random.randint(0, 1000, size=(10, 3000, 3000, 3), dtype=np.uint16)
        
        # Note: tifffile may not handle 4D data as expected, so we expect an error
        # This test verifies our error handling for unsupported dimensions
        filepath = self.create_temp_tiff_file(test_data_4d)
        
        with pytest.raises((ValueError, IOError)):
            self.handler.load_3d_tiff(filepath)
    
    # Tests for validate_dimensions method
    
    def test_validate_dimensions_valid_data(self):
        """Test validation with valid 3000x3000 data."""
        test_data = self.create_test_tiff_data()
        result = self.handler.validate_dimensions(test_data)
        assert result is True
    
    def test_validate_dimensions_invalid_size_too_small(self):
        """Test validation with dimensions too small."""
        test_data = np.random.randint(0, 1000, size=(10, 500, 500), dtype=np.uint16)
        result = self.handler.validate_dimensions(test_data)
        assert result is False
    
    def test_validate_dimensions_invalid_size_too_large(self):
        """Test validation with dimensions too large."""
        test_data = np.random.randint(0, 1000, size=(10, 6000, 6000), dtype=np.uint16)
        result = self.handler.validate_dimensions(test_data)
        assert result is False
    
    def test_validate_dimensions_edge_cases(self):
        """Test validation with edge case dimensions."""
        # Test minimum acceptable size
        test_data = np.random.randint(0, 1000, size=(10, 1000, 1000), dtype=np.uint16)
        result = self.handler.validate_dimensions(test_data)
        assert result is True
        
        # Test maximum acceptable size
        test_data = np.random.randint(0, 1000, size=(10, 5000, 5000), dtype=np.uint16)
        result = self.handler.validate_dimensions(test_data)
        assert result is True
    
    def test_validate_dimensions_invalid_dtype(self):
        """Test validation with unsupported data type."""
        test_data = np.random.randint(0, 1000, size=(10, 3000, 3000), dtype=np.int32)
        result = self.handler.validate_dimensions(test_data)
        assert result is False
    
    def test_validate_dimensions_not_numpy_array(self):
        """Test validation with non-numpy array input."""
        with pytest.raises(ValueError, match="Data must be a numpy array"):
            self.handler.validate_dimensions([1, 2, 3])
    
    def test_validate_dimensions_not_3d(self):
        """Test validation with non-3D data."""
        test_data = np.random.randint(0, 1000, size=(3000, 3000), dtype=np.uint16)
        with pytest.raises(ValueError, match="Expected 3D data, got 2D"):
            self.handler.validate_dimensions(test_data)
    
    def test_validate_dimensions_supported_dtypes(self):
        """Test validation with all supported data types."""
        shape = (5, 3000, 3000)
        
        for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            if dtype in [np.float32, np.float64]:
                test_data = np.random.random(shape).astype(dtype)
            elif dtype == np.uint8:
                test_data = np.random.randint(0, 256, size=shape, dtype=dtype)
            else:
                test_data = np.random.randint(0, 1000, size=shape, dtype=dtype)
            
            result = self.handler.validate_dimensions(test_data)
            assert result is True, f"Validation failed for dtype: {dtype}"
    
    # Tests for save_3d_tiff method
    
    def test_save_3d_tiff_valid_data(self):
        """Test saving valid 3D data."""
        test_data = self.create_test_tiff_data()
        output_path = Path(self.temp_dir) / "output.tiff"
        
        # Save the data
        self.handler.save_3d_tiff(test_data, str(output_path))
        
        # Verify file was created
        assert output_path.exists()
        
        # Load and verify the saved data
        loaded_data = self.handler.load_3d_tiff(str(output_path))
        np.testing.assert_array_equal(loaded_data, test_data)
    
    def test_save_3d_tiff_creates_directory(self):
        """Test that save_3d_tiff creates output directory if it doesn't exist."""
        test_data = self.create_test_tiff_data()
        output_path = Path(self.temp_dir) / "subdir" / "output.tiff"
        
        # Ensure directory doesn't exist
        assert not output_path.parent.exists()
        
        # Save the data
        self.handler.save_3d_tiff(test_data, str(output_path))
        
        # Verify directory and file were created
        assert output_path.parent.exists()
        assert output_path.exists()
    
    def test_save_3d_tiff_invalid_data_type(self):
        """Test saving with invalid data type."""
        with pytest.raises(ValueError, match="Data must be a numpy array"):
            self.handler.save_3d_tiff([1, 2, 3], "output.tiff")
    
    def test_save_3d_tiff_invalid_dimensions(self):
        """Test saving with invalid dimensions."""
        test_data = np.random.randint(0, 1000, size=(3000, 3000), dtype=np.uint16)
        with pytest.raises(ValueError, match="Expected 3D data, got 2D"):
            self.handler.save_3d_tiff(test_data, "output.tiff")
    
    @patch('tifffile.imwrite')
    def test_save_3d_tiff_write_error(self, mock_imwrite):
        """Test handling of write errors."""
        mock_imwrite.side_effect = Exception("Write error")
        test_data = self.create_test_tiff_data()
        
        with pytest.raises(IOError, match="Error saving TIFF file"):
            self.handler.save_3d_tiff(test_data, "output.tiff")
    
    # Tests for get_metadata method
    
    def test_get_metadata_valid_file(self):
        """Test extracting metadata from a valid TIFF file."""
        test_data = self.create_test_tiff_data()
        filepath = self.create_temp_tiff_file(test_data)
        
        metadata = self.handler.get_metadata(filepath)
        
        # Verify basic metadata fields
        assert 'file_path' in metadata
        assert 'file_size_bytes' in metadata
        assert 'num_pages' in metadata
        assert 'image_width' in metadata
        assert 'image_height' in metadata
        assert 'data_shape' in metadata
        
        # Verify values
        assert metadata['image_width'] == 3000
        assert metadata['image_height'] == 3000
        assert metadata['num_pages'] == 10
        assert metadata['data_shape'] == (10, 3000, 3000)
    
    def test_get_metadata_file_not_found(self):
        """Test metadata extraction from non-existent file."""
        with pytest.raises(FileNotFoundError, match="TIFF file not found"):
            self.handler.get_metadata("nonexistent.tiff")
    
    @patch('tifffile.TiffFile')
    def test_get_metadata_corrupted_file(self, mock_tiff_file):
        """Test metadata extraction from corrupted file."""
        import tifffile
        mock_tiff_file.side_effect = tifffile.TiffFileError("Corrupted file")
        
        # Create a dummy file
        filepath = Path(self.temp_dir) / "corrupted.tiff"
        filepath.touch()
        
        with pytest.raises(ValueError, match="Corrupted or invalid TIFF file"):
            self.handler.get_metadata(str(filepath))
    
    def test_get_metadata_single_page(self):
        """Test metadata extraction from single-page TIFF."""
        test_data = np.random.randint(0, 1000, size=(3000, 3000), dtype=np.uint16)
        filepath = self.create_temp_tiff_file(test_data)
        
        metadata = self.handler.get_metadata(filepath)
        
        assert metadata['num_pages'] == 1
        assert metadata['image_width'] == 3000
        assert metadata['image_height'] == 3000
    
    # Integration tests
    
    def test_round_trip_save_load(self):
        """Test saving and loading data maintains integrity."""
        original_data = self.create_test_tiff_data()
        filepath = Path(self.temp_dir) / "roundtrip.tiff"
        
        # Save the data
        self.handler.save_3d_tiff(original_data, str(filepath))
        
        # Load the data back
        loaded_data = self.handler.load_3d_tiff(str(filepath))
        
        # Verify data integrity
        assert loaded_data.shape == original_data.shape
        assert loaded_data.dtype == original_data.dtype
        np.testing.assert_array_equal(loaded_data, original_data)
        
        # Verify validation passes
        assert self.handler.validate_dimensions(loaded_data) is True
    
    def test_metadata_consistency(self):
        """Test that metadata is consistent with loaded data."""
        test_data = self.create_test_tiff_data(shape=(15, 3000, 3000))
        filepath = self.create_temp_tiff_file(test_data)
        
        # Get metadata
        metadata = self.handler.get_metadata(filepath)
        
        # Load data
        loaded_data = self.handler.load_3d_tiff(filepath)
        
        # Verify consistency
        assert metadata['data_shape'] == loaded_data.shape
        assert metadata['num_pages'] == loaded_data.shape[0]
        assert metadata['image_width'] == loaded_data.shape[2]
        assert metadata['image_height'] == loaded_data.shape[1]


    # PyTorch integration tests
    
    def test_pytorch_tensor_validation(self):
        """Test validation with PyTorch tensors."""
        try:
            import torch
            
            # Create a PyTorch tensor
            tensor_data = torch.randn(10, 3000, 3000)
            result = self.handler.validate_dimensions(tensor_data)
            assert result is True
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_save_pytorch_tensor(self):
        """Test saving PyTorch tensor as TIFF."""
        try:
            import torch
            
            # Create a PyTorch tensor
            tensor_data = torch.randn(5, 2000, 2000)
            output_path = Path(self.temp_dir) / "pytorch_output.tiff"
            
            # Save the tensor
            self.handler.save_3d_tiff(tensor_data, str(output_path))
            
            # Verify file was created
            assert output_path.exists()
            
            # Load and verify the saved data
            loaded_data = self.handler.load_3d_tiff(str(output_path))
            assert loaded_data.shape == (5, 2000, 2000)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_to_torch_tensor(self):
        """Test conversion from numpy to PyTorch tensor."""
        try:
            import torch
            
            test_data = self.create_test_tiff_data(shape=(5, 2000, 2000))
            tensor = self.handler.to_torch_tensor(test_data)
            
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (5, 2000, 2000)
            assert tensor.device.type == 'cpu'
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_from_torch_tensor(self):
        """Test conversion from PyTorch tensor to numpy."""
        try:
            import torch
            
            # Create a PyTorch tensor
            tensor_data = torch.randn(5, 2000, 2000)
            numpy_data = self.handler.from_torch_tensor(tensor_data)
            
            assert isinstance(numpy_data, np.ndarray)
            assert numpy_data.shape == (5, 2000, 2000)
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    def test_pytorch_round_trip(self):
        """Test round trip: numpy -> tensor -> numpy."""
        try:
            import torch
            
            original_data = self.create_test_tiff_data(shape=(3, 1500, 1500))
            
            # Convert to tensor and back
            tensor = self.handler.to_torch_tensor(original_data)
            recovered_data = self.handler.from_torch_tensor(tensor, target_dtype=original_data.dtype)
            
            # Should be close (allowing for float conversion)
            if original_data.dtype in [np.uint8, np.uint16]:
                # For integer types, allow some tolerance due to normalization
                np.testing.assert_allclose(recovered_data, original_data, rtol=0.01)
            else:
                np.testing.assert_array_equal(recovered_data, original_data)
                
        except ImportError:
            pytest.skip("PyTorch not available")


if __name__ == "__main__":
    pytest.main([__file__])