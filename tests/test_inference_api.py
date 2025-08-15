"""
Unit tests for ImageEnhancementAPI.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.inference.api import ImageEnhancementAPI
from src.core.config import InferenceConfig, EnhancementResult
from src.core.data_models import SliceStack


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestImageEnhancementAPI:
    """Test ImageEnhancementAPI functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = InferenceConfig(
            patch_size=(256, 256),
            overlap=32,
            batch_size=4,
            device='cpu',
            memory_limit_gb=2.0,
            enable_quality_metrics=True
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_api_init(self):
        """Test API initialization."""
        api = ImageEnhancementAPI(self.config)
        
        assert api.config == self.config
        assert api.image_processor is not None
        assert api.enhancement_processor is None
        assert api.model_loaded is False
        assert api.model_path is None
    
    def test_api_init_default_config(self):
        """Test API initialization with default config."""
        api = ImageEnhancementAPI()
        
        assert api.config is not None
        assert isinstance(api.config, InferenceConfig)
        assert api.image_processor is not None
    
    def test_load_model_success(self):
        """Test successful model loading."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock enhancement processor
        mock_processor = Mock()
        mock_processor.load_model = Mock()
        mock_processor.validate_model_compatibility = Mock(return_value=True)
        mock_processor.get_model_info = Mock(return_value={'loaded': True})
        
        with patch('src.inference.api.EnhancementProcessor', return_value=mock_processor):
            result = api.load_model("test_model.pth")
        
        assert result is True
        assert api.model_loaded is True
        assert api.model_path == "test_model.pth"
        mock_processor.load_model.assert_called_once_with("test_model.pth")
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock enhancement processor that fails
        mock_processor = Mock()
        mock_processor.load_model = Mock(side_effect=Exception("Model load failed"))
        
        with patch('src.inference.api.EnhancementProcessor', return_value=mock_processor):
            result = api.load_model("invalid_model.pth")
        
        assert result is False
        assert api.model_loaded is False
    
    def test_load_model_incompatible(self):
        """Test loading incompatible model."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock enhancement processor with incompatible model
        mock_processor = Mock()
        mock_processor.load_model = Mock()
        mock_processor.validate_model_compatibility = Mock(return_value=False)
        
        with patch('src.inference.api.EnhancementProcessor', return_value=mock_processor):
            result = api.load_model("incompatible_model.pth")
        
        assert result is False
        assert api.model_loaded is False
    
    @patch('src.inference.api.Path')
    def test_enhance_3d_tiff_file_not_found(self, mock_path):
        """Test enhancement with non-existent input file."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock file not existing
        mock_path.return_value.exists.return_value = False
        
        result = api.enhance_3d_tiff("nonexistent.tif", "output.tif")
        
        assert result.success is False
        assert "not found" in result.error_message
    
    def test_enhance_3d_tiff_no_model(self):
        """Test enhancement without loaded model."""
        api = ImageEnhancementAPI(self.config)
        
        # Create dummy input file
        input_path = Path(self.temp_dir) / "input.tif"
        input_path.touch()
        
        result = api.enhance_3d_tiff(str(input_path), "output.tif")
        
        assert result.success is False
        assert "No model loaded" in result.error_message
    
    @patch('src.inference.api.ImageProcessor')
    @patch('src.inference.api.EnhancementProcessor')
    def test_enhance_3d_tiff_success(self, mock_enhancement_class, mock_processor_class):
        """Test successful TIFF enhancement."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        mock_enhancement = Mock()
        mock_enhancement_class.return_value = mock_enhancement
        
        # Mock volume data
        test_data = np.random.rand(2, 256, 256).astype(np.float32)
        mock_volume = SliceStack(test_data)
        
        mock_processor.load_3d_volume.return_value = mock_volume
        mock_processor.check_memory_constraints.return_value = True
        mock_processor.preprocess_volume.return_value = mock_volume
        mock_processor.postprocess_volume.return_value = mock_volume
        mock_processor.extract_patches_for_inference.return_value = (torch.randn(4, 256, 256), [])
        mock_processor.reconstruct_volume_from_patches.return_value = mock_volume
        mock_processor.save_volume.return_value = None
        mock_processor.get_volume_quality_metrics.return_value = {'psnr': 30.0}
        
        mock_enhancement.enhance_volume_patches.return_value = torch.randn(4, 256, 256)
        
        api = ImageEnhancementAPI(self.config)
        api.model_loaded = True
        api.enhancement_processor = mock_enhancement
        
        # Create dummy input file
        input_path = Path(self.temp_dir) / "input.tif"
        input_path.touch()
        output_path = str(Path(self.temp_dir) / "output.tif")
        
        result = api.enhance_3d_tiff(str(input_path), output_path)
        
        assert result.success is True
        assert result.processing_time > 0
        assert result.quality_metrics is not None
        assert result.output_path == output_path
    
    def test_enhance_3d_array_invalid_input(self):
        """Test array enhancement with invalid input."""
        api = ImageEnhancementAPI(self.config)
        
        # Test with 2D array (should be 3D)
        invalid_array = np.random.rand(256, 256)
        result = api.enhance_3d_array(invalid_array)
        
        assert result.success is False
        assert "must be 3D" in result.error_message
    
    def test_enhance_3d_array_no_model(self):
        """Test array enhancement without loaded model."""
        api = ImageEnhancementAPI(self.config)
        
        test_array = np.random.rand(2, 256, 256).astype(np.float32)
        result = api.enhance_3d_array(test_array)
        
        assert result.success is False
        assert "No model loaded" in result.error_message
    
    @patch('src.inference.api.ImageProcessor')
    @patch('src.inference.api.EnhancementProcessor')
    def test_enhance_3d_array_success(self, mock_enhancement_class, mock_processor_class):
        """Test successful array enhancement."""
        # Setup mocks
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        mock_enhancement = Mock()
        mock_enhancement_class.return_value = mock_enhancement
        
        # Mock processing
        test_data = np.random.rand(2, 256, 256).astype(np.float32)
        mock_volume = SliceStack(test_data)
        
        mock_processor.check_memory_constraints.return_value = True
        mock_processor.preprocess_volume.return_value = mock_volume
        mock_processor.postprocess_volume.return_value = mock_volume
        mock_processor.extract_patches_for_inference.return_value = (torch.randn(4, 256, 256), [])
        mock_processor.reconstruct_volume_from_patches.return_value = mock_volume
        mock_processor.get_volume_quality_metrics.return_value = {'psnr': 30.0}
        
        mock_enhancement.enhance_volume_patches.return_value = torch.randn(4, 256, 256)
        
        api = ImageEnhancementAPI(self.config)
        api.model_loaded = True
        api.enhancement_processor = mock_enhancement
        
        result = api.enhance_3d_array(test_data)
        
        assert result.success is True
        assert result.processing_time > 0
        assert result.quality_metrics is not None
        assert 'enhanced_array' in result.quality_metrics
    
    def test_get_enhancement_metrics(self):
        """Test getting enhancement metrics."""
        api = ImageEnhancementAPI(self.config)
        
        # Create test result
        result = EnhancementResult(
            success=True,
            processing_time=10.0,
            input_shape=(5, 1024, 1024),
            output_path="output.tif"
        )
        result.quality_metrics = {'psnr': 30.0}
        
        metrics = api.get_enhancement_metrics(result)
        
        assert metrics['success'] is True
        assert metrics['processing_time'] == 10.0
        assert metrics['input_shape'] == (5, 1024, 1024)
        assert 'processing_efficiency' in metrics
        assert 'pixels_per_second' in metrics['processing_efficiency']
    
    def test_set_processing_config(self):
        """Test updating processing configuration."""
        api = ImageEnhancementAPI(self.config)
        
        # Update config
        api.set_processing_config(batch_size=8, memory_limit_gb=4.0)
        
        assert api.config.batch_size == 8
        assert api.config.memory_limit_gb == 4.0
    
    def test_set_processing_config_device_change(self):
        """Test config update that requires processor reinitialization."""
        api = ImageEnhancementAPI(self.config)
        original_processor = api.image_processor
        
        # Update device (should reinitialize processors)
        api.set_processing_config(device='cpu')
        
        # Should have new processor instance
        assert api.image_processor is not original_processor
    
    def test_get_system_info(self):
        """Test getting system information."""
        api = ImageEnhancementAPI(self.config)
        
        info = api.get_system_info()
        
        assert 'torch_available' in info
        assert 'device' in info
        assert 'config' in info
        assert 'model_loaded' in info
        
        if TORCH_AVAILABLE:
            assert 'torch_info' in info
            assert 'version' in info['torch_info']
    
    def test_cleanup(self):
        """Test resource cleanup."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock enhancement processor
        mock_processor = Mock()
        api.enhancement_processor = mock_processor
        api.model_loaded = True
        api.model_path = "test.pth"
        
        # Cleanup
        api.cleanup()
        
        mock_processor.cleanup.assert_called_once()
        assert api.model_loaded is False
        assert api.model_path is None
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with ImageEnhancementAPI(self.config) as api:
            assert api is not None
            api.model_loaded = True  # Simulate loaded state
        
        # Should be cleaned up after context exit
        assert api.model_loaded is False
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        api = ImageEnhancementAPI(self.config)
        
        progress_calls = []
        
        def progress_callback(message: str, progress: float):
            progress_calls.append((message, progress))
        
        # Test with array enhancement (mocked)
        with patch.object(api, '_enhance_volume_with_chunking') as mock_enhance:
            mock_enhance.return_value = SliceStack(np.random.rand(2, 64, 64))
            
            api.model_loaded = True
            test_array = np.random.rand(2, 64, 64).astype(np.float32)
            
            result = api.enhance_3d_array(test_array, progress_callback=progress_callback)
            
            # Should have received progress callbacks
            assert len(progress_calls) > 0
            assert any("completed" in call[0] for call in progress_calls)
    
    def test_chunked_processing(self):
        """Test chunked processing for large volumes."""
        api = ImageEnhancementAPI(self.config)
        
        # Mock processors
        with patch.object(api.image_processor, 'check_memory_constraints', return_value=False):
            with patch.object(api.image_processor, 'get_processing_chunks', return_value=[(0, 2), (2, 4)]):
                with patch.object(api, '_enhance_volume_direct') as mock_direct:
                    mock_direct.return_value = SliceStack(np.random.rand(2, 64, 64))
                    
                    test_volume = SliceStack(np.random.rand(4, 64, 64))
                    result = api._enhance_volume_chunked(test_volume)
                    
                    # Should have called direct enhancement for each chunk
                    assert mock_direct.call_count == 2
                    assert isinstance(result, SliceStack)


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_api_without_torch():
    """Test API behavior when PyTorch is not available."""
    from src.inference.api import ImageEnhancementAPI
    from src.core.config import InferenceConfig
    
    config = InferenceConfig()
    api = ImageEnhancementAPI(config)
    
    # Should initialize but with warnings
    assert api.config == config
    
    # Model loading should fail
    result = api.load_model("test_model.pth")
    assert result is False