"""
Unit tests for inference pipeline components.
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

from src.inference.image_processor import ImageProcessor
from src.core.config import InferenceConfig
from src.core.data_models import SliceStack


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestImageProcessor:
    """Test ImageProcessor functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = InferenceConfig(
            patch_size=(256, 256),
            overlap=32,
            batch_size=4,
            device='cpu',
            memory_limit_gb=2.0
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_image_processor_init(self):
        """Test ImageProcessor initialization."""
        processor = ImageProcessor(self.config)
        
        assert processor.config == self.config
        assert processor.device == 'cpu'
        assert processor.memory_limit_bytes == int(2.0 * 1024**3)
        assert processor.tiff_handler is not None
        assert processor.patch_processor is not None
    
    def test_device_resolution(self):
        """Test device resolution logic."""
        processor = ImageProcessor(self.config)
        
        # Test CPU fallback
        assert processor._resolve_device('cpu') == 'cpu'
        
        # Test auto resolution (should be CPU in test environment)
        auto_device = processor._resolve_device('auto')
        assert auto_device in ['cpu', 'cuda', 'mps']
    
    @patch('src.core.data_models.SliceStack.from_tiff_handler')
    def test_load_3d_volume(self, mock_from_tiff):
        """Test loading 3D volume."""
        processor = ImageProcessor(self.config)
        
        # Create mock volume
        mock_data = np.random.rand(5, 256, 256).astype(np.float32)
        mock_volume = SliceStack(mock_data)
        mock_from_tiff.return_value = mock_volume
        
        # Test loading
        volume = processor.load_3d_volume("test.tif")
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (5, 256, 256)
        mock_from_tiff.assert_called_once()
    
    def test_preprocess_volume(self):
        """Test volume preprocessing."""
        processor = ImageProcessor(self.config)
        
        # Create test volume with known range
        data = np.random.rand(2, 64, 64).astype(np.float32) * 255  # 0-255 range
        volume = SliceStack(data)
        
        # Preprocess
        preprocessed = processor.preprocess_volume(volume)
        
        # Check normalization
        preprocessed_data = preprocessed.to_numpy()
        assert preprocessed_data.min() >= 0.0
        assert preprocessed_data.max() <= 1.0
        
        # Check metadata
        assert 'normalization' in preprocessed.metadata
        assert 'min_value' in preprocessed.metadata['normalization']
        assert 'max_value' in preprocessed.metadata['normalization']
    
    def test_postprocess_volume(self):
        """Test volume postprocessing."""
        processor = ImageProcessor(self.config)
        
        # Create original volume
        original_data = np.random.rand(2, 64, 64).astype(np.float32) * 255
        original_volume = SliceStack(original_data)
        
        # Preprocess to get normalization params
        preprocessed = processor.preprocess_volume(original_volume)
        
        # Create enhanced volume (normalized)
        enhanced_data = np.random.rand(2, 64, 64).astype(np.float32)
        enhanced_volume = SliceStack(enhanced_data)
        
        # Postprocess
        postprocessed = processor.postprocess_volume(enhanced_volume, preprocessed)
        
        # Check that range is restored
        postprocessed_data = postprocessed.to_numpy()
        assert postprocessed_data.min() >= 0
        assert postprocessed_data.max() <= 255
    
    def test_extract_patches_for_inference(self):
        """Test patch extraction for inference."""
        processor = ImageProcessor(self.config)
        
        # Create test volume
        data = np.random.rand(2, 512, 512).astype(np.float32)
        volume = SliceStack(data)
        
        # Extract patches
        patches, patch_infos = processor.extract_patches_for_inference(volume)
        
        assert isinstance(patches, torch.Tensor)
        assert len(patch_infos) == len(patches)
        assert patches.shape[1:] == self.config.patch_size  # Check patch dimensions
    
    def test_reconstruct_volume_from_patches(self):
        """Test volume reconstruction from patches."""
        processor = ImageProcessor(self.config)
        
        # Create test volume and extract patches
        original_shape = (2, 512, 512)
        data = np.random.rand(*original_shape).astype(np.float32)
        volume = SliceStack(data)
        
        patches, patch_infos = processor.extract_patches_for_inference(volume)
        
        # Reconstruct
        reconstructed = processor.reconstruct_volume_from_patches(
            patches, patch_infos, original_shape
        )
        
        assert isinstance(reconstructed, SliceStack)
        assert reconstructed.shape == original_shape
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        processor = ImageProcessor(self.config)
        
        volume_shape = (10, 1024, 1024)
        memory_estimate = processor.estimate_memory_usage(volume_shape)
        
        assert isinstance(memory_estimate, dict)
        assert 'volume_mb' in memory_estimate
        assert 'total_estimated_mb' in memory_estimate
        assert memory_estimate['total_estimated_mb'] > 0
    
    def test_memory_constraints_check(self):
        """Test memory constraints checking."""
        processor = ImageProcessor(self.config)
        
        # Small volume should be feasible
        small_shape = (2, 256, 256)
        assert processor.check_memory_constraints(small_shape) is True
        
        # Very large volume should exceed limits
        large_shape = (1000, 4096, 4096)
        assert processor.check_memory_constraints(large_shape) is False
    
    def test_processing_chunks(self):
        """Test processing chunk calculation."""
        processor = ImageProcessor(self.config)
        
        volume_shape = (100, 512, 512)
        chunks = processor.get_processing_chunks(volume_shape, max_slices_per_chunk=10)
        
        assert len(chunks) == 10  # 100 slices / 10 per chunk
        assert chunks[0] == (0, 10)
        assert chunks[-1] == (90, 100)
    
    @patch('src.core.data_models.SliceStack.save_with_tiff_handler')
    def test_save_volume(self, mock_save):
        """Test volume saving."""
        processor = ImageProcessor(self.config)
        
        # Create test volume
        data = np.random.rand(2, 256, 256).astype(np.float32)
        volume = SliceStack(data)
        
        # Save volume
        output_path = str(Path(self.temp_dir) / "output.tif")
        processor.save_volume(volume, output_path)
        
        mock_save.assert_called_once_with(processor.tiff_handler, output_path)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        processor = ImageProcessor(self.config)
        
        # Create test volumes
        enhanced_data = np.random.rand(2, 64, 64).astype(np.float32)
        enhanced_volume = SliceStack(enhanced_data)
        
        original_data = np.random.rand(2, 64, 64).astype(np.float32)
        original_volume = SliceStack(original_data)
        
        # Get metrics without comparison
        metrics = processor.get_volume_quality_metrics(enhanced_volume)
        
        assert 'mean_intensity' in metrics
        assert 'std_intensity' in metrics
        assert 'dynamic_range' in metrics
        
        # Get metrics with comparison
        comparison_metrics = processor.get_volume_quality_metrics(
            enhanced_volume, original_volume
        )
        
        assert 'mse' in comparison_metrics
        assert 'psnr' in comparison_metrics
        assert 'mae' in comparison_metrics


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEnhancementProcessor:
    """Test EnhancementProcessor functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = InferenceConfig(
            patch_size=(256, 256),
            batch_size=4,
            device='cpu'
        )
    
    def test_enhancement_processor_init(self):
        """Test EnhancementProcessor initialization."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        assert processor.config == self.config
        assert processor.device == 'cpu'
        assert processor.is_model_loaded is False
    
    def test_device_resolution(self):
        """Test device resolution."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        assert processor._resolve_device('cpu') == 'cpu'
        auto_device = processor._resolve_device('auto')
        assert auto_device in ['cpu', 'cuda', 'mps']
    
    def test_load_model(self):
        """Test model loading."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        # Mock the model manager
        mock_model = nn.Conv2d(1, 1, 3, padding=1)
        processor.model_manager.load_pretrained_model = Mock(return_value=mock_model)
        processor.model_manager.validate_architecture = Mock(return_value=True)
        
        # Test loading
        processor.load_model("test_model.pth")
        
        assert processor.is_model_loaded is True
        assert processor.model is not None
        processor.model_manager.load_pretrained_model.assert_called_once_with("test_model.pth")
    
    def test_enhance_patches(self):
        """Test patch enhancement."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        # Create simple model
        model = nn.Conv2d(1, 1, 3, padding=1)
        processor.model = model.to(processor.device)
        processor.model.eval()
        processor.is_model_loaded = True
        
        # Create test patches
        patches = torch.randn(8, 64, 64)
        
        # Enhance patches
        enhanced = processor.enhance_patches(patches, batch_size=4)
        
        assert enhanced.shape == patches.shape
        assert isinstance(enhanced, torch.Tensor)
    
    def test_model_compatibility_validation(self):
        """Test model compatibility validation."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        # Test without loaded model
        assert processor.validate_model_compatibility((256, 256)) is False
        
        # Load simple model
        model = nn.Conv2d(1, 1, 3, padding=1)
        processor.model = model.to(processor.device)
        processor.model.eval()
        processor.is_model_loaded = True
        
        # Test compatibility
        assert processor.validate_model_compatibility((256, 256)) is True
    
    def test_model_info(self):
        """Test getting model information."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        # Test without loaded model
        info = processor.get_model_info()
        assert info['loaded'] is False
        
        # Load model
        model = nn.Conv2d(1, 1, 3, padding=1)
        processor.model = model.to(processor.device)
        processor.is_model_loaded = True
        
        # Test with loaded model
        info = processor.get_model_info()
        assert info['loaded'] is True
        assert 'total_parameters' in info
        assert 'model_size_mb' in info
        assert info['device'] == 'cpu'
    
    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        estimate = processor.estimate_processing_time(100, batch_size=4)
        
        assert 'num_patches' in estimate
        assert 'estimated_seconds' in estimate
        assert 'patches_per_second' in estimate
        assert estimate['num_patches'] == 100
        assert estimate['batch_size'] == 4
    
    def test_cleanup(self):
        """Test resource cleanup."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        processor = EnhancementProcessor(self.config)
        
        # Load model
        model = nn.Conv2d(1, 1, 3, padding=1)
        processor.model = model
        processor.is_model_loaded = True
        
        # Cleanup
        processor.cleanup()
        
        assert processor.model is None
        assert processor.is_model_loaded is False
    
    def test_context_manager(self):
        """Test context manager functionality."""
        from src.inference.enhancement_processor import EnhancementProcessor
        
        with EnhancementProcessor(self.config) as processor:
            assert processor is not None
        
        # Should be cleaned up after context exit
        assert processor.model is None
        assert processor.is_model_loaded is False


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_import_error_without_torch():
    """Test that appropriate errors are raised when PyTorch is not available."""
    from src.inference.enhancement_processor import EnhancementProcessor
    from src.core.config import InferenceConfig
    
    config = InferenceConfig()
    
    with pytest.raises(ImportError, match="PyTorch is required"):
        EnhancementProcessor(config)