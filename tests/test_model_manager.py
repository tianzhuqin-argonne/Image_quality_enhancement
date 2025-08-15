"""
Unit tests for UNetModelManager class.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.models.model_manager import UNetModelManager
from src.models.unet import UNet, create_unet_model


class TestUNetModelManager:
    """Test cases for UNetModelManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = UNetModelManager(cache_dir=self.temp_dir, device='cpu')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_manager_initialization(self):
        """Test model manager initialization."""
        assert self.manager.cache_dir == Path(self.temp_dir)
        assert self.manager.device == torch.device('cpu')
    
    def test_list_recommended_models(self):
        """Test listing recommended models."""
        models = self.manager.list_recommended_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Check that all models have required metadata
        for model_name, info in models.items():
            required_keys = ['description', 'license', 'commercial_use', 'architecture']
            for key in required_keys:
                assert key in info, f"Missing key {key} in model {model_name}"
            
            # Check that commercial use is allowed
            assert info['commercial_use'] is True, f"Model {model_name} not commercially usable"
    
    def test_create_default_model(self):
        """Test creating default model."""
        model = self.manager.create_default_model()
        
        assert isinstance(model, UNet)
        assert model.n_channels == 1
        assert model.n_classes == 1
        assert model.base_channels == 64
        
        # Test that model is on correct device
        assert next(model.parameters()).device == self.manager.device
    
    def test_validate_architecture(self):
        """Test architecture validation."""
        # Test with valid U-Net
        valid_model = create_unet_model()
        assert self.manager.validate_architecture(valid_model) is True
        
        # Test with invalid model (not U-Net)
        invalid_model = torch.nn.Linear(10, 5)
        assert self.manager.validate_architecture(invalid_model) is False
    
    def test_predict_batch(self):
        """Test batch prediction."""
        model = create_unet_model()
        patches = torch.randn(4, 128, 128)  # 4 patches without channel dimension
        
        enhanced = self.manager.predict_batch(model, patches)
        
        assert enhanced.shape == (4, 128, 128)
        assert torch.isfinite(enhanced).all()
    
    def test_predict_batch_with_channels(self):
        """Test batch prediction with channel dimension."""
        model = create_unet_model()
        patches = torch.randn(4, 1, 128, 128)  # 4 patches with channel dimension
        
        enhanced = self.manager.predict_batch(model, patches)
        
        assert enhanced.shape == (4, 128, 128)  # Channel dimension removed
        assert torch.isfinite(enhanced).all()
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Create and train a model slightly
        original_model = create_unet_model()
        
        # Modify some weights to make it unique
        with torch.no_grad():
            for param in original_model.parameters():
                param.add_(torch.randn_like(param) * 0.01)
                break  # Just modify first parameter
        
        # Save model
        save_path = Path(self.temp_dir) / "test_model.pth"
        self.manager.save_model(original_model, str(save_path))
        
        assert save_path.exists()
        
        # Load model
        loaded_model = self.manager.load_model(str(save_path))
        
        assert loaded_model is not None
        assert isinstance(loaded_model, UNet)
        
        # Test that loaded model produces same output
        test_input = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            original_output = original_model(test_input)
            loaded_output = loaded_model(test_input)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
    
    def test_load_nonexistent_model(self):
        """Test loading non-existent model."""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.pth"
        loaded_model = self.manager.load_model(str(nonexistent_path))
        
        assert loaded_model is None
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        model = create_unet_model()
        summary = self.manager.get_model_summary(model)
        
        required_keys = [
            'architecture', 'input_channels', 'output_classes', 'base_channels',
            'total_parameters', 'trainable_parameters', 'device', 'memory_usage_mb'
        ]
        
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"
        
        assert summary['device'] == 'cpu'
        assert summary['memory_usage_mb'] > 0
    
    @patch('src.models.model_manager.HF_AVAILABLE', False)
    def test_manager_without_huggingface(self):
        """Test model manager without Hugging Face."""
        manager = UNetModelManager(cache_dir=self.temp_dir, device='cpu')
        
        # Should still be able to create default models
        model = manager.create_default_model()
        assert isinstance(model, UNet)
        
        # Download should fail gracefully
        result = manager.download_pretrained_model("test/model")
        assert result is None
    
    def test_fine_tune_model_setup(self):
        """Test fine-tuning setup (without actual training)."""
        model = create_unet_model()
        
        # Create dummy data loader
        class DummyDataLoader:
            def __init__(self):
                self.data = [(torch.randn(2, 1, 128, 128), torch.randn(2, 1, 128, 128)) for _ in range(3)]
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        train_loader = DummyDataLoader()
        
        # Test fine-tuning for 1 epoch
        history = self.manager.fine_tune_model(
            model=model,
            train_loader=train_loader,
            epochs=1,
            learning_rate=1e-4
        )
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 1
        assert isinstance(history['train_loss'][0], float)
        assert history['train_loss'][0] >= 0
    
    def test_fine_tune_with_validation(self):
        """Test fine-tuning with validation data."""
        model = create_unet_model()
        
        # Create dummy data loaders
        class DummyDataLoader:
            def __init__(self, size=2):
                self.data = [(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64)) for _ in range(size)]
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return len(self.data)
        
        train_loader = DummyDataLoader(3)
        val_loader = DummyDataLoader(2)
        
        # Test fine-tuning with validation
        history = self.manager.fine_tune_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            learning_rate=1e-4
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 1
        assert len(history['val_loss']) == 1
        assert isinstance(history['val_loss'][0], float)
    
    def test_device_handling(self):
        """Test device handling in model manager."""
        # Test CPU manager
        cpu_manager = UNetModelManager(device='cpu')
        cpu_model = cpu_manager.create_default_model()
        
        assert next(cpu_model.parameters()).device.type == 'cpu'
        
        # Test CUDA manager if available
        if torch.cuda.is_available():
            cuda_manager = UNetModelManager(device='cuda')
            cuda_model = cuda_manager.create_default_model()
            
            assert next(cuda_model.parameters()).device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")
    
    def test_model_info_consistency(self):
        """Test consistency between model info and actual model."""
        model = create_unet_model(
            input_channels=3,
            output_channels=2,
            base_channels=32
        )
        
        info = model.get_model_info()
        summary = self.manager.get_model_summary(model)
        
        # Check consistency
        assert info['input_channels'] == 3
        assert info['output_classes'] == 2
        assert info['base_channels'] == 32
        
        # Summary should include all info fields
        for key, value in info.items():
            if key in summary:
                assert summary[key] == value
    
    @patch('src.models.model_manager.hf_hub_download')
    @patch('src.models.model_manager.model_info')
    @patch('src.models.model_manager.list_repo_files')
    def test_download_pretrained_model_mock(self, mock_list_files, mock_model_info, mock_download):
        """Test downloading pre-trained model with mocks."""
        # Setup mocks
        mock_info = MagicMock()
        mock_info.license = "MIT"
        mock_model_info.return_value = mock_info
        
        mock_list_files.return_value = ['config.json', 'pytorch_model.bin']
        mock_download.return_value = str(Path(self.temp_dir) / 'downloaded_file.bin')
        
        # Test download
        result = self.manager.download_pretrained_model("test/model")
        
        assert result is not None
        assert isinstance(result, Path)
        
        # Verify mocks were called
        mock_model_info.assert_called_once_with("test/model")
        mock_list_files.assert_called_once_with("test/model")
        assert mock_download.call_count >= 1
    
    def test_load_pretrained_model_fallback(self):
        """Test loading pre-trained model with fallback to default."""
        # Test with non-existent path (should create default model)
        model = self.manager.load_pretrained_model("nonexistent/path")
        
        # Should return None since path doesn't exist and download would fail
        assert model is None
    
    def test_cache_directory_creation(self):
        """Test cache directory creation."""
        new_cache_dir = Path(self.temp_dir) / "new_cache"
        manager = UNetModelManager(cache_dir=str(new_cache_dir))
        
        assert new_cache_dir.exists()
        assert manager.cache_dir == new_cache_dir


class TestModelManagerIntegration:
    """Integration tests for model manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = UNetModelManager(cache_dir=self.temp_dir, device='cpu')
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow(self):
        """Test complete model workflow."""
        # 1. Create model
        model = self.manager.create_default_model()
        assert isinstance(model, UNet)
        
        # 2. Validate architecture
        is_valid = self.manager.validate_architecture(model)
        assert is_valid is True
        
        # 3. Get model summary
        summary = self.manager.get_model_summary(model)
        assert 'total_parameters' in summary
        
        # 4. Test prediction
        test_patches = torch.randn(2, 64, 64)
        enhanced = self.manager.predict_batch(model, test_patches)
        assert enhanced.shape == (2, 64, 64)
        
        # 5. Save model
        save_path = Path(self.temp_dir) / "workflow_model.pth"
        self.manager.save_model(model, str(save_path))
        assert save_path.exists()
        
        # 6. Load model
        loaded_model = self.manager.load_model(str(save_path))
        assert loaded_model is not None
        
        # 7. Verify loaded model works
        enhanced_loaded = self.manager.predict_batch(loaded_model, test_patches)
        assert torch.allclose(enhanced, enhanced_loaded, atol=1e-6)
    
    def test_model_compatibility(self):
        """Test model compatibility across different configurations."""
        configs = [
            {'input_channels': 1, 'output_channels': 1, 'base_channels': 32},
            {'input_channels': 1, 'output_channels': 1, 'base_channels': 64},
            {'input_channels': 1, 'output_channels': 1, 'base_channels': 128}
        ]
        
        for config in configs:
            model = create_unet_model(**config)
            
            # All models should be valid
            assert self.manager.validate_architecture(model) is True
            
            # All models should work for prediction
            test_input = torch.randn(1, 64, 64)
            enhanced = self.manager.predict_batch(model, test_input)
            assert enhanced.shape == (1, 64, 64)
    
    def test_memory_management(self):
        """Test memory management with large models."""
        # Create model with larger base channels
        large_model = create_unet_model(base_channels=128)
        
        # Get memory usage
        summary = self.manager.get_model_summary(large_model)
        memory_mb = summary['memory_usage_mb']
        
        assert memory_mb > 0
        assert memory_mb < 2000  # Should be reasonable size
        
        # Test prediction doesn't cause memory issues
        large_batch = torch.randn(8, 128, 128)
        
        with torch.no_grad():
            enhanced = self.manager.predict_batch(large_model, large_batch)
            assert enhanced.shape == (8, 128, 128)


if __name__ == "__main__":
    pytest.main([__file__])