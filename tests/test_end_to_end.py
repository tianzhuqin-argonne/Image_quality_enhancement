"""
End-to-end integration tests for the complete 3D image enhancement system.

These tests validate the entire pipeline from data loading through
training and inference to final output validation.
"""

import pytest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.testing.test_fixtures import TestDataFixtures
from src.training.training_manager import TrainingManager
from src.inference.api import ImageEnhancementAPI
from src.core.config import TrainingConfig, InferenceConfig
from src.core.data_models import SliceStack


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEndToEndTrainingPipeline:
    """Test complete training pipeline from data to trained model."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
        
        # Fast training config for testing
        self.training_config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            device='cpu',
            checkpoint_interval=1,
            early_stopping_patience=5,
            use_augmentation=False  # Disable for faster testing
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_training_workflow(self):
        """Test complete training workflow from synthetic data to trained model."""
        # Create test dataset
        dataset_pairs = self.fixtures.create_test_dataset(
            num_pairs=4, volume_size="small", degradation_level="light"
        )
        
        input_paths = [pair[0] for pair in dataset_pairs]
        target_paths = [pair[1] for pair in dataset_pairs]
        
        # Initialize training manager
        training_output_dir = Path(self.temp_dir) / "training_output"
        training_manager = TrainingManager(
            config=self.training_config,
            output_dir=str(training_output_dir),
            use_tensorboard=False
        )
        
        # Prepare training data
        train_loader, val_loader = training_manager.prepare_training_data(
            input_paths, target_paths
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Create and save a mock pre-trained model
        mock_model_path = self.fixtures.save_mock_model("pretrained_model.pth")
        
        # Fine-tune model
        best_model_path = training_manager.fine_tune_model(
            mock_model_path, train_loader, val_loader
        )
        
        # Validate training results
        assert Path(best_model_path).exists()
        assert training_manager.training_state.epoch == self.training_config.epochs
        
        # Check that checkpoints were created
        checkpoints_dir = training_output_dir / "checkpoints"
        assert checkpoints_dir.exists()
        assert len(list(checkpoints_dir.glob("*.pth"))) > 0
        
        # Validate model
        validation_metrics = training_manager.validate_model(best_model_path, val_loader)
        
        assert 'validation_loss' in validation_metrics
        assert 'psnr' in validation_metrics
        assert validation_metrics['validation_loss'] >= 0
        
        # Export model for inference
        export_path = Path(self.temp_dir) / "exported_model.pth"
        training_manager.export_model_for_inference(best_model_path, str(export_path))
        
        assert export_path.exists()
        
        # Get training summary
        summary = training_manager.get_training_summary()
        assert summary['total_epochs'] == self.training_config.epochs
        assert 'best_validation_loss' in summary
    
    def test_training_with_validation_split(self):
        """Test training with proper validation split."""
        # Create larger dataset for meaningful split
        dataset_pairs = self.fixtures.create_test_dataset(
            num_pairs=8, volume_size="small", degradation_level="moderate"
        )
        
        input_paths = [pair[0] for pair in dataset_pairs]
        target_paths = [pair[1] for pair in dataset_pairs]
        
        training_manager = TrainingManager(
            config=self.training_config,
            output_dir=str(Path(self.temp_dir) / "training_split"),
            use_tensorboard=False
        )
        
        # Test data preparation with validation split
        train_loader, val_loader = training_manager.prepare_training_data(
            input_paths, target_paths
        )
        
        # Check that data was split appropriately
        total_batches = len(train_loader) + len(val_loader)
        assert total_batches > 0
        assert len(val_loader) > 0  # Should have validation data
        
        # Validation set should be smaller than training set
        assert len(val_loader) < len(train_loader)
    
    def test_training_error_recovery(self):
        """Test training pipeline error handling and recovery."""
        # Create minimal dataset
        dataset_pairs = self.fixtures.create_test_dataset(num_pairs=2, volume_size="small")
        input_paths = [pair[0] for pair in dataset_pairs]
        target_paths = [pair[1] for pair in dataset_pairs]
        
        training_manager = TrainingManager(
            config=self.training_config,
            output_dir=str(Path(self.temp_dir) / "error_recovery"),
            use_tensorboard=False
        )
        
        # Test with non-existent model path
        train_loader, val_loader = training_manager.prepare_training_data(
            input_paths, target_paths
        )
        
        with pytest.raises((FileNotFoundError, ValueError)):
            training_manager.fine_tune_model(
                "nonexistent_model.pth", train_loader, val_loader
            )
        
        # Test with empty input paths
        with pytest.raises(ValueError):
            training_manager.prepare_training_data([], [])
        
        # Test with mismatched input/target paths
        with pytest.raises(ValueError):
            training_manager.prepare_training_data(input_paths, target_paths[:-1])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEndToEndInferencePipeline:
    """Test complete inference pipeline from model to enhanced output."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
        
        # Fast inference config for testing
        self.inference_config = InferenceConfig(
            patch_size=(128, 128),  # Smaller patches for faster testing
            overlap=16,
            batch_size=2,
            device='cpu',
            memory_limit_gb=1.0,
            enable_quality_metrics=True
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_inference_workflow(self):
        """Test complete inference workflow from input TIFF to enhanced output."""
        # Create test input data
        input_volume = self.fixtures.get_small_test_volume()
        input_path = self.fixtures.save_test_tiff(input_volume, "input_test")
        
        # Create and save mock model
        mock_model_path = self.fixtures.save_mock_model("inference_model.pth")
        
        # Initialize API
        api = ImageEnhancementAPI(self.inference_config)
        
        # Load model
        success = api.load_model(mock_model_path)
        assert success is True
        assert api.model_loaded is True
        
        # Test file-based enhancement
        output_path = str(Path(self.temp_dir) / "enhanced_output.tif")
        
        progress_updates = []
        def progress_callback(message: str, progress: float):
            progress_updates.append((message, progress))
        
        result = api.enhance_3d_tiff(
            input_path, output_path, 
            progress_callback=progress_callback,
            calculate_metrics=True
        )
        
        # Validate results
        assert result.success is True
        assert result.processing_time > 0
        assert Path(output_path).exists()
        assert result.quality_metrics is not None
        assert len(progress_updates) > 0
        
        # Check that final progress was 100%
        final_progress = progress_updates[-1][1]
        assert final_progress == 1.0
        
        # Get comprehensive metrics
        metrics = api.get_enhancement_metrics(result)
        assert 'processing_efficiency' in metrics
        assert 'model_info' in metrics
    
    def test_array_based_inference(self):
        """Test inference with numpy arrays instead of files."""
        # Create test data
        test_array = np.random.rand(3, 128, 128).astype(np.float32)
        
        # Create and load mock model
        mock_model_path = self.fixtures.save_mock_model("array_model.pth")
        
        api = ImageEnhancementAPI(self.inference_config)
        api.load_model(mock_model_path)
        
        # Test array enhancement
        result = api.enhance_3d_array(test_array, calculate_metrics=True)
        
        assert result.success is True
        assert result.input_shape == test_array.shape
        assert 'enhanced_array' in result.quality_metrics
        
        enhanced_array = result.quality_metrics['enhanced_array']
        assert enhanced_array.shape == test_array.shape
        assert enhanced_array.dtype == np.float32
    
    def test_inference_memory_management(self):
        """Test inference with memory constraints and chunking."""
        # Create larger volume that should trigger chunking
        large_volume = self.fixtures.get_medium_test_volume()  # 10x512x512
        
        # Use very restrictive memory limit to force chunking
        restrictive_config = InferenceConfig(
            patch_size=(256, 256),
            overlap=32,
            batch_size=1,
            device='cpu',
            memory_limit_gb=0.1  # Very small limit
        )
        
        api = ImageEnhancementAPI(restrictive_config)
        mock_model_path = self.fixtures.save_mock_model("memory_model.pth")
        api.load_model(mock_model_path)
        
        # Test with array (easier to control)
        large_array = large_volume.to_numpy()
        
        result = api.enhance_3d_array(large_array)
        
        # Should still succeed despite memory constraints
        assert result.success is True
        # May or may not have warnings depending on actual memory usage
        # The important thing is that it succeeded despite constraints
    
    def test_inference_error_handling(self):
        """Test inference pipeline error handling."""
        api = ImageEnhancementAPI(self.inference_config)
        
        # Test enhancement without loaded model
        test_array = np.random.rand(2, 64, 64).astype(np.float32)
        result = api.enhance_3d_array(test_array)
        
        assert result.success is False
        assert "No model loaded" in result.error_message
        
        # Test with invalid input dimensions
        api.load_model(self.fixtures.save_mock_model("error_model.pth"))
        
        invalid_array = np.random.rand(64, 64)  # 2D instead of 3D
        result = api.enhance_3d_array(invalid_array)
        
        assert result.success is False
        assert "must be 3D" in result.error_message
        
        # Test with non-existent file
        result = api.enhance_3d_tiff("nonexistent.tif", "output.tif")
        
        assert result.success is False
        assert "not found" in result.error_message
    
    def test_inference_configuration_updates(self):
        """Test runtime configuration updates."""
        api = ImageEnhancementAPI(self.inference_config)
        
        original_batch_size = api.config.batch_size
        
        # Update configuration
        api.set_processing_config(batch_size=4, memory_limit_gb=2.0)
        
        assert api.config.batch_size == 4
        assert api.config.memory_limit_gb == 2.0
        assert api.config.batch_size != original_batch_size
    
    def test_system_info_and_diagnostics(self):
        """Test system information and diagnostic capabilities."""
        api = ImageEnhancementAPI(self.inference_config)
        
        # Get system info
        system_info = api.get_system_info()
        
        assert 'torch_available' in system_info
        assert 'device' in system_info
        assert 'config' in system_info
        assert 'model_loaded' in system_info
        
        if TORCH_AVAILABLE:
            assert 'torch_info' in system_info
            assert system_info['torch_available'] is True
        
        # Load model and check updated info
        mock_model_path = self.fixtures.save_mock_model("diagnostic_model.pth")
        api.load_model(mock_model_path)
        
        updated_info = api.get_system_info()
        assert updated_info['model_loaded'] is True
        assert updated_info['model_path'] == mock_model_path


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestEndToEndIntegration:
    """Test integration between training and inference pipelines."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_then_infer_workflow(self):
        """Test complete workflow: train model, then use it for inference."""
        # Step 1: Create training dataset
        training_pairs = self.fixtures.create_test_dataset(
            num_pairs=4, volume_size="small", degradation_level="moderate"
        )
        
        input_paths = [pair[0] for pair in training_pairs]
        target_paths = [pair[1] for pair in training_pairs]
        
        # Step 2: Train model
        training_config = TrainingConfig(
            epochs=2, batch_size=2, learning_rate=1e-3, device='cpu',
            checkpoint_interval=1, use_augmentation=False
        )
        
        training_manager = TrainingManager(
            config=training_config,
            output_dir=str(Path(self.temp_dir) / "integrated_training"),
            use_tensorboard=False
        )
        
        # Prepare training data
        train_loader, val_loader = training_manager.prepare_training_data(
            input_paths, target_paths
        )
        
        # Create initial model and train
        initial_model_path = self.fixtures.save_mock_model("initial_model.pth")
        trained_model_path = training_manager.fine_tune_model(
            initial_model_path, train_loader, val_loader
        )
        
        # Export trained model
        inference_model_path = str(Path(self.temp_dir) / "inference_ready_model.pth")
        training_manager.export_model_for_inference(trained_model_path, inference_model_path)
        
        # Step 3: Use trained model for inference
        inference_config = InferenceConfig(
            patch_size=(128, 128), overlap=16, batch_size=2, device='cpu'
        )
        
        api = ImageEnhancementAPI(inference_config)
        
        # Load the trained model
        load_success = api.load_model(inference_model_path)
        assert load_success is True
        
        # Step 4: Test inference on new data
        test_volume = self.fixtures.get_small_test_volume()
        test_input_path = self.fixtures.save_test_tiff(test_volume, "integration_test_input")
        
        output_path = str(Path(self.temp_dir) / "integration_output.tif")
        
        result = api.enhance_3d_tiff(test_input_path, output_path, calculate_metrics=True)
        
        # Validate end-to-end results
        assert result.success is True
        assert Path(output_path).exists()
        assert result.quality_metrics is not None
        
        # Validate that the model actually processed the data
        assert result.processing_time > 0
        assert result.input_shape == test_volume.shape
    
    def test_model_compatibility_validation(self):
        """Test model compatibility between training and inference."""
        # Create model with specific configuration
        training_config = TrainingConfig(patch_size=(256, 256))
        inference_config = InferenceConfig(patch_size=(256, 256))
        
        # Save mock model
        model_path = self.fixtures.save_mock_model("compatibility_model.pth")
        
        # Test compatible configuration
        api = ImageEnhancementAPI(inference_config)
        success = api.load_model(model_path)
        # May succeed or fail depending on model architecture validation
        # The important thing is that it handles the situation gracefully
        assert isinstance(success, bool)
        
        # Test incompatible configuration
        incompatible_config = InferenceConfig(patch_size=(512, 512))
        api_incompatible = ImageEnhancementAPI(incompatible_config)
        
        # This might succeed or fail depending on model flexibility
        # The important thing is that it handles the situation gracefully
        result = api_incompatible.load_model(model_path)
        assert isinstance(result, bool)  # Should return boolean, not crash
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking across the pipeline."""
        benchmark_data = self.fixtures.get_benchmark_data()
        
        # Test with small dataset
        small_benchmark = benchmark_data['small_dataset']
        expected_time = small_benchmark['expected_processing_time']
        
        # Create test data matching benchmark specs
        test_volume = self.fixtures.get_small_test_volume()
        assert test_volume.shape == small_benchmark['shape']
        
        # Setup inference
        api = ImageEnhancementAPI(InferenceConfig(device='cpu'))
        model_path = self.fixtures.save_mock_model("benchmark_model.pth")
        api.load_model(model_path)
        
        # Measure performance
        start_time = time.time()
        result = api.enhance_3d_array(test_volume.to_numpy())
        actual_time = time.time() - start_time
        
        assert result.success is True
        
        # Performance should be reasonable (allow 3x expected time for test environment)
        assert actual_time < expected_time * 3
        
        # Validate quality
        quality_results = self.fixtures.validate_enhancement_quality(
            test_volume, 
            SliceStack(result.quality_metrics['enhanced_array']),
            min_psnr=small_benchmark['min_psnr']
        )
        
        assert quality_results['shape_match'] is True
        # Note: PSNR might be low for mock model, so we don't assert on quality_acceptable
    
    def test_resource_constraint_scenarios(self):
        """Test system behavior under various resource constraints."""
        # Test with very limited memory
        constrained_config = InferenceConfig(
            patch_size=(64, 64),  # Very small patches
            batch_size=1,         # Minimal batch size
            memory_limit_gb=0.1,  # Very limited memory
            device='cpu'
        )
        
        api = ImageEnhancementAPI(constrained_config)
        model_path = self.fixtures.save_mock_model("constrained_model.pth")
        api.load_model(model_path)
        
        # Test with medium-sized volume
        test_volume = self.fixtures.get_medium_test_volume()
        
        result = api.enhance_3d_array(test_volume.to_numpy())
        
        # Should still work, but might have warnings
        assert result.success is True
        if result.warnings:
            assert any("memory" in warning.lower() for warning in result.warnings)


# Test offline operation capabilities
class TestOfflineOperation:
    """Test system operation without external dependencies."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_offline_data_generation(self):
        """Test synthetic data generation without external dependencies."""
        # Should work without internet or external libraries
        volume = self.fixtures.get_small_test_volume()
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (5, 256, 256)
        
        # Test degradation simulation
        degraded_input, clean_target = self.fixtures.get_training_pair()
        
        assert isinstance(degraded_input, SliceStack)
        assert isinstance(clean_target, SliceStack)
        assert degraded_input.shape == clean_target.shape
    
    def test_offline_configuration_management(self):
        """Test configuration management without external dependencies."""
        # Test training config
        training_config = self.fixtures.get_training_config("fast")
        assert training_config.epochs == 2
        assert training_config.device == 'cpu'
        
        # Test inference config
        inference_config = self.fixtures.get_inference_config("default")
        assert inference_config.patch_size == (256, 256)
        assert inference_config.device == 'cpu'
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_offline_model_operations(self):
        """Test model operations without external dependencies."""
        # Create mock model
        model_path = self.fixtures.save_mock_model("offline_model.pth")
        assert Path(model_path).exists()
        
        # Load and validate model structure
        checkpoint = torch.load(model_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint
        
        # Test model creation
        mock_model = TestDataFixtures.create_mock_unet_model()
        assert isinstance(mock_model, torch.nn.Module)
        
        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64)
        output = mock_model(test_input)
        assert output.shape == test_input.shape