"""
Integration tests for complete training pipeline.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.training.training_manager import TrainingManager
from src.core.config import TrainingConfig
from src.core.data_models import SliceStack, Patch2D, TrainingPair2D


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            device='cpu',
            checkpoint_interval=1,
            early_stopping_patience=5,
            use_augmentation=False  # Disable for simpler testing
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_tiff_files(self):
        """Create mock TIFF files for testing."""
        # Create mock input and target TIFF files
        input_path = Path(self.temp_dir) / "input.tif"
        target_path = Path(self.temp_dir) / "target.tif"
        
        # Create dummy data
        input_data = np.random.rand(2, 256, 256).astype(np.float32)
        target_data = np.random.rand(2, 256, 256).astype(np.float32)
        
        # Save as numpy files (simulating TIFF files for testing)
        np.save(input_path.with_suffix('.npy'), input_data)
        np.save(target_path.with_suffix('.npy'), target_data)
        
        return str(input_path.with_suffix('.npy')), str(target_path.with_suffix('.npy'))
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        class SimpleUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.conv2(x)
                return x
        
        return SimpleUNet()
    
    @patch('src.training.training_manager.TrainingDataLoader')
    @patch('src.training.training_manager.UNetModelManager')
    def test_complete_training_workflow(self, mock_model_manager_class, mock_data_loader_class):
        """Test complete training workflow from data preparation to model export."""
        # Setup mocks
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        
        # Create training pairs
        training_pairs = self._create_mock_training_pairs(8)
        mock_data_loader.load_training_data.return_value = training_pairs
        mock_data_loader.get_data_statistics.return_value = {'num_pairs': 8}
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.randn(6, 1, 64, 64),  # 6 training samples
            torch.randn(6, 1, 64, 64)
        )
        val_dataset = TensorDataset(
            torch.randn(2, 1, 64, 64),  # 2 validation samples
            torch.randn(2, 1, 64, 64)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=2)
        mock_data_loader.create_data_loaders.return_value = (train_loader, val_loader)
        
        # Setup model
        model = self.create_simple_model()
        mock_model_manager.load_pretrained_model.return_value = model
        mock_model_manager.validate_architecture.return_value = True
        
        # Create training manager
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Test data preparation
        train_loader, val_loader = manager.prepare_training_data(
            input_paths=['input.tif'],
            target_paths=['target.tif']
        )
        
        # Check that statistics file was created
        stats_file = Path(self.temp_dir) / "data_statistics.json"
        assert stats_file.exists()
        
        # Test model fine-tuning
        pretrained_model_path = Path(self.temp_dir) / "pretrained_model.pth"
        torch.save({'model_state_dict': model.state_dict()}, pretrained_model_path)
        
        best_model_path = manager.fine_tune_model(
            str(pretrained_model_path),
            train_loader,
            val_loader
        )
        
        # Check that training completed and model was saved
        assert Path(best_model_path).exists()
        
        # Check that checkpoints directory exists and has files
        checkpoints_dir = Path(self.temp_dir) / "checkpoints"
        assert checkpoints_dir.exists()
        assert len(list(checkpoints_dir.glob("*.pth"))) > 0
        
        # Check that training state was saved
        training_state_file = Path(self.temp_dir) / "training_state.json"
        assert training_state_file.exists()
        
        # Test model validation
        validation_metrics = manager.validate_model(best_model_path, val_loader)
        
        assert isinstance(validation_metrics, dict)
        assert 'validation_loss' in validation_metrics
        assert 'mse' in validation_metrics
        assert 'mae' in validation_metrics
        assert 'psnr' in validation_metrics
        
        # Test model export
        export_path = Path(self.temp_dir) / "exported_model.pth"
        manager.export_model_for_inference(best_model_path, str(export_path))
        
        assert export_path.exists()
        
        # Verify exported model structure
        exported_model = torch.load(export_path)
        assert 'model_state_dict' in exported_model
        assert 'model_config' in exported_model
        assert 'training_info' in exported_model
        
        # Test training summary
        summary = manager.get_training_summary()
        assert 'total_epochs' in summary
        assert 'best_epoch' in summary
        assert 'best_validation_loss' in summary
    
    def test_training_error_handling_corrupted_data(self):
        """Test error handling when training data is corrupted."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Test with empty training pairs
        with patch.object(manager.data_loader, 'load_training_data') as mock_load:
            mock_load.return_value = []
            
            with pytest.raises(ValueError, match="No training pairs loaded"):
                manager.prepare_training_data(['input.tif'], ['target.tif'])
    
    def test_training_error_handling_invalid_model(self):
        """Test error handling when model architecture is invalid."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Mock data loaders
        train_loader = DataLoader(TensorDataset(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)))
        val_loader = DataLoader(TensorDataset(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)))
        
        # Mock model manager to return invalid architecture
        with patch.object(manager.model_manager, 'load_pretrained_model') as mock_load:
            with patch.object(manager.model_manager, 'validate_architecture') as mock_validate:
                mock_load.return_value = nn.Linear(1, 1)
                mock_validate.return_value = False
                
                with pytest.raises(ValueError, match="Model architecture validation failed"):
                    manager.fine_tune_model("dummy_path.pth", train_loader, val_loader)
    
    def test_training_error_handling_checkpoint_loading(self):
        """Test error handling when loading corrupted checkpoints."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Test loading non-existent checkpoint
        success = manager.load_checkpoint("non_existent.pth")
        assert success is False
        
        # Test loading corrupted checkpoint
        corrupted_path = Path(self.temp_dir) / "corrupted.pth"
        with open(corrupted_path, 'w') as f:
            f.write("corrupted data")
        
        success = manager.load_checkpoint(str(corrupted_path))
        assert success is False
    
    def test_training_memory_management(self):
        """Test training with memory constraints."""
        # Test with very small batch size to simulate memory constraints
        small_config = TrainingConfig(
            epochs=1,
            batch_size=1,  # Very small batch size
            learning_rate=1e-3,
            device='cpu',
            gradient_clip_value=0.5  # Enable gradient clipping
        )
        
        manager = TrainingManager(
            config=small_config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup simple model and data
        manager.model = self.create_simple_model()
        manager._setup_training_components()
        
        # Create small dataset
        train_loader = DataLoader(
            TensorDataset(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)),
            batch_size=1
        )
        
        # Test training epoch with gradient clipping
        avg_loss, train_time = manager._train_epoch(train_loader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert train_time >= 0
    
    def test_early_stopping_mechanism(self):
        """Test early stopping mechanism."""
        # Configure for early stopping
        early_stop_config = TrainingConfig(
            epochs=10,
            batch_size=2,
            learning_rate=1e-3,
            device='cpu',
            early_stopping_patience=2,  # Very low patience for testing
            checkpoint_interval=1
        )
        
        manager = TrainingManager(
            config=early_stop_config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup model and data
        model = self.create_simple_model()
        
        with patch.object(manager.model_manager, 'load_pretrained_model') as mock_load:
            with patch.object(manager.model_manager, 'validate_architecture') as mock_validate:
                mock_load.return_value = model
                mock_validate.return_value = True
                
                # Create data loaders
                train_loader = DataLoader(
                    TensorDataset(torch.randn(4, 1, 64, 64), torch.randn(4, 1, 64, 64)),
                    batch_size=2
                )
                val_loader = DataLoader(
                    TensorDataset(torch.randn(2, 1, 64, 64), torch.randn(2, 1, 64, 64)),
                    batch_size=2
                )
                
                # Mock validation to return increasing loss (no improvement)
                with patch.object(manager, '_validate_epoch') as mock_validate_epoch:
                    mock_validate_epoch.side_effect = [
                        (1.0, 1.0),  # First epoch
                        (1.1, 1.0),  # Second epoch (worse)
                        (1.2, 1.0),  # Third epoch (worse)
                        (1.3, 1.0),  # Fourth epoch (should trigger early stopping)
                    ]
                    
                    with patch.object(manager, '_train_epoch') as mock_train_epoch:
                        mock_train_epoch.return_value = (0.5, 1.0)
                        
                        # Run training (should stop early)
                        best_model_path = manager.fine_tune_model(
                            "dummy_path.pth", train_loader, val_loader
                        )
                        
                        # Should have stopped before 10 epochs due to early stopping
                        assert manager.training_state.epoch < 10
                        assert manager.training_state.patience_counter >= early_stop_config.early_stopping_patience
    
    def _create_mock_training_pairs(self, count: int) -> list:
        """Create mock training pairs for testing."""
        pairs = []
        for i in range(count):
            input_data = np.random.rand(64, 64).astype(np.float32)
            target_data = np.random.rand(64, 64).astype(np.float32)
            
            input_patch = Patch2D(
                data=input_data, slice_idx=i % 2, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            target_patch = Patch2D(
                data=target_data, slice_idx=i % 2, start_row=0, start_col=0,
                patch_size=(64, 64), original_slice_shape=(256, 256)
            )
            
            pairs.append(TrainingPair2D(input_patch, target_patch))
        
        return pairs


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingErrorRecovery:
    """Test error recovery mechanisms in training."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            device='cpu'
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_interruption_recovery(self):
        """Test recovery from training interruption."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup model and training state
        manager.model = nn.Linear(1, 1)
        manager._setup_training_components()
        
        # Simulate training state
        from src.training.training_manager import TrainingState, TrainingMetrics
        
        metrics = TrainingMetrics(
            epoch=1, train_loss=0.5, val_loss=0.4,
            train_time=10.0, val_time=5.0, learning_rate=1e-4
        )
        
        manager.training_state = TrainingState(
            epoch=1,
            best_val_loss=0.4,
            best_epoch=1,
            patience_counter=0,
            total_train_time=15.0,
            metrics_history=[metrics]
        )
        
        # Save checkpoint
        checkpoint_path = manager._save_checkpoint("recovery_test.pth")
        
        # Create new manager instance (simulating restart)
        new_manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        new_manager.model = nn.Linear(1, 1)
        new_manager._setup_training_components()
        
        # Load checkpoint
        success = new_manager.load_checkpoint(checkpoint_path)
        assert success is True
        
        # Verify state was restored
        assert new_manager.training_state.epoch == 1
        assert new_manager.training_state.best_val_loss == 0.4
        assert len(new_manager.training_state.metrics_history) == 1
    
    def test_training_state_persistence(self):
        """Test that training state is properly saved and can be restored."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Create training state
        from src.training.training_manager import TrainingState, TrainingMetrics
        
        metrics = [
            TrainingMetrics(1, 0.5, 0.4, 10.0, 5.0, 1e-4),
            TrainingMetrics(2, 0.3, 0.35, 10.0, 5.0, 1e-4)
        ]
        
        manager.training_state = TrainingState(
            epoch=2,
            best_val_loss=0.35,
            best_epoch=2,
            patience_counter=0,
            total_train_time=30.0,
            metrics_history=metrics
        )
        
        # Save training state
        manager._save_training_state()
        
        # Verify file was created
        state_file = Path(self.temp_dir) / "training_state.json"
        assert state_file.exists()
        
        # Load and verify state
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
        
        assert saved_state['epoch'] == 2
        assert saved_state['best_val_loss'] == 0.35
        assert saved_state['best_epoch'] == 2
        assert len(saved_state['metrics_history']) == 2


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_integration_import_error_without_torch():
    """Test that integration tests handle PyTorch unavailability."""
    # This test would run if PyTorch wasn't available
    # Just verify that the test framework works
    assert True