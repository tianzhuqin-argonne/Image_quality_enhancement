"""
Unit tests for TrainingManager.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.training.training_manager import TrainingManager, TrainingMetrics, TrainingState
from src.core.config import TrainingConfig


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            epoch=1,
            train_loss=0.5,
            val_loss=0.4,
            train_time=10.0,
            val_time=5.0,
            learning_rate=1e-4
        )
        
        assert metrics.epoch == 1
        assert metrics.train_loss == 0.5
        assert metrics.val_loss == 0.4
        assert metrics.train_time == 10.0
        assert metrics.val_time == 5.0
        assert metrics.learning_rate == 1e-4
    
    def test_training_metrics_to_dict(self):
        """Test converting TrainingMetrics to dictionary."""
        metrics = TrainingMetrics(
            epoch=1, train_loss=0.5, val_loss=0.4,
            train_time=10.0, val_time=5.0, learning_rate=1e-4
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict['epoch'] == 1
        assert metrics_dict['train_loss'] == 0.5
        assert metrics_dict['val_loss'] == 0.4


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingState:
    """Test TrainingState dataclass."""
    
    def test_training_state_creation(self):
        """Test creating TrainingState."""
        state = TrainingState(
            epoch=10,
            best_val_loss=0.3,
            best_epoch=8,
            patience_counter=2,
            total_train_time=100.0,
            metrics_history=[]
        )
        
        assert state.epoch == 10
        assert state.best_val_loss == 0.3
        assert state.best_epoch == 8
        assert state.patience_counter == 2
        assert state.total_train_time == 100.0
        assert state.metrics_history == []
    
    def test_training_state_serialization(self):
        """Test TrainingState serialization and deserialization."""
        metrics = TrainingMetrics(
            epoch=1, train_loss=0.5, val_loss=0.4,
            train_time=10.0, val_time=5.0, learning_rate=1e-4
        )
        
        state = TrainingState(
            epoch=1,
            best_val_loss=0.4,
            best_epoch=1,
            patience_counter=0,
            total_train_time=15.0,
            metrics_history=[metrics]
        )
        
        # Convert to dict and back
        state_dict = state.to_dict()
        restored_state = TrainingState.from_dict(state_dict)
        
        assert restored_state.epoch == state.epoch
        assert restored_state.best_val_loss == state.best_val_loss
        assert restored_state.best_epoch == state.best_epoch
        assert restored_state.patience_counter == state.patience_counter
        assert restored_state.total_train_time == state.total_train_time
        assert len(restored_state.metrics_history) == 1
        assert restored_state.metrics_history[0].epoch == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingManager:
    """Test TrainingManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            device='cpu',
            checkpoint_interval=1,
            early_stopping_patience=5
        )
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_manager_init(self):
        """Test TrainingManager initialization."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        assert manager.config == self.config
        assert manager.output_dir == Path(self.temp_dir)
        assert manager.use_tensorboard is False
        assert manager.training_state is None
        assert manager.model is None
        
        # Check directories were created
        assert (Path(self.temp_dir) / "checkpoints").exists()
        assert (Path(self.temp_dir) / "logs").exists()
    
    @patch('src.training.training_manager.TrainingDataLoader')
    def test_prepare_training_data(self, mock_data_loader_class):
        """Test preparing training data."""
        # Setup mocks
        mock_data_loader = Mock()
        mock_data_loader_class.return_value = mock_data_loader
        
        # Mock training pairs
        mock_pairs = [Mock() for _ in range(10)]
        mock_data_loader.load_training_data.return_value = mock_pairs
        mock_data_loader.get_data_statistics.return_value = {'num_pairs': 10}
        
        # Mock data loaders
        mock_train_loader = Mock()
        mock_train_loader.__len__ = Mock(return_value=5)
        mock_val_loader = Mock()
        mock_val_loader.__len__ = Mock(return_value=2)
        mock_data_loader.create_data_loaders.return_value = (mock_train_loader, mock_val_loader)
        
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Test data preparation
        train_loader, val_loader = manager.prepare_training_data(
            input_paths=['input1.tif'],
            target_paths=['target1.tif']
        )
        
        assert train_loader == mock_train_loader
        assert val_loader == mock_val_loader
        
        # Check that statistics were saved
        stats_file = Path(self.temp_dir) / "data_statistics.json"
        assert stats_file.exists()
    
    def test_setup_training_components(self):
        """Test setting up training components."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Create a simple model for testing
        manager.model = nn.Linear(10, 1)
        
        # Setup components
        manager._setup_training_components()
        
        assert manager.optimizer is not None
        assert manager.scheduler is not None
        assert manager.criterion is not None
        assert isinstance(manager.criterion, nn.MSELoss)
    
    def test_setup_training_components_different_configs(self):
        """Test setup with different configurations."""
        config = TrainingConfig(
            optimizer="adamw",
            loss_function="l1",
            use_scheduler=False
        )
        
        manager = TrainingManager(
            config=config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        manager.model = nn.Linear(10, 1)
        manager._setup_training_components()
        
        assert isinstance(manager.optimizer, torch.optim.AdamW)
        assert isinstance(manager.criterion, nn.L1Loss)
        assert manager.scheduler is None
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup simple model and components
        manager.model = nn.Linear(1, 1)
        manager._setup_training_components()
        
        # Create simple data loader
        inputs = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        dataset = TensorDataset(inputs, targets)
        train_loader = DataLoader(dataset, batch_size=2)
        
        # Train one epoch
        avg_loss, train_time = manager._train_epoch(train_loader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(train_time, float)
        assert train_time >= 0
    
    def test_validate_epoch(self):
        """Test validation for one epoch."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup simple model and components
        manager.model = nn.Linear(1, 1)
        manager._setup_training_components()
        
        # Create simple data loader
        inputs = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        dataset = TensorDataset(inputs, targets)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # Validate one epoch
        avg_loss, val_time = manager._validate_epoch(val_loader)
        
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert isinstance(val_time, float)
        assert val_time >= 0
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup model and training state
        manager.model = nn.Linear(1, 1)
        manager._setup_training_components()
        manager.training_state = TrainingState(
            epoch=1, best_val_loss=0.5, best_epoch=1,
            patience_counter=0, total_train_time=10.0, metrics_history=[]
        )
        
        # Save checkpoint
        checkpoint_path = manager._save_checkpoint("test_checkpoint.pth")
        
        assert Path(checkpoint_path).exists()
        
        # Test loading checkpoint
        success = manager.load_checkpoint(checkpoint_path)
        assert success is True
    
    @patch('src.training.training_manager.UNetModelManager')
    def test_validate_model(self, mock_model_manager_class):
        """Test model validation."""
        # Setup mocks
        mock_model_manager = Mock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_model = nn.Linear(1, 1)
        mock_model_manager.load_pretrained_model.return_value = mock_model
        
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Set up model before calling _setup_training_components
        manager.model = nn.Linear(1, 1)
        manager._setup_training_components()
        
        # Create validation data
        inputs = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        dataset = TensorDataset(inputs, targets)
        val_loader = DataLoader(dataset, batch_size=2)
        
        # Validate model
        metrics = manager.validate_model("dummy_model.pth", val_loader)
        
        assert isinstance(metrics, dict)
        assert 'validation_loss' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'psnr' in metrics
        assert 'num_samples' in metrics
    
    def test_export_model_for_inference(self):
        """Test exporting model for inference."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Create a dummy checkpoint
        model = nn.Linear(1, 1)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'training_state': {
                'best_epoch': 5,
                'best_val_loss': 0.3
            }
        }
        
        checkpoint_path = Path(self.temp_dir) / "test_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Export model
        export_path = Path(self.temp_dir) / "exported_model.pth"
        manager.export_model_for_inference(str(checkpoint_path), str(export_path))
        
        assert export_path.exists()
        
        # Check exported model structure
        exported = torch.load(export_path)
        assert 'model_state_dict' in exported
        assert 'model_config' in exported
        assert 'training_info' in exported
    
    def test_get_training_summary_empty(self):
        """Test getting training summary when no training has occurred."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        summary = manager.get_training_summary()
        assert summary == {}
    
    def test_get_training_summary_with_state(self):
        """Test getting training summary with training state."""
        manager = TrainingManager(
            config=self.config,
            output_dir=self.temp_dir,
            use_tensorboard=False
        )
        
        # Setup training state
        metrics = TrainingMetrics(
            epoch=1, train_loss=0.5, val_loss=0.4,
            train_time=10.0, val_time=5.0, learning_rate=1e-4
        )
        
        manager.training_state = TrainingState(
            epoch=1, best_val_loss=0.4, best_epoch=1,
            patience_counter=0, total_train_time=15.0, metrics_history=[metrics]
        )
        
        summary = manager.get_training_summary()
        
        assert 'total_epochs' in summary
        assert 'best_epoch' in summary
        assert 'best_validation_loss' in summary
        assert 'total_training_time' in summary
        assert 'final_learning_rate' in summary
        assert 'training_config' in summary
        
        assert summary['total_epochs'] == 1
        assert summary['best_epoch'] == 1
        assert summary['best_validation_loss'] == 0.4


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_import_error_without_torch():
    """Test that appropriate errors are raised when PyTorch is not available."""
    config = TrainingConfig()
    
    with pytest.raises(ImportError, match="PyTorch is required"):
        TrainingManager(config)