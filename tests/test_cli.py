"""
Unit tests for command-line interfaces.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.cli.train_cli import TrainingCLI
from src.cli.inference_cli import InferenceCLI
from src.testing.test_fixtures import TestDataFixtures


class TestTrainingCLI:
    """Test training command-line interface."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
        self.cli = TrainingCLI()
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_init(self):
        """Test CLI initialization."""
        assert self.cli.parser is not None
        assert self.cli.logger is not None
    
    def test_parser_required_arguments(self):
        """Test parser with missing required arguments."""
        with pytest.raises(SystemExit):
            self.cli.parser.parse_args([])
    
    def test_parser_basic_arguments(self):
        """Test parser with basic required arguments."""
        args = self.cli.parser.parse_args([
            '--input-dir', 'inputs',
            '--target-dir', 'targets', 
            '--output-dir', 'outputs'
        ])
        
        assert args.input_dir == 'inputs'
        assert args.target_dir == 'targets'
        assert args.output_dir == 'outputs'
        assert args.epochs == 100  # default
        assert args.batch_size == 16  # default
    
    def test_parser_custom_arguments(self):
        """Test parser with custom arguments."""
        args = self.cli.parser.parse_args([
            '--input-dir', 'inputs',
            '--target-dir', 'targets',
            '--output-dir', 'outputs',
            '--epochs', '50',
            '--batch-size', '8',
            '--learning-rate', '1e-3',
            '--device', 'cuda',
            '--patch-size', '512', '512'
        ])
        
        assert args.epochs == 50
        assert args.batch_size == 8
        assert args.learning_rate == 1e-3
        assert args.device == 'cuda'
        assert args.patch_size == [512, 512]
    
    def test_create_training_config(self):
        """Test creating training configuration from arguments."""
        args = self.cli.parser.parse_args([
            '--input-dir', 'inputs',
            '--target-dir', 'targets',
            '--output-dir', 'outputs',
            '--epochs', '20',
            '--batch-size', '4'
        ])
        
        config = self.cli._create_training_config(args)
        
        assert config.epochs == 20
        assert config.batch_size == 4
        assert config.patch_size == (256, 256)  # default
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        # Create test config
        args = self.cli.parser.parse_args([
            '--input-dir', 'inputs',
            '--target-dir', 'targets',
            '--output-dir', 'outputs'
        ])
        
        config = self.cli._create_training_config(args)
        
        # Save config
        config_file = str(Path(self.temp_dir) / "test_config.json")
        self.cli._save_config_to_file(config, config_file)
        
        assert Path(config_file).exists()
        
        # Load config
        loaded_config = self.cli._load_config_from_file(config_file)
        
        assert loaded_config['epochs'] == config.epochs
        assert loaded_config['batch_size'] == config.batch_size
    
    def test_find_tiff_files(self):
        """Test finding TIFF files in directory."""
        # Create test TIFF files
        test_dir = Path(self.temp_dir) / "tiff_test"
        test_dir.mkdir()
        
        (test_dir / "file1.tif").touch()
        (test_dir / "file2.tiff").touch()
        (test_dir / "file3.txt").touch()  # Should be ignored
        
        files = self.cli._find_tiff_files(str(test_dir))
        
        assert len(files) == 2
        assert all(f.endswith(('.tif', '.tiff')) for f in files)
    
    def test_find_tiff_files_max_limit(self):
        """Test finding TIFF files with maximum limit."""
        test_dir = Path(self.temp_dir) / "tiff_limit_test"
        test_dir.mkdir()
        
        # Create 5 TIFF files
        for i in range(5):
            (test_dir / f"file{i}.tif").touch()
        
        files = self.cli._find_tiff_files(str(test_dir), max_files=3)
        
        assert len(files) == 3
    
    def test_find_tiff_files_nonexistent_dir(self):
        """Test finding TIFF files in non-existent directory."""
        with pytest.raises(FileNotFoundError):
            self.cli._find_tiff_files("nonexistent_directory")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('src.cli.train_cli.TrainingManager')
    def test_run_success(self, mock_training_manager_class):
        """Test successful CLI run."""
        # Setup mocks
        mock_manager = Mock()
        mock_training_manager_class.return_value = mock_manager
        
        mock_manager.prepare_training_data.return_value = (Mock(), Mock())
        mock_manager.fine_tune_model.return_value = "best_model.pth"
        mock_manager.export_model_for_inference.return_value = None
        mock_manager.get_training_summary.return_value = {
            'total_epochs': 5,
            'best_validation_loss': 0.1,
            'total_training_time': 100.0
        }
        
        # Create test files
        input_dir = Path(self.temp_dir) / "inputs"
        target_dir = Path(self.temp_dir) / "targets"
        input_dir.mkdir()
        target_dir.mkdir()
        
        (input_dir / "input1.tif").touch()
        (target_dir / "target1.tif").touch()
        
        # Create mock model
        model_path = self.fixtures.save_mock_model("pretrained.pth")
        
        # Run CLI
        args = [
            '--input-dir', str(input_dir),
            '--target-dir', str(target_dir),
            '--output-dir', str(Path(self.temp_dir) / "output"),
            '--pretrained-model', model_path,
            '--epochs', '5',
            '--log-level', 'ERROR'  # Reduce log output
        ]
        
        result = self.cli.run(args)
        
        assert result == 0  # Success
        mock_manager.prepare_training_data.assert_called_once()
        mock_manager.fine_tune_model.assert_called_once()
    
    def test_run_no_pytorch(self):
        """Test CLI run without PyTorch."""
        with patch('src.cli.train_cli.TORCH_AVAILABLE', False):
            result = self.cli.run(['--input-dir', 'in', '--target-dir', 'tgt', '--output-dir', 'out'])
            assert result == 1  # Failure


class TestInferenceCLI:
    """Test inference command-line interface."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
        self.cli = InferenceCLI()
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_init(self):
        """Test CLI initialization."""
        assert self.cli.parser is not None
        assert self.cli.logger is not None
    
    def test_parser_required_arguments(self):
        """Test parser with missing required arguments."""
        with pytest.raises(SystemExit):
            self.cli.parser.parse_args([])
    
    def test_parser_single_file_arguments(self):
        """Test parser with single file arguments."""
        args = self.cli.parser.parse_args([
            '--model', 'model.pth',
            '--input', 'input.tif',
            '--output', 'output.tif'
        ])
        
        assert args.model == 'model.pth'
        assert args.input == 'input.tif'
        assert args.output == 'output.tif'
        assert args.patch_size == [256, 256]  # default
    
    def test_parser_batch_arguments(self):
        """Test parser with batch processing arguments."""
        args = self.cli.parser.parse_args([
            '--model', 'model.pth',
            '--input-dir', 'inputs',
            '--output-dir', 'outputs'
        ])
        
        assert args.model == 'model.pth'
        assert args.input_dir == 'inputs'
        assert args.output_dir == 'outputs'
    
    def test_parser_custom_arguments(self):
        """Test parser with custom arguments."""
        args = self.cli.parser.parse_args([
            '--model', 'model.pth',
            '--input', 'input.tif',
            '--output', 'output.tif',
            '--patch-size', '512', '512',
            '--batch-size', '8',
            '--device', 'cuda',
            '--calculate-metrics'
        ])
        
        assert args.patch_size == [512, 512]
        assert args.batch_size == 8
        assert args.device == 'cuda'
        assert args.calculate_metrics is True
    
    def test_create_inference_config(self):
        """Test creating inference configuration from arguments."""
        args = self.cli.parser.parse_args([
            '--model', 'model.pth',
            '--input', 'input.tif',
            '--output', 'output.tif',
            '--batch-size', '16',
            '--memory-limit', '4.0'
        ])
        
        config = self.cli._create_inference_config(args)
        
        assert config.batch_size == 16
        assert config.memory_limit_gb == 4.0
        assert config.patch_size == (256, 256)  # default
    
    def test_config_file_operations(self):
        """Test configuration file save/load operations."""
        # Create test config
        args = self.cli.parser.parse_args([
            '--model', 'model.pth',
            '--input', 'input.tif',
            '--output', 'output.tif'
        ])
        
        config = self.cli._create_inference_config(args)
        
        # Save config
        config_file = str(Path(self.temp_dir) / "inference_config.json")
        self.cli._save_config_to_file(config, config_file)
        
        assert Path(config_file).exists()
        
        # Load config
        loaded_config = self.cli._load_config_from_file(config_file)
        
        assert loaded_config['batch_size'] == config.batch_size
        assert loaded_config['patch_size'] == list(config.patch_size)
    
    def test_find_input_files(self):
        """Test finding input files for batch processing."""
        # Create test files
        test_dir = Path(self.temp_dir) / "input_test"
        test_dir.mkdir()
        
        (test_dir / "image1.tif").touch()
        (test_dir / "image2.tiff").touch()
        (test_dir / "image3.jpg").touch()  # Should be ignored with default pattern
        
        files = self.cli._find_input_files(str(test_dir), "*.tif", False, None)
        
        assert len(files) == 1  # Only .tif files
        assert files[0].endswith("image1.tif")
    
    def test_find_input_files_custom_pattern(self):
        """Test finding input files with custom pattern."""
        test_dir = Path(self.temp_dir) / "pattern_test"
        test_dir.mkdir()
        
        (test_dir / "image1.tif").touch()
        (test_dir / "image2.tiff").touch()
        
        files = self.cli._find_input_files(str(test_dir), "*.tif*", False, None)
        
        assert len(files) == 2  # Both .tif and .tiff
    
    def test_save_metrics(self):
        """Test saving metrics to JSON file."""
        metrics = {
            'success': True,
            'processing_time': 10.5,
            'quality_metrics': {'psnr': 30.0}
        }
        
        metrics_file = str(Path(self.temp_dir) / "metrics.json")
        self.cli._save_metrics(metrics, metrics_file)
        
        assert Path(metrics_file).exists()
        
        # Load and verify
        with open(metrics_file, 'r') as f:
            loaded_metrics = json.load(f)
        
        assert loaded_metrics['success'] is True
        assert loaded_metrics['processing_time'] == 10.5
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('src.cli.inference_cli.ImageEnhancementAPI')
    def test_run_single_file_success(self, mock_api_class):
        """Test successful single file processing."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        mock_api.load_model.return_value = True
        mock_api.get_system_info.return_value = {
            'device': 'cpu',
            'model_loaded': True
        }
        
        # Mock enhancement result
        mock_result = Mock()
        mock_result.success = True
        mock_result.processing_time = 5.0
        mock_result.quality_metrics = {'psnr': 25.0}
        mock_api.enhance_3d_tiff.return_value = mock_result
        mock_api.get_enhancement_metrics.return_value = {'success': True}
        
        # Create test files
        input_file = Path(self.temp_dir) / "input.tif"
        input_file.touch()
        output_file = Path(self.temp_dir) / "output.tif"
        
        model_path = self.fixtures.save_mock_model("inference_model.pth")
        
        # Run CLI
        args = [
            '--model', model_path,
            '--input', str(input_file),
            '--output', str(output_file),
            '--log-level', 'ERROR'  # Reduce log output
        ]
        
        result = self.cli.run(args)
        
        assert result == 0  # Success
        mock_api.load_model.assert_called_once_with(model_path)
        mock_api.enhance_3d_tiff.assert_called_once()
    
    def test_run_no_pytorch(self):
        """Test CLI run without PyTorch."""
        with patch('src.cli.inference_cli.TORCH_AVAILABLE', False):
            result = self.cli.run(['--model', 'model.pth', '--input', 'in.tif', '--output', 'out.tif'])
            assert result == 1  # Failure
    
    def test_argument_validation(self):
        """Test argument validation."""
        # Test missing output with input - this should be caught at runtime
        result = self.cli.run([
            '--model', 'model.pth',
            '--input', 'input.tif'
            # Missing --output
        ])
        assert result == 1  # Should fail
        
        # Test missing output-dir with input-dir - this should be caught at runtime
        result = self.cli.run([
            '--model', 'model.pth',
            '--input-dir', 'inputs'
            # Missing --output-dir
        ])
        assert result == 1  # Should fail
    
    def test_progress_callback_creation(self):
        """Test progress callback creation."""
        # Test with progress enabled
        callback = self.cli._create_progress_callback(True, "test.tif")
        assert callback is not None
        
        # Test callback execution (should not raise)
        callback("Processing", 0.5)
        callback("Complete", 1.0)
        
        # Test with progress disabled
        callback = self.cli._create_progress_callback(False)
        assert callback is None


# Integration tests for CLI components
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCLIIntegration:
    """Integration tests for CLI components."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.fixtures = TestDataFixtures(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_help_commands(self):
        """Test help commands for both CLIs."""
        train_cli = TrainingCLI()
        inference_cli = InferenceCLI()
        
        # Test help doesn't crash
        with pytest.raises(SystemExit):  # argparse exits with help
            train_cli.run(['--help'])
        
        with pytest.raises(SystemExit):  # argparse exits with help
            inference_cli.run(['--help'])
    
    def test_config_file_workflow(self):
        """Test complete workflow with configuration files."""
        # Create training config
        train_config = {
            "epochs": 5,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "device": "cpu"
        }
        
        train_config_file = Path(self.temp_dir) / "train_config.json"
        with open(train_config_file, 'w') as f:
            json.dump(train_config, f)
        
        # Create inference config
        inference_config = {
            "patch_size": [128, 128],
            "batch_size": 4,
            "device": "cpu"
        }
        
        inference_config_file = Path(self.temp_dir) / "inference_config.json"
        with open(inference_config_file, 'w') as f:
            json.dump(inference_config, f)
        
        # Test loading configs
        train_cli = TrainingCLI()
        loaded_train_config = train_cli._load_config_from_file(str(train_config_file))
        assert loaded_train_config['epochs'] == 5
        
        inference_cli = InferenceCLI()
        loaded_inference_config = inference_cli._load_config_from_file(str(inference_config_file))
        assert loaded_inference_config['batch_size'] == 4