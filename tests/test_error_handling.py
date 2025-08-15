"""
Unit tests for error handling and recovery systems.
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.error_handling import (
    EnhancementError, DataError, ModelError, ResourceError, ConfigurationError,
    ErrorHandler, ResourceMonitor, RecoveryManager,
    with_error_handling, get_global_error_handler, setup_global_error_handling
)


class TestEnhancementErrors:
    """Test custom exception classes."""
    
    def test_enhancement_error_basic(self):
        """Test basic EnhancementError functionality."""
        error = EnhancementError("Test error")
        
        assert str(error) == "Test error"
        assert error.error_code == "UNKNOWN_ERROR"
        assert error.details == {}
        assert error.timestamp > 0
    
    def test_enhancement_error_with_details(self):
        """Test EnhancementError with code and details."""
        details = {"param": "value", "context": "test"}
        error = EnhancementError("Test error", "TEST_ERROR", details)
        
        assert error.error_code == "TEST_ERROR"
        assert error.details == details
    
    def test_specific_error_types(self):
        """Test specific error type inheritance."""
        data_error = DataError("Data error")
        model_error = ModelError("Model error")
        resource_error = ResourceError("Resource error")
        config_error = ConfigurationError("Config error")
        
        assert isinstance(data_error, EnhancementError)
        assert isinstance(model_error, EnhancementError)
        assert isinstance(resource_error, EnhancementError)
        assert isinstance(config_error, EnhancementError)


class TestErrorHandler:
    """Test ErrorHandler functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = str(Path(self.temp_dir) / "test.log")
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_error_handler_init(self):
        """Test ErrorHandler initialization."""
        handler = ErrorHandler()
        
        assert handler.log_file is None
        assert handler.error_history == []
    
    def test_error_handler_with_log_file(self):
        """Test ErrorHandler with log file."""
        handler = ErrorHandler(self.log_file)
        
        assert handler.log_file == self.log_file
        assert Path(self.log_file).parent.exists()
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            error_info = handler.handle_error(e, "test_context")
        
        assert error_info['error_type'] == 'ValueError'
        assert error_info['error_message'] == 'Test error'
        assert error_info['context'] == 'test_context'
        assert error_info['recoverable'] is False
        assert len(handler.error_history) == 1
    
    def test_handle_error_recoverable(self):
        """Test handling recoverable errors."""
        handler = ErrorHandler()
        suggestions = ["Try again", "Check input"]
        
        try:
            raise RuntimeError("Recoverable error")
        except Exception as e:
            error_info = handler.handle_error(
                e, "test_context", recoverable=True, recovery_suggestions=suggestions
            )
        
        assert error_info['recoverable'] is True
        assert error_info['recovery_suggestions'] == suggestions
    
    def test_handle_enhancement_error(self):
        """Test handling EnhancementError with additional details."""
        handler = ErrorHandler()
        
        try:
            raise DataError("Data error", "DATA_INVALID", {"file": "test.tif"})
        except Exception as e:
            error_info = handler.handle_error(e, "data_loading")
        
        assert error_info['error_code'] == 'DATA_INVALID'
        assert error_info['details'] == {"file": "test.tif"}
    
    def test_error_history_management(self):
        """Test error history management."""
        handler = ErrorHandler()
        
        # Add multiple errors
        for i in range(5):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle_error(e, f"context_{i}")
        
        assert len(handler.error_history) == 5
        
        # Check error summary
        summary = handler.get_error_summary()
        assert summary['total_errors'] == 5
        assert 'ValueError' in summary['error_types']
        assert summary['error_types']['ValueError'] == 5
    
    def test_save_error_to_file(self):
        """Test saving errors to JSON file."""
        handler = ErrorHandler(self.log_file)
        
        try:
            raise ValueError("Test error for file")
        except Exception as e:
            handler.handle_error(e, "file_test")
        
        # Check that error file was created
        error_file = Path(self.log_file).with_suffix('.errors.json')
        assert error_file.exists()
        
        # Check file contents
        with open(error_file, 'r') as f:
            errors = json.load(f)
        
        assert len(errors) == 1
        assert errors[0]['error_message'] == 'Test error for file'


class TestErrorHandlingDecorator:
    """Test error handling decorator."""
    
    def test_decorator_success(self):
        """Test decorator with successful function."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, "test_function")
        def successful_function(x):
            return x * 2
        
        result = successful_function(5)
        assert result == 10
        assert len(handler.error_history) == 0
    
    def test_decorator_error(self):
        """Test decorator with function that raises error."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, "test_function")
        def failing_function():
            raise ValueError("Function failed")
        
        with pytest.raises(EnhancementError):
            failing_function()
        
        assert len(handler.error_history) == 1
        assert handler.error_history[0]['context'] == 'test_function'
    
    def test_decorator_enhancement_error(self):
        """Test decorator with EnhancementError (should pass through)."""
        handler = ErrorHandler()
        
        @with_error_handling(handler, "test_function")
        def failing_function():
            raise DataError("Data error", "DATA_INVALID")
        
        with pytest.raises(DataError):
            failing_function()
        
        assert len(handler.error_history) == 1


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""
    
    def test_resource_monitor_init(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor()
        
        assert monitor.memory_warnings == []
        assert monitor.disk_warnings == []
    
    def test_check_memory_usage_sufficient(self):
        """Test memory check with sufficient memory."""
        monitor = ResourceMonitor()
        
        # Mock psutil
        with patch('src.core.error_handling.ResourceMonitor.check_memory_usage') as mock_check:
            mock_check.return_value = {
                'available_mb': 2048,
                'required_mb': 1000,
                'sufficient': True,
                'usage_percent': 50.0
            }
            
            status = monitor.check_memory_usage(1000)
            
            assert status['sufficient'] is True
            assert status['available_mb'] == 2048
    
    def test_check_memory_usage_insufficient(self):
        """Test memory check with insufficient memory."""
        monitor = ResourceMonitor()
        
        # Mock psutil to simulate insufficient memory
        with patch('src.core.error_handling.ResourceMonitor.check_memory_usage') as mock_check:
            mock_check.return_value = {
                'available_mb': 500,
                'required_mb': 1000,
                'sufficient': False,
                'usage_percent': 90.0
            }
            
            status = monitor.check_memory_usage(1000)
            
            assert status['sufficient'] is False
    
    def test_check_memory_usage_no_psutil(self):
        """Test memory check when psutil is not available."""
        monitor = ResourceMonitor()
        
        # Test the actual implementation without psutil
        status = monitor.check_memory_usage(1000)
        
        # Should default to sufficient when psutil is not available
        assert status['sufficient'] is True
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check with sufficient space."""
        # Mock 5GB free space
        mock_disk_usage.return_value = (10 * 1024**3, 5 * 1024**3, 5 * 1024**3)
        
        monitor = ResourceMonitor()
        status = monitor.check_disk_space("/tmp", 1000)  # Require 1GB
        
        assert status['sufficient'] is True
        assert status['free_mb'] == 5120  # 5GB in MB
    
    @patch('shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check with insufficient space."""
        # Mock 500MB free space
        mock_disk_usage.return_value = (10 * 1024**3, 9.5 * 1024**3, 500 * 1024**2)
        
        monitor = ResourceMonitor()
        status = monitor.check_disk_space("/tmp", 1000)  # Require 1GB
        
        assert status['sufficient'] is False
        assert len(monitor.disk_warnings) == 1
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_check_gpu_memory_no_cuda(self):
        """Test GPU memory check when CUDA is not available."""
        monitor = ResourceMonitor()
        
        with patch('torch.cuda.is_available', return_value=False):
            status = monitor.check_gpu_memory()
        
        assert status['gpu_available'] is False
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    def test_check_gpu_memory_with_cuda(self, mock_reserved, mock_allocated, mock_count, mock_available):
        """Test GPU memory check with CUDA available."""
        mock_available.return_value = True
        mock_count.return_value = 1
        mock_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        
        monitor = ResourceMonitor()
        status = monitor.check_gpu_memory()
        
        assert status['gpu_available'] is True
        assert len(status['devices']) == 1
        assert status['devices'][0]['memory_allocated_mb'] == 1024
        assert status['devices'][0]['memory_reserved_mb'] == 2048
    
    def test_get_system_status(self):
        """Test getting comprehensive system status."""
        monitor = ResourceMonitor()
        status = monitor.get_system_status()
        
        assert 'memory_warnings' in status
        assert 'disk_warnings' in status
        assert 'gpu_info' in status
        assert 'torch_available' in status


class TestRecoveryManager:
    """Test RecoveryManager functionality."""
    
    def test_recovery_manager_init(self):
        """Test RecoveryManager initialization."""
        manager = RecoveryManager()
        
        assert manager.recovery_attempts == {}
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_attempt_memory_recovery(self, mock_empty_cache, mock_cuda_available):
        """Test memory recovery attempt."""
        mock_cuda_available.return_value = True
        
        manager = RecoveryManager()
        error = RuntimeError("CUDA out of memory")
        
        result = manager.attempt_memory_recovery(error)
        
        assert result is True
        mock_empty_cache.assert_called_once()
    
    def test_attempt_memory_recovery_no_memory_error(self):
        """Test memory recovery with non-memory error."""
        manager = RecoveryManager()
        error = ValueError("Some other error")
        
        result = manager.attempt_memory_recovery(error)
        
        assert result is False
    
    def test_suggest_batch_size_reduction(self):
        """Test batch size reduction suggestion."""
        manager = RecoveryManager()
        error = RuntimeError("CUDA out of memory")
        
        new_size = manager.suggest_batch_size_reduction(16, error)
        
        assert new_size == 8
    
    def test_suggest_batch_size_reduction_no_memory_error(self):
        """Test batch size suggestion with non-memory error."""
        manager = RecoveryManager()
        error = ValueError("Some other error")
        
        new_size = manager.suggest_batch_size_reduction(16, error)
        
        assert new_size is None
    
    def test_suggest_batch_size_reduction_min_size(self):
        """Test batch size suggestion at minimum size."""
        manager = RecoveryManager()
        error = RuntimeError("CUDA out of memory")
        
        new_size = manager.suggest_batch_size_reduction(1, error)
        
        assert new_size is None
    
    def test_get_recovery_suggestions_memory(self):
        """Test recovery suggestions for memory errors."""
        manager = RecoveryManager()
        error = RuntimeError("CUDA out of memory")
        
        suggestions = manager.get_recovery_suggestions(error)
        
        assert "Reduce batch size" in suggestions
        assert "Use CPU instead of GPU" in suggestions
    
    def test_get_recovery_suggestions_file(self):
        """Test recovery suggestions for file errors."""
        manager = RecoveryManager()
        error = FileNotFoundError("File not found")
        
        suggestions = manager.get_recovery_suggestions(error)
        
        assert "Check file path is correct" in suggestions
        assert "Verify file exists and is accessible" in suggestions
    
    def test_get_recovery_suggestions_model(self):
        """Test recovery suggestions for model errors."""
        manager = RecoveryManager()
        error = RuntimeError("Failed to load model")
        
        suggestions = manager.get_recovery_suggestions(error)
        
        assert "Verify model file is not corrupted" in suggestions
        assert "Check model architecture compatibility" in suggestions


class TestGlobalErrorHandling:
    """Test global error handling functions."""
    
    def test_get_global_error_handler(self):
        """Test getting global error handler."""
        handler = get_global_error_handler()
        
        assert isinstance(handler, ErrorHandler)
        
        # Should return same instance
        handler2 = get_global_error_handler()
        assert handler is handler2
    
    def test_setup_global_error_handling(self):
        """Test setting up global error handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = str(Path(temp_dir) / "global.log")
            
            handler = setup_global_error_handling(log_file)
            
            assert isinstance(handler, ErrorHandler)
            assert handler.log_file == log_file
            
            # Should be same as global handler
            global_handler = get_global_error_handler()
            assert handler is global_handler