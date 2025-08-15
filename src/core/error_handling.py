"""
Comprehensive error handling and recovery utilities.

This module provides error handling, logging, and recovery mechanisms
for both training and inference pipelines.
"""

import logging
import traceback
import functools
import time
from typing import Dict, Any, Optional, Callable, Union, List
from pathlib import Path
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancementError(Exception):
    """Base exception for enhancement system errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        self.timestamp = time.time()


class DataError(EnhancementError):
    """Errors related to data loading, validation, or processing."""
    pass


class ModelError(EnhancementError):
    """Errors related to model loading, validation, or inference."""
    pass


class ResourceError(EnhancementError):
    """Errors related to system resources (memory, disk, etc.)."""
    pass


class ConfigurationError(EnhancementError):
    """Errors related to configuration or setup."""
    pass


class ErrorHandler:
    """
    Centralized error handling and recovery system.
    
    Provides consistent error handling, logging, and recovery mechanisms
    across the enhancement system.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize error handler.
        
        Args:
            log_file: Optional path to log file for error recording
        """
        self.log_file = log_file
        self.error_history: List[Dict[str, Any]] = []
        
        # Setup logging
        if log_file:
            self._setup_file_logging(log_file)
    
    def _setup_file_logging(self, log_file: str) -> None:
        """Setup file logging for errors."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        recoverable: bool = False,
        recovery_suggestions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Handle and log an error with context and recovery information.
        
        Args:
            error: The exception that occurred
            context: Context description where error occurred
            recoverable: Whether the error is potentially recoverable
            recovery_suggestions: List of suggested recovery actions
            
        Returns:
            Dictionary with error information
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'recoverable': recoverable,
            'recovery_suggestions': recovery_suggestions or [],
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }
        
        # Add specific error details
        if isinstance(error, EnhancementError):
            error_info['error_code'] = error.error_code
            error_info['details'] = error.details
        
        # Log error
        logger.error(f"Error in {context}: {error}")
        if recoverable:
            logger.info(f"Error is recoverable. Suggestions: {recovery_suggestions}")
        
        # Store in history
        self.error_history.append(error_info)
        
        # Save to file if configured
        if self.log_file:
            self._save_error_to_file(error_info)
        
        return error_info
    
    def _save_error_to_file(self, error_info: Dict[str, Any]) -> None:
        """Save error information to JSON file."""
        try:
            error_log_path = Path(self.log_file).with_suffix('.errors.json')
            
            # Load existing errors
            existing_errors = []
            if error_log_path.exists():
                try:
                    with open(error_log_path, 'r') as f:
                        existing_errors = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_errors = []
            
            # Add new error
            existing_errors.append(error_info)
            
            # Keep only last 100 errors
            if len(existing_errors) > 100:
                existing_errors = existing_errors[-100:]
            
            # Save back
            with open(error_log_path, 'w') as f:
                json.dump(existing_errors, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Failed to save error to file: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {'total_errors': 0, 'recent_errors': []}
        
        recent_errors = self.error_history[-10:]  # Last 10 errors
        error_types = {}
        
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': recent_errors,
            'recoverable_errors': sum(1 for e in self.error_history if e['recoverable'])
        }


def with_error_handling(
    error_handler: ErrorHandler,
    context: str = "",
    recoverable: bool = False,
    recovery_suggestions: List[str] = None
):
    """
    Decorator for automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance
        context: Context description
        recoverable: Whether errors are recoverable
        recovery_suggestions: Recovery suggestions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(
                    e, context or func.__name__, recoverable, recovery_suggestions
                )
                
                # Re-raise with additional context
                if isinstance(e, EnhancementError):
                    raise
                else:
                    raise EnhancementError(
                        f"Error in {context or func.__name__}: {str(e)}",
                        error_code="FUNCTION_ERROR",
                        details=error_info
                    ) from e
        
        return wrapper
    return decorator


class ResourceMonitor:
    """
    Monitor system resources and detect potential issues.
    
    Helps prevent resource-related errors by monitoring memory usage,
    disk space, and other system resources.
    """
    
    def __init__(self):
        """Initialize resource monitor."""
        self.memory_warnings = []
        self.disk_warnings = []
    
    def check_memory_usage(self, required_mb: float) -> Dict[str, Any]:
        """
        Check if sufficient memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            Dictionary with memory status
        """
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 ** 2)
            
            status = {
                'available_mb': available_mb,
                'required_mb': required_mb,
                'sufficient': available_mb >= required_mb,
                'usage_percent': memory.percent
            }
            
            if not status['sufficient']:
                warning = f"Insufficient memory: {available_mb:.1f}MB available, {required_mb:.1f}MB required"
                self.memory_warnings.append(warning)
                logger.warning(warning)
            
            return status
            
        except ImportError:
            logger.warning("psutil not available, cannot check memory usage")
            return {'available_mb': float('inf'), 'sufficient': True}
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {'available_mb': 0, 'sufficient': False, 'error': str(e)}
    
    def check_disk_space(self, path: str, required_mb: float) -> Dict[str, Any]:
        """
        Check if sufficient disk space is available.
        
        Args:
            path: Path to check disk space for
            required_mb: Required disk space in MB
            
        Returns:
            Dictionary with disk space status
        """
        try:
            import shutil
            
            total, used, free = shutil.disk_usage(path)
            free_mb = free / (1024 ** 2)
            
            status = {
                'free_mb': free_mb,
                'required_mb': required_mb,
                'sufficient': free_mb >= required_mb,
                'path': path
            }
            
            if not status['sufficient']:
                warning = f"Insufficient disk space at {path}: {free_mb:.1f}MB free, {required_mb:.1f}MB required"
                self.disk_warnings.append(warning)
                logger.warning(warning)
            
            return status
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return {'free_mb': 0, 'sufficient': False, 'error': str(e)}
    
    def check_gpu_memory(self) -> Dict[str, Any]:
        """
        Check GPU memory usage if CUDA is available.
        
        Returns:
            Dictionary with GPU memory status
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {'gpu_available': False}
        
        try:
            device_count = torch.cuda.device_count()
            gpu_info = {'gpu_available': True, 'devices': []}
            
            for i in range(device_count):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
                
                # Get total memory (this is approximate)
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)
                except:
                    total_memory = None
                
                device_info = {
                    'device_id': i,
                    'memory_allocated_mb': memory_allocated,
                    'memory_reserved_mb': memory_reserved,
                    'total_memory_mb': total_memory
                }
                
                gpu_info['devices'].append(device_info)
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return {'gpu_available': True, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'memory_warnings': self.memory_warnings,
            'disk_warnings': self.disk_warnings,
            'gpu_info': self.check_gpu_memory(),
            'torch_available': TORCH_AVAILABLE
        }


class RecoveryManager:
    """
    Manage recovery strategies for common error scenarios.
    
    Provides automatic recovery mechanisms for recoverable errors.
    """
    
    def __init__(self):
        """Initialize recovery manager."""
        self.recovery_attempts = {}
    
    def attempt_memory_recovery(self, error: Exception) -> bool:
        """
        Attempt to recover from memory-related errors.
        
        Args:
            error: The memory-related error
            
        Returns:
            True if recovery was attempted
        """
        if "out of memory" in str(error).lower():
            logger.info("Attempting memory recovery...")
            
            # Clear GPU cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")
            
            # Force garbage collection
            import gc
            gc.collect()
            logger.info("Forced garbage collection")
            
            return True
        
        return False
    
    def suggest_batch_size_reduction(self, current_batch_size: int, error: Exception) -> Optional[int]:
        """
        Suggest reduced batch size for memory errors.
        
        Args:
            current_batch_size: Current batch size
            error: The error that occurred
            
        Returns:
            Suggested new batch size or None
        """
        if "out of memory" in str(error).lower() and current_batch_size > 1:
            new_batch_size = max(1, current_batch_size // 2)
            logger.info(f"Suggesting batch size reduction: {current_batch_size} -> {new_batch_size}")
            return new_batch_size
        
        return None
    
    def get_recovery_suggestions(self, error: Exception, context: str = "") -> List[str]:
        """
        Get recovery suggestions for an error.
        
        Args:
            error: The error that occurred
            context: Context where error occurred
            
        Returns:
            List of recovery suggestions
        """
        suggestions = []
        error_str = str(error).lower()
        
        # Memory-related suggestions
        if "out of memory" in error_str:
            suggestions.extend([
                "Reduce batch size",
                "Use CPU instead of GPU",
                "Process data in smaller chunks",
                "Close other applications to free memory",
                "Restart the process to clear memory leaks"
            ])
        
        # File-related suggestions
        if "file not found" in error_str or "no such file" in error_str:
            suggestions.extend([
                "Check file path is correct",
                "Verify file exists and is accessible",
                "Check file permissions",
                "Use absolute path instead of relative path"
            ])
        
        # Model-related suggestions
        if "model" in error_str and ("load" in error_str or "invalid" in error_str):
            suggestions.extend([
                "Verify model file is not corrupted",
                "Check model architecture compatibility",
                "Try downloading model again",
                "Use a different model checkpoint"
            ])
        
        # TIFF-related suggestions
        if "tiff" in error_str or "corrupted" in error_str:
            suggestions.extend([
                "Verify TIFF file is not corrupted",
                "Check TIFF file format and dimensions",
                "Try opening file with image viewer first",
                "Convert file to standard TIFF format"
            ])
        
        # Configuration suggestions
        if "config" in error_str or "parameter" in error_str:
            suggestions.extend([
                "Check configuration parameters",
                "Verify all required parameters are set",
                "Use default configuration as starting point",
                "Check parameter value ranges"
            ])
        
        return suggestions


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> ErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def setup_global_error_handling(log_file: Optional[str] = None) -> ErrorHandler:
    """
    Setup global error handling.
    
    Args:
        log_file: Optional log file path
        
    Returns:
        Configured ErrorHandler instance
    """
    global _global_error_handler
    _global_error_handler = ErrorHandler(log_file)
    return _global_error_handler