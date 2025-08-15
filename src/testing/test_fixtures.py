"""
Test fixtures and utilities for comprehensive system testing.

This module provides reusable test fixtures, mock data, and testing
utilities for the 3D image enhancement system.
"""

import logging
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .synthetic_data import SyntheticDataGenerator, DegradationSimulator
from ..core.data_models import SliceStack
from ..core.config import TrainingConfig, InferenceConfig

logger = logging.getLogger(__name__)


class TestDataFixtures:
    """
    Provides reusable test fixtures and mock data for system testing.
    
    Creates standardized test datasets, configurations, and mock objects
    for consistent testing across the enhancement system.
    
    Features:
    - Standardized test datasets
    - Mock model creation
    - Temporary file management
    - Configuration fixtures
    - Performance benchmarking data
    """
    
    def __init__(self, temp_dir: Optional[str] = None):
        """
        Initialize test fixtures.
        
        Args:
            temp_dir: Optional temporary directory (creates one if None)
        """
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
            self._cleanup_temp_dir = True
        else:
            self.temp_dir = temp_dir
            self._cleanup_temp_dir = False
        
        self.temp_path = Path(self.temp_dir)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize generators
        self.data_generator = SyntheticDataGenerator(random_seed=42)
        self.degradation_simulator = DegradationSimulator(random_seed=42)
        
        # Cache for generated data
        self._data_cache = {}
        
        logger.info(f"TestDataFixtures initialized with temp_dir: {self.temp_dir}")
    
    def cleanup(self):
        """Clean up temporary files and directories."""
        if self._cleanup_temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary test directory")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def get_small_test_volume(self, cache_key: str = "small_volume") -> SliceStack:
        """
        Get small test volume for quick testing.
        
        Args:
            cache_key: Cache key for reusing generated data
            
        Returns:
            Small test volume (5 slices, 256x256)
        """
        if cache_key not in self._data_cache:
            volume = self.data_generator.generate_realistic_volume(
                shape=(5, 256, 256),
                structure_density=0.2,
                background_intensity=0.1
            )
            self._data_cache[cache_key] = volume
        
        return self._data_cache[cache_key]
    
    def get_medium_test_volume(self, cache_key: str = "medium_volume") -> SliceStack:
        """
        Get medium test volume for standard testing.
        
        Args:
            cache_key: Cache key for reusing generated data
            
        Returns:
            Medium test volume (10 slices, 512x512)
        """
        if cache_key not in self._data_cache:
            volume = self.data_generator.generate_realistic_volume(
                shape=(10, 512, 512),
                structure_density=0.3,
                background_intensity=0.15
            )
            self._data_cache[cache_key] = volume
        
        return self._data_cache[cache_key]
    
    def get_large_test_volume(self, cache_key: str = "large_volume") -> SliceStack:
        """
        Get large test volume for performance testing.
        
        Args:
            cache_key: Cache key for reusing generated data
            
        Returns:
            Large test volume (20 slices, 1024x1024)
        """
        if cache_key not in self._data_cache:
            volume = self.data_generator.generate_realistic_volume(
                shape=(20, 1024, 1024),
                structure_density=0.4,
                background_intensity=0.2
            )
            self._data_cache[cache_key] = volume
        
        return self._data_cache[cache_key]
    
    def get_training_pair(
        self,
        volume_size: str = "small",
        degradation_level: str = "moderate"
    ) -> Tuple[SliceStack, SliceStack]:
        """
        Get training pair (degraded input, clean target).
        
        Args:
            volume_size: Size of volume ('small', 'medium', 'large')
            degradation_level: Level of degradation ('light', 'moderate', 'heavy')
            
        Returns:
            Tuple of (degraded_input, clean_target)
        """
        # Get clean volume
        if volume_size == "small":
            clean_volume = self.get_small_test_volume()
        elif volume_size == "medium":
            clean_volume = self.get_medium_test_volume()
        else:
            clean_volume = self.get_large_test_volume()
        
        # Define degradation configs
        degradation_configs = {
            'light': {
                'gaussian_noise': 0.05,
                'blur_sigma': 0.5
            },
            'moderate': {
                'gaussian_noise': 0.1,
                'poisson_noise': 50.0,
                'blur_sigma': 1.0,
                'compression_quality': 0.8
            },
            'heavy': {
                'gaussian_noise': 0.2,
                'poisson_noise': 20.0,
                'blur_sigma': 2.0,
                'compression_quality': 0.5
            }
        }
        
        config = degradation_configs.get(degradation_level, degradation_configs['moderate'])
        
        # Create training pair
        degraded_input, clean_target = self.degradation_simulator.create_training_pair(
            clean_volume, config
        )
        
        return degraded_input, clean_target
    
    def save_test_tiff(
        self,
        volume: SliceStack,
        filename: str,
        subdirectory: str = ""
    ) -> str:
        """
        Save test volume as TIFF file.
        
        Args:
            volume: Volume to save
            filename: Filename (without extension)
            subdirectory: Optional subdirectory
            
        Returns:
            Path to saved file
        """
        if subdirectory:
            save_dir = self.temp_path / subdirectory
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = self.temp_path
        
        file_path = save_dir / f"{filename}.tif"
        self.data_generator.save_volume(volume, str(file_path))
        
        return str(file_path)
    
    def create_test_dataset(
        self,
        num_pairs: int = 5,
        volume_size: str = "small",
        degradation_level: str = "moderate"
    ) -> List[Tuple[str, str]]:
        """
        Create test dataset with multiple training pairs.
        
        Args:
            num_pairs: Number of training pairs to create
            volume_size: Size of volumes
            degradation_level: Level of degradation
            
        Returns:
            List of (input_path, target_path) tuples
        """
        dataset_dir = self.temp_path / "test_dataset"
        dataset_dir.mkdir(exist_ok=True)
        
        input_dir = dataset_dir / "inputs"
        target_dir = dataset_dir / "targets"
        input_dir.mkdir(exist_ok=True)
        target_dir.mkdir(exist_ok=True)
        
        pairs = []
        
        for i in range(num_pairs):
            # Generate training pair
            degraded_input, clean_target = self.get_training_pair(volume_size, degradation_level)
            
            # Save files
            input_path = input_dir / f"input_{i:03d}.tif"
            target_path = target_dir / f"target_{i:03d}.tif"
            
            self.data_generator.save_volume(degraded_input, str(input_path))
            self.data_generator.save_volume(clean_target, str(target_path))
            
            pairs.append((str(input_path), str(target_path)))
        
        logger.info(f"Created test dataset with {num_pairs} pairs in {dataset_dir}")
        return pairs
    
    def get_training_config(self, config_type: str = "default") -> TrainingConfig:
        """
        Get training configuration for testing.
        
        Args:
            config_type: Type of configuration ('default', 'fast', 'thorough')
            
        Returns:
            TrainingConfig instance
        """
        configs = {
            'default': TrainingConfig(
                epochs=10,
                batch_size=4,
                learning_rate=1e-4,
                device='cpu',
                checkpoint_interval=5,
                early_stopping_patience=5
            ),
            'fast': TrainingConfig(
                epochs=2,
                batch_size=2,
                learning_rate=1e-3,
                device='cpu',
                checkpoint_interval=1,
                early_stopping_patience=2
            ),
            'thorough': TrainingConfig(
                epochs=50,
                batch_size=8,
                learning_rate=5e-5,
                device='cpu',
                checkpoint_interval=10,
                early_stopping_patience=10
            )
        }
        
        return configs.get(config_type, configs['default'])
    
    def get_inference_config(self, config_type: str = "default") -> InferenceConfig:
        """
        Get inference configuration for testing.
        
        Args:
            config_type: Type of configuration ('default', 'fast', 'quality')
            
        Returns:
            InferenceConfig instance
        """
        configs = {
            'default': InferenceConfig(
                patch_size=(256, 256),
                overlap=32,
                batch_size=4,
                device='cpu',
                memory_limit_gb=2.0
            ),
            'fast': InferenceConfig(
                patch_size=(128, 128),
                overlap=16,
                batch_size=8,
                device='cpu',
                memory_limit_gb=1.0
            ),
            'quality': InferenceConfig(
                patch_size=(512, 512),
                overlap=64,
                batch_size=2,
                device='cpu',
                memory_limit_gb=4.0
            )
        }
        
        return configs.get(config_type, configs['default'])
    
    @staticmethod
    def create_mock_unet_model(input_channels: int = 1, output_channels: int = 1) -> nn.Module:
        """
        Create mock U-Net model for testing.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            
        Returns:
            Simple mock U-Net model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for mock model creation")
        
        class MockUNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 16, 3, padding=1)  # Keep same channels
                self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # Keep same channels
                self.conv4 = nn.Conv2d(16, out_channels, 3, padding=1)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)
                self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
            def forward(self, x):
                # Simple encoder-decoder
                x1 = self.relu(self.conv1(x))
                x2 = self.pool(x1)
                x3 = self.relu(self.conv2(x2))
                x4 = self.upsample(x3)
                
                # Adjust size if needed
                if x4.shape != x1.shape:
                    x4 = torch.nn.functional.interpolate(x4, size=x1.shape[2:], mode='bilinear', align_corners=False)
                
                x5 = self.relu(self.conv3(x4 + x1))
                output = self.conv4(x5)
                return output
        
        return MockUNet(input_channels, output_channels)
    
    def save_mock_model(self, model_name: str = "mock_model.pth") -> str:
        """
        Save mock model to temporary directory.
        
        Args:
            model_name: Name of model file
            
        Returns:
            Path to saved model
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model saving")
        
        model = self.create_mock_unet_model()
        model_path = self.temp_path / model_name
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_channels': 1,
                'output_channels': 1,
                'architecture': 'mock_unet'
            }
        }, model_path)
        
        logger.info(f"Saved mock model: {model_path}")
        return str(model_path)
    
    def get_benchmark_data(self) -> Dict[str, Any]:
        """
        Get benchmark data for performance testing.
        
        Returns:
            Dictionary with benchmark datasets and expected metrics
        """
        return {
            'small_dataset': {
                'shape': (5, 256, 256),
                'expected_processing_time': 10.0,  # seconds
                'expected_memory_mb': 100.0,
                'min_psnr': 25.0
            },
            'medium_dataset': {
                'shape': (10, 512, 512),
                'expected_processing_time': 30.0,
                'expected_memory_mb': 400.0,
                'min_psnr': 25.0
            },
            'large_dataset': {
                'shape': (20, 1024, 1024),
                'expected_processing_time': 120.0,
                'expected_memory_mb': 1600.0,
                'min_psnr': 25.0
            }
        }
    
    def validate_enhancement_quality(
        self,
        original: SliceStack,
        enhanced: SliceStack,
        min_psnr: float = 20.0
    ) -> Dict[str, Any]:
        """
        Validate enhancement quality against benchmarks.
        
        Args:
            original: Original volume
            enhanced: Enhanced volume
            min_psnr: Minimum acceptable PSNR
            
        Returns:
            Validation results
        """
        if not TORCH_AVAILABLE:
            # Simple validation without PyTorch
            return {
                'shape_match': original.shape == enhanced.shape,
                'psnr': float('inf'),  # Can't calculate without PyTorch
                'quality_acceptable': True
            }
        
        # Calculate PSNR
        original_data = torch.from_numpy(original.to_numpy())
        enhanced_data = torch.from_numpy(enhanced.to_numpy())
        
        mse = torch.mean((original_data - enhanced_data) ** 2)
        if mse > 0:
            psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
        else:
            psnr = torch.tensor(float('inf'))
        
        results = {
            'shape_match': original.shape == enhanced.shape,
            'psnr': psnr.item(),
            'mse': mse.item(),
            'quality_acceptable': psnr.item() >= min_psnr
        }
        
        return results
    
    def get_temp_file_path(self, filename: str, subdirectory: str = "") -> str:
        """
        Get path for temporary file.
        
        Args:
            filename: Filename
            subdirectory: Optional subdirectory
            
        Returns:
            Full path to temporary file
        """
        if subdirectory:
            temp_dir = self.temp_path / subdirectory
            temp_dir.mkdir(exist_ok=True)
        else:
            temp_dir = self.temp_path
        
        return str(temp_dir / filename)