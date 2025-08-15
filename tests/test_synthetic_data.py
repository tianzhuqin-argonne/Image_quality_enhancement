"""
Unit tests for synthetic data generation and test fixtures.
"""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.testing.synthetic_data import SyntheticDataGenerator, DegradationSimulator
from src.testing.test_fixtures import TestDataFixtures
from src.core.data_models import SliceStack


class TestSyntheticDataGenerator:
    """Test synthetic data generation functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.generator = SyntheticDataGenerator(random_seed=42)
    
    def test_generator_init(self):
        """Test generator initialization."""
        assert self.generator.tiff_handler is not None
    
    def test_generate_geometric_volume_spheres(self):
        """Test generating volume with spheres."""
        volume = self.generator.generate_geometric_volume(
            shape=(5, 128, 128),
            pattern_type="spheres",
            num_objects=3
        )
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (5, 128, 128)
        assert volume.metadata['pattern_type'] == 'spheres'
        assert volume.metadata['num_objects'] == 3
        
        # Check that volume contains non-zero values
        data = volume.to_numpy()
        assert data.max() > 0
    
    def test_generate_geometric_volume_cubes(self):
        """Test generating volume with cubes."""
        volume = self.generator.generate_geometric_volume(
            shape=(3, 64, 64),
            pattern_type="cubes",
            num_objects=2
        )
        
        assert volume.shape == (3, 64, 64)
        assert volume.metadata['pattern_type'] == 'cubes'
    
    def test_generate_geometric_volume_cylinders(self):
        """Test generating volume with cylinders."""
        volume = self.generator.generate_geometric_volume(
            shape=(4, 96, 96),
            pattern_type="cylinders",
            num_objects=2
        )
        
        assert volume.shape == (4, 96, 96)
        assert volume.metadata['pattern_type'] == 'cylinders'
    
    def test_generate_textured_volume_perlin(self):
        """Test generating textured volume with Perlin noise."""
        volume = self.generator.generate_textured_volume(
            shape=(3, 64, 64),
            texture_type="perlin",
            scale=0.1,
            octaves=3
        )
        
        assert volume.shape == (3, 64, 64)
        assert volume.metadata['texture_type'] == 'perlin'
        assert volume.metadata['scale'] == 0.1
        assert volume.metadata['octaves'] == 3
        
        # Check normalization
        data = volume.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 1
    
    def test_generate_textured_volume_fractal(self):
        """Test generating textured volume with fractal noise."""
        volume = self.generator.generate_textured_volume(
            shape=(2, 32, 32),
            texture_type="fractal"
        )
        
        assert volume.shape == (2, 32, 32)
        assert volume.metadata['texture_type'] == 'fractal'
    
    def test_generate_textured_volume_cellular(self):
        """Test generating textured volume with cellular pattern."""
        volume = self.generator.generate_textured_volume(
            shape=(2, 32, 32),
            texture_type="cellular"
        )
        
        assert volume.shape == (2, 32, 32)
        assert volume.metadata['texture_type'] == 'cellular'
    
    def test_generate_realistic_volume(self):
        """Test generating realistic volume."""
        volume = self.generator.generate_realistic_volume(
            shape=(3, 64, 64),
            structure_density=0.3,
            background_intensity=0.1
        )
        
        assert volume.shape == (3, 64, 64)
        assert volume.metadata['structure_density'] == 0.3
        assert volume.metadata['background_intensity'] == 0.1
        assert volume.metadata['generation_type'] == 'realistic'
        
        # Check value range
        data = volume.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 1
    
    def test_save_volume(self):
        """Test saving generated volume."""
        with tempfile.TemporaryDirectory() as temp_dir:
            volume = self.generator.generate_geometric_volume(shape=(2, 32, 32))
            output_path = Path(temp_dir) / "test_volume.tif"
            
            self.generator.save_volume(volume, str(output_path))
            
            assert output_path.exists()


class TestDegradationSimulator:
    """Test degradation simulation functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.simulator = DegradationSimulator(random_seed=42)
        
        # Create test volume
        data = np.random.rand(3, 64, 64).astype(np.float32)
        self.test_volume = SliceStack(data)
    
    def test_simulator_init(self):
        """Test simulator initialization."""
        assert self.simulator is not None
    
    def test_add_gaussian_noise(self):
        """Test adding Gaussian noise."""
        degraded = self.simulator.add_gaussian_noise(
            self.test_volume, noise_std=0.1
        )
        
        assert isinstance(degraded, SliceStack)
        assert degraded.shape == self.test_volume.shape
        assert 'degradation_noise_std' in degraded.metadata
        assert degraded.metadata['degradation_noise_std'] == 0.1
        
        # Check that noise was added (should be different)
        original_data = self.test_volume.to_numpy()
        degraded_data = degraded.to_numpy()
        assert not np.array_equal(original_data, degraded_data)
    
    def test_add_poisson_noise(self):
        """Test adding Poisson noise."""
        degraded = self.simulator.add_poisson_noise(
            self.test_volume, scale_factor=100.0
        )
        
        assert isinstance(degraded, SliceStack)
        assert degraded.shape == self.test_volume.shape
        assert 'degradation_poisson_scale' in degraded.metadata
        
        # Check that noise was added
        original_data = self.test_volume.to_numpy()
        degraded_data = degraded.to_numpy()
        assert not np.array_equal(original_data, degraded_data)
    
    def test_add_blur_gaussian(self):
        """Test adding Gaussian blur."""
        degraded = self.simulator.add_blur(
            self.test_volume, blur_sigma=1.0, blur_type="gaussian"
        )
        
        assert isinstance(degraded, SliceStack)
        assert degraded.shape == self.test_volume.shape
        assert 'degradation_blur_sigma' in degraded.metadata
        assert degraded.metadata['degradation_blur_type'] == 'gaussian'
    
    def test_add_blur_motion(self):
        """Test adding motion blur."""
        degraded = self.simulator.add_blur(
            self.test_volume, blur_sigma=1.0, blur_type="motion"
        )
        
        assert isinstance(degraded, SliceStack)
        assert degraded.metadata['degradation_blur_type'] == 'motion'
    
    def test_add_compression_artifacts(self):
        """Test adding compression artifacts."""
        degraded = self.simulator.add_compression_artifacts(
            self.test_volume, quality_factor=0.5, block_size=8
        )
        
        assert isinstance(degraded, SliceStack)
        assert degraded.shape == self.test_volume.shape
        assert 'degradation_quality_factor' in degraded.metadata
        assert degraded.metadata['degradation_quality_factor'] == 0.5
    
    def test_add_mixed_degradation(self):
        """Test adding mixed degradation effects."""
        config = {
            'gaussian_noise': 0.1,
            'blur_sigma': 1.0,
            'compression_quality': 0.7
        }
        
        degraded = self.simulator.add_mixed_degradation(self.test_volume, config)
        
        assert isinstance(degraded, SliceStack)
        assert 'degradation_config' in degraded.metadata
        assert degraded.metadata['degradation_config'] == config
        
        # Should have multiple degradation metadata entries
        assert 'degradation_noise_std' in degraded.metadata
        assert 'degradation_blur_sigma' in degraded.metadata
        assert 'degradation_quality_factor' in degraded.metadata
    
    def test_create_training_pair(self):
        """Test creating training pair."""
        config = {
            'gaussian_noise': 0.05,
            'blur_sigma': 0.5
        }
        
        degraded_input, clean_target = self.simulator.create_training_pair(
            self.test_volume, config
        )
        
        assert isinstance(degraded_input, SliceStack)
        assert isinstance(clean_target, SliceStack)
        assert degraded_input.shape == clean_target.shape
        assert clean_target is self.test_volume  # Should be same reference
        
        # Degraded should be different from clean
        degraded_data = degraded_input.to_numpy()
        clean_data = clean_target.to_numpy()
        assert not np.array_equal(degraded_data, clean_data)


class TestTestDataFixtures:
    """Test test data fixtures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.fixtures = TestDataFixtures()
    
    def teardown_method(self):
        """Cleanup test environment."""
        self.fixtures.cleanup()
    
    def test_fixtures_init(self):
        """Test fixtures initialization."""
        assert Path(self.fixtures.temp_dir).exists()
        assert self.fixtures.data_generator is not None
        assert self.fixtures.degradation_simulator is not None
    
    def test_get_small_test_volume(self):
        """Test getting small test volume."""
        volume = self.fixtures.get_small_test_volume()
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (5, 256, 256)
        
        # Should return same instance when called again (cached)
        volume2 = self.fixtures.get_small_test_volume()
        assert volume is volume2
    
    def test_get_medium_test_volume(self):
        """Test getting medium test volume."""
        volume = self.fixtures.get_medium_test_volume()
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (10, 512, 512)
    
    def test_get_large_test_volume(self):
        """Test getting large test volume."""
        volume = self.fixtures.get_large_test_volume()
        
        assert isinstance(volume, SliceStack)
        assert volume.shape == (20, 1024, 1024)
    
    def test_get_training_pair(self):
        """Test getting training pair."""
        degraded_input, clean_target = self.fixtures.get_training_pair(
            volume_size="small", degradation_level="moderate"
        )
        
        assert isinstance(degraded_input, SliceStack)
        assert isinstance(clean_target, SliceStack)
        assert degraded_input.shape == clean_target.shape
        
        # Should have degradation metadata
        assert 'degradation_config' in degraded_input.metadata
    
    def test_save_test_tiff(self):
        """Test saving test TIFF file."""
        volume = self.fixtures.get_small_test_volume()
        file_path = self.fixtures.save_test_tiff(volume, "test_volume")
        
        assert Path(file_path).exists()
        assert file_path.endswith(".tif")
    
    def test_create_test_dataset(self):
        """Test creating test dataset."""
        pairs = self.fixtures.create_test_dataset(
            num_pairs=3, volume_size="small", degradation_level="light"
        )
        
        assert len(pairs) == 3
        
        for input_path, target_path in pairs:
            assert Path(input_path).exists()
            assert Path(target_path).exists()
            assert "input_" in input_path
            assert "target_" in target_path
    
    def test_get_training_config(self):
        """Test getting training configuration."""
        config = self.fixtures.get_training_config("fast")
        
        assert config.epochs == 2
        assert config.batch_size == 2
        assert config.device == 'cpu'
    
    def test_get_inference_config(self):
        """Test getting inference configuration."""
        config = self.fixtures.get_inference_config("quality")
        
        assert config.patch_size == (512, 512)
        assert config.overlap == 64
        assert config.device == 'cpu'
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_create_mock_unet_model(self):
        """Test creating mock U-Net model."""
        model = TestDataFixtures.create_mock_unet_model()
        
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        test_input = torch.randn(1, 1, 64, 64)
        output = model(test_input)
        
        assert output.shape == test_input.shape
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_mock_model(self):
        """Test saving mock model."""
        model_path = self.fixtures.save_mock_model("test_model.pth")
        
        assert Path(model_path).exists()
        
        # Test loading
        checkpoint = torch.load(model_path)
        assert 'model_state_dict' in checkpoint
        assert 'model_config' in checkpoint
    
    def test_get_benchmark_data(self):
        """Test getting benchmark data."""
        benchmark_data = self.fixtures.get_benchmark_data()
        
        assert 'small_dataset' in benchmark_data
        assert 'medium_dataset' in benchmark_data
        assert 'large_dataset' in benchmark_data
        
        for dataset_name, data in benchmark_data.items():
            assert 'shape' in data
            assert 'expected_processing_time' in data
            assert 'expected_memory_mb' in data
            assert 'min_psnr' in data
    
    def test_validate_enhancement_quality(self):
        """Test enhancement quality validation."""
        original = self.fixtures.get_small_test_volume()
        
        # Create slightly modified version as "enhanced"
        enhanced_data = original.to_numpy() + np.random.normal(0, 0.01, original.shape)
        enhanced = SliceStack(enhanced_data.astype(np.float32))
        
        results = self.fixtures.validate_enhancement_quality(original, enhanced)
        
        assert 'shape_match' in results
        assert 'psnr' in results
        assert 'quality_acceptable' in results
        assert results['shape_match'] is True
    
    def test_get_temp_file_path(self):
        """Test getting temporary file path."""
        file_path = self.fixtures.get_temp_file_path("test.txt", "subdir")
        
        assert "test.txt" in file_path
        assert "subdir" in file_path
        assert Path(file_path).parent.exists()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        temp_dir = None
        
        with TestDataFixtures() as fixtures:
            temp_dir = fixtures.temp_dir
            assert Path(temp_dir).exists()
        
        # Should be cleaned up after context exit
        assert not Path(temp_dir).exists()


# Test without PyTorch
@pytest.mark.skipif(TORCH_AVAILABLE, reason="Testing PyTorch unavailable case")
def test_fixtures_without_torch():
    """Test fixtures behavior when PyTorch is not available."""
    fixtures = TestDataFixtures()
    
    try:
        # Should work without PyTorch
        volume = fixtures.get_small_test_volume()
        assert isinstance(volume, SliceStack)
        
        # Mock model creation should fail
        with pytest.raises(ImportError):
            TestDataFixtures.create_mock_unet_model()
        
        # Model saving should fail
        with pytest.raises(ImportError):
            fixtures.save_mock_model()
    
    finally:
        fixtures.cleanup()