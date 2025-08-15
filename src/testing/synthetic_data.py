"""
Synthetic test data generation for 3D image enhancement system.

This module provides utilities for generating synthetic 3D TIFF datasets
with controlled characteristics and degradation patterns for testing
and validation purposes.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.data_models import SliceStack
from ..core.tiff_handler import TIFFDataHandler

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic 3D image data with controlled characteristics.
    
    Creates realistic synthetic volumes with various patterns, textures,
    and structures suitable for testing enhancement algorithms.
    
    Features:
    - Multiple pattern types (geometric, organic, textured)
    - Configurable dimensions and data types
    - Realistic noise and artifact simulation
    - Ground truth generation for validation
    - Batch generation for large datasets
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize synthetic data generator.
        
        Args:
            random_seed: Optional random seed for reproducible generation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            if TORCH_AVAILABLE:
                torch.manual_seed(random_seed)
        
        self.tiff_handler = TIFFDataHandler()
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_geometric_volume(
        self,
        shape: Tuple[int, int, int] = (10, 1024, 1024),
        pattern_type: str = "spheres",
        num_objects: int = 5,
        intensity_range: Tuple[float, float] = (0.0, 1.0)
    ) -> SliceStack:
        """
        Generate volume with geometric patterns.
        
        Args:
            shape: Volume shape (slices, height, width)
            pattern_type: Type of geometric pattern ('spheres', 'cubes', 'cylinders')
            num_objects: Number of geometric objects
            intensity_range: Intensity value range
            
        Returns:
            SliceStack with geometric patterns
        """
        num_slices, height, width = shape
        volume = np.zeros(shape, dtype=np.float32)
        
        min_intensity, max_intensity = intensity_range
        
        for _ in range(num_objects):
            # Random object parameters
            center_z = np.random.randint(0, num_slices)
            center_y = np.random.randint(height // 4, 3 * height // 4)
            center_x = np.random.randint(width // 4, 3 * width // 4)
            
            max_size = max(20, min(height, width) // 8)
            size = np.random.randint(10, max_size)
            intensity = np.random.uniform(min_intensity, max_intensity)
            
            if pattern_type == "spheres":
                self._add_sphere(volume, (center_z, center_y, center_x), size, intensity)
            elif pattern_type == "cubes":
                self._add_cube(volume, (center_z, center_y, center_x), size, intensity)
            elif pattern_type == "cylinders":
                self._add_cylinder(volume, (center_z, center_y, center_x), size, intensity)
        
        # Add background texture
        background_noise = np.random.normal(0, 0.05, shape).astype(np.float32)
        volume = np.clip(volume + background_noise, 0, 1)
        
        metadata = {
            'pattern_type': pattern_type,
            'num_objects': num_objects,
            'intensity_range': intensity_range,
            'generation_type': 'geometric'
        }
        
        result = SliceStack(volume, metadata)
        logger.info(f"Generated geometric volume: {shape}, pattern: {pattern_type}")
        return result
    
    def _add_sphere(self, volume: np.ndarray, center: Tuple[int, int, int], radius: int, intensity: float):
        """Add a sphere to the volume."""
        cz, cy, cx = center
        nz, ny, nx = volume.shape
        
        # Create coordinate grids
        z_indices, y_indices, x_indices = np.mgrid[0:nz, 0:ny, 0:nx]
        
        # Calculate distances from center
        distances = np.sqrt(
            (z_indices - cz) ** 2 + 
            (y_indices - cy) ** 2 + 
            (x_indices - cx) ** 2
        )
        
        # Create sphere mask
        sphere_mask = distances <= radius
        
        # Add sphere with smooth edges
        smooth_factor = np.exp(-(distances - radius) ** 2 / (radius * 0.1) ** 2)
        smooth_factor = np.clip(smooth_factor, 0, 1)
        
        volume[sphere_mask] = np.maximum(volume[sphere_mask], intensity * smooth_factor[sphere_mask])
    
    def _add_cube(self, volume: np.ndarray, center: Tuple[int, int, int], size: int, intensity: float):
        """Add a cube to the volume."""
        cz, cy, cx = center
        half_size = size // 2
        
        z_start = max(0, cz - half_size)
        z_end = min(volume.shape[0], cz + half_size)
        y_start = max(0, cy - half_size)
        y_end = min(volume.shape[1], cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(volume.shape[2], cx + half_size)
        
        volume[z_start:z_end, y_start:y_end, x_start:x_end] = np.maximum(
            volume[z_start:z_end, y_start:y_end, x_start:x_end], intensity
        )
    
    def _add_cylinder(self, volume: np.ndarray, center: Tuple[int, int, int], radius: int, intensity: float):
        """Add a cylinder to the volume (along z-axis)."""
        cz, cy, cx = center
        nz, ny, nx = volume.shape
        
        # Create coordinate grids for y and x
        y_indices, x_indices = np.mgrid[0:ny, 0:nx]
        
        # Calculate radial distances from center
        radial_distances = np.sqrt((y_indices - cy) ** 2 + (x_indices - cx) ** 2)
        
        # Create cylinder mask
        cylinder_mask = radial_distances <= radius
        
        # Apply to all slices
        for z in range(nz):
            volume[z, cylinder_mask] = np.maximum(volume[z, cylinder_mask], intensity)
    
    def generate_textured_volume(
        self,
        shape: Tuple[int, int, int] = (10, 1024, 1024),
        texture_type: str = "perlin",
        scale: float = 0.1,
        octaves: int = 4
    ) -> SliceStack:
        """
        Generate volume with textured patterns.
        
        Args:
            shape: Volume shape (slices, height, width)
            texture_type: Type of texture ('perlin', 'fractal', 'cellular')
            scale: Texture scale factor
            octaves: Number of octaves for fractal textures
            
        Returns:
            SliceStack with textured patterns
        """
        num_slices, height, width = shape
        
        if texture_type == "perlin":
            volume = self._generate_perlin_noise(shape, scale, octaves)
        elif texture_type == "fractal":
            volume = self._generate_fractal_noise(shape, scale, octaves)
        elif texture_type == "cellular":
            volume = self._generate_cellular_pattern(shape, scale)
        else:
            # Default to random texture
            volume = np.random.rand(*shape).astype(np.float32)
        
        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        metadata = {
            'texture_type': texture_type,
            'scale': scale,
            'octaves': octaves,
            'generation_type': 'textured'
        }
        
        result = SliceStack(volume, metadata)
        logger.info(f"Generated textured volume: {shape}, texture: {texture_type}")
        return result
    
    def _generate_perlin_noise(self, shape: Tuple[int, int, int], scale: float, octaves: int) -> np.ndarray:
        """Generate Perlin-like noise (simplified implementation)."""
        volume = np.zeros(shape, dtype=np.float32)
        
        for octave in range(octaves):
            octave_scale = scale * (2 ** octave)
            octave_amplitude = 1.0 / (2 ** octave)
            
            # Generate noise for this octave
            noise = np.random.rand(*shape) * octave_amplitude
            
            # Apply smoothing (simple box filter)
            kernel_size = max(1, int(1.0 / octave_scale))
            if kernel_size > 1:
                from scipy import ndimage
                noise = ndimage.uniform_filter(noise, size=kernel_size)
            
            volume += noise
        
        return volume
    
    def _generate_fractal_noise(self, shape: Tuple[int, int, int], scale: float, octaves: int) -> np.ndarray:
        """Generate fractal noise pattern."""
        volume = np.zeros(shape, dtype=np.float32)
        
        # Create coordinate grids
        z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        
        for octave in range(octaves):
            frequency = scale * (2 ** octave)
            amplitude = 1.0 / (2 ** octave)
            
            # Simple fractal pattern
            pattern = np.sin(z * frequency) * np.cos(y * frequency) * np.sin(x * frequency)
            volume += pattern * amplitude
        
        return volume
    
    def _generate_cellular_pattern(self, shape: Tuple[int, int, int], scale: float) -> np.ndarray:
        """Generate cellular automata-like pattern."""
        volume = np.random.rand(*shape) > 0.5
        
        # Apply cellular automata rules (simplified)
        for _ in range(3):  # 3 iterations
            new_volume = volume.copy()
            
            # Simple neighbor counting (3D)
            for z in range(1, shape[0] - 1):
                for y in range(1, shape[1] - 1):
                    for x in range(1, shape[2] - 1):
                        neighbors = volume[z-1:z+2, y-1:y+2, x-1:x+2].sum() - volume[z, y, x]
                        
                        if volume[z, y, x]:
                            new_volume[z, y, x] = neighbors >= 4
                        else:
                            new_volume[z, y, x] = neighbors >= 5
            
            volume = new_volume
        
        return volume.astype(np.float32)
    
    def generate_realistic_volume(
        self,
        shape: Tuple[int, int, int] = (10, 1024, 1024),
        structure_density: float = 0.3,
        background_intensity: float = 0.1
    ) -> SliceStack:
        """
        Generate realistic-looking volume combining multiple patterns.
        
        Args:
            shape: Volume shape (slices, height, width)
            structure_density: Density of structures (0-1)
            background_intensity: Background intensity level
            
        Returns:
            SliceStack with realistic patterns
        """
        # Start with textured background
        background = self.generate_textured_volume(shape, "perlin", 0.05, 3)
        volume = background.to_numpy() * background_intensity
        
        # Add geometric structures
        num_structures = int(structure_density * 20)
        for _ in range(num_structures):
            structure_type = random.choice(["spheres", "cylinders"])
            structures = self.generate_geometric_volume(
                shape, structure_type, 1, (0.3, 0.8)
            )
            volume = np.maximum(volume, structures.to_numpy())
        
        # Add fine texture overlay
        fine_texture = self.generate_textured_volume(shape, "fractal", 0.2, 2)
        volume += fine_texture.to_numpy() * 0.1
        
        # Normalize and clip
        volume = np.clip(volume, 0, 1)
        
        metadata = {
            'structure_density': structure_density,
            'background_intensity': background_intensity,
            'generation_type': 'realistic'
        }
        
        result = SliceStack(volume, metadata)
        logger.info(f"Generated realistic volume: {shape}")
        return result
    
    def save_volume(self, volume: SliceStack, output_path: str) -> None:
        """
        Save generated volume to TIFF file.
        
        Args:
            volume: Volume to save
            output_path: Output file path
        """
        self.tiff_handler.save_3d_tiff(volume.to_numpy(), output_path)
        logger.info(f"Saved synthetic volume: {output_path}")


class DegradationSimulator:
    """
    Simulate various image degradation patterns for testing enhancement algorithms.
    
    Applies realistic degradation effects to high-quality synthetic data
    to create input/target pairs for training and validation.
    
    Features:
    - Multiple degradation types (noise, blur, compression artifacts)
    - Configurable degradation severity
    - Realistic artifact simulation
    - Batch processing capabilities
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize degradation simulator.
        
        Args:
            random_seed: Optional random seed for reproducible degradation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        logger.info("DegradationSimulator initialized")
    
    def add_gaussian_noise(
        self,
        volume: SliceStack,
        noise_std: float = 0.1,
        preserve_range: bool = True
    ) -> SliceStack:
        """
        Add Gaussian noise to volume.
        
        Args:
            volume: Input volume
            noise_std: Standard deviation of noise
            preserve_range: Whether to preserve original value range
            
        Returns:
            Degraded volume with noise
        """
        data = volume.to_numpy()
        noise = np.random.normal(0, noise_std, data.shape).astype(data.dtype)
        degraded_data = data + noise
        
        if preserve_range:
            degraded_data = np.clip(degraded_data, data.min(), data.max())
        
        metadata = volume.metadata.copy()
        metadata['degradation_noise_std'] = noise_std
        
        result = SliceStack(degraded_data, metadata)
        logger.debug(f"Added Gaussian noise: std={noise_std}")
        return result
    
    def add_poisson_noise(
        self,
        volume: SliceStack,
        scale_factor: float = 100.0
    ) -> SliceStack:
        """
        Add Poisson noise to volume (simulates photon noise).
        
        Args:
            volume: Input volume
            scale_factor: Scale factor for Poisson distribution
            
        Returns:
            Degraded volume with Poisson noise
        """
        data = volume.to_numpy()
        
        # Scale up, add Poisson noise, scale back down
        scaled_data = data * scale_factor
        noisy_data = np.random.poisson(scaled_data).astype(np.float32)
        degraded_data = noisy_data / scale_factor
        
        # Preserve original range
        degraded_data = np.clip(degraded_data, data.min(), data.max())
        
        metadata = volume.metadata.copy()
        metadata['degradation_poisson_scale'] = scale_factor
        
        result = SliceStack(degraded_data, metadata)
        logger.debug(f"Added Poisson noise: scale={scale_factor}")
        return result
    
    def add_blur(
        self,
        volume: SliceStack,
        blur_sigma: float = 1.0,
        blur_type: str = "gaussian"
    ) -> SliceStack:
        """
        Add blur to volume.
        
        Args:
            volume: Input volume
            blur_sigma: Blur kernel standard deviation
            blur_type: Type of blur ('gaussian', 'motion', 'defocus')
            
        Returns:
            Blurred volume
        """
        data = volume.to_numpy()
        
        try:
            from scipy import ndimage
            
            if blur_type == "gaussian":
                # Apply Gaussian blur to each slice
                degraded_data = np.zeros_like(data)
                for i in range(data.shape[0]):
                    degraded_data[i] = ndimage.gaussian_filter(data[i], sigma=blur_sigma)
            
            elif blur_type == "motion":
                # Simulate motion blur with directional kernel
                kernel_size = int(blur_sigma * 6)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[kernel_size // 2, :] = 1.0
                kernel = kernel / kernel.sum()
                
                degraded_data = np.zeros_like(data)
                for i in range(data.shape[0]):
                    degraded_data[i] = ndimage.convolve(data[i], kernel)
            
            else:  # defocus
                # Simple defocus blur
                degraded_data = np.zeros_like(data)
                for i in range(data.shape[0]):
                    degraded_data[i] = ndimage.uniform_filter(data[i], size=int(blur_sigma * 2))
        
        except ImportError:
            logger.warning("scipy not available, using simple blur approximation")
            # Simple approximation without scipy
            degraded_data = data.copy()
            for i in range(int(blur_sigma)):
                degraded_data = 0.5 * degraded_data + 0.25 * np.roll(degraded_data, 1, axis=1) + 0.25 * np.roll(degraded_data, -1, axis=1)
        
        metadata = volume.metadata.copy()
        metadata['degradation_blur_sigma'] = blur_sigma
        metadata['degradation_blur_type'] = blur_type
        
        result = SliceStack(degraded_data, metadata)
        logger.debug(f"Added {blur_type} blur: sigma={blur_sigma}")
        return result
    
    def add_compression_artifacts(
        self,
        volume: SliceStack,
        quality_factor: float = 0.5,
        block_size: int = 8
    ) -> SliceStack:
        """
        Simulate compression artifacts.
        
        Args:
            volume: Input volume
            quality_factor: Quality factor (0-1, lower = more artifacts)
            block_size: Block size for artifacts
            
        Returns:
            Volume with compression artifacts
        """
        data = volume.to_numpy()
        degraded_data = data.copy()
        
        # Simple block-based quantization to simulate compression
        for i in range(data.shape[0]):
            slice_data = data[i]
            
            # Divide into blocks and quantize
            h, w = slice_data.shape
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block = slice_data[y:y+block_size, x:x+block_size]
                    
                    # Quantize block based on quality factor
                    quantization_step = (1.0 - quality_factor) * 0.1
                    if quantization_step > 0:
                        quantized_block = np.round(block / quantization_step) * quantization_step
                        degraded_data[i, y:y+block_size, x:x+block_size] = quantized_block
        
        metadata = volume.metadata.copy()
        metadata['degradation_quality_factor'] = quality_factor
        metadata['degradation_block_size'] = block_size
        
        result = SliceStack(degraded_data, metadata)
        logger.debug(f"Added compression artifacts: quality={quality_factor}")
        return result
    
    def add_mixed_degradation(
        self,
        volume: SliceStack,
        degradation_config: Dict[str, Any]
    ) -> SliceStack:
        """
        Apply multiple degradation effects.
        
        Args:
            volume: Input volume
            degradation_config: Configuration for degradation effects
            
        Returns:
            Volume with mixed degradation
        """
        degraded_volume = volume
        
        # Apply degradations in sequence
        if degradation_config.get('gaussian_noise', 0) > 0:
            degraded_volume = self.add_gaussian_noise(
                degraded_volume, degradation_config['gaussian_noise']
            )
        
        if degradation_config.get('poisson_noise', 0) > 0:
            degraded_volume = self.add_poisson_noise(
                degraded_volume, degradation_config['poisson_noise']
            )
        
        if degradation_config.get('blur_sigma', 0) > 0:
            blur_type = degradation_config.get('blur_type', 'gaussian')
            degraded_volume = self.add_blur(
                degraded_volume, degradation_config['blur_sigma'], blur_type
            )
        
        if degradation_config.get('compression_quality', 1.0) < 1.0:
            degraded_volume = self.add_compression_artifacts(
                degraded_volume, degradation_config['compression_quality']
            )
        
        # Store degradation config in metadata
        metadata = degraded_volume.metadata.copy()
        metadata['degradation_config'] = degradation_config
        degraded_volume.metadata = metadata
        
        logger.info(f"Applied mixed degradation: {list(degradation_config.keys())}")
        return degraded_volume
    
    def create_training_pair(
        self,
        clean_volume: SliceStack,
        degradation_config: Dict[str, Any]
    ) -> Tuple[SliceStack, SliceStack]:
        """
        Create training pair from clean volume.
        
        Args:
            clean_volume: Clean (target) volume
            degradation_config: Degradation configuration
            
        Returns:
            Tuple of (degraded_input, clean_target)
        """
        degraded_input = self.add_mixed_degradation(clean_volume, degradation_config)
        clean_target = clean_volume
        
        logger.debug("Created training pair with degradation")
        return degraded_input, clean_target