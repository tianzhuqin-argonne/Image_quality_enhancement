#!/usr/bin/env python3
"""
Basic inference example for 3D image enhancement.

This example demonstrates how to use a trained model to enhance 3D TIFF images
using the programmatic API.
"""

import logging
from pathlib import Path
import tempfile
import shutil
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.testing.test_fixtures import TestDataFixtures
    from src.inference.api import ImageEnhancementAPI
    from src.core.config import InferenceConfig
    from src.core.data_models import SliceStack
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


def main():
    """Run basic inference example."""
    logger.info("Starting basic inference example")
    
    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Using temporary directory: {temp_dir}")
    
    try:
        # Initialize test fixtures
        with TestDataFixtures(temp_dir) as fixtures:
            
            # Step 1: Create test input data
            logger.info("Creating test input data...")
            
            # Create a test volume with some degradation
            clean_volume = fixtures.get_small_test_volume()  # 5x256x256
            degraded_input, _ = fixtures.get_training_pair(
                volume_size="small", 
                degradation_level="moderate"
            )
            
            # Save test input as TIFF file
            input_path = fixtures.save_test_tiff(degraded_input, "test_input")
            logger.info(f"Created test input: {input_path}")
            logger.info(f"Input volume shape: {degraded_input.shape}")
            
            # Step 2: Create and save a mock trained model
            logger.info("Creating mock trained model...")
            model_path = fixtures.save_mock_model("trained_model.pth")
            logger.info(f"Mock model saved: {model_path}")
            
            # Step 3: Configure inference parameters
            inference_config = InferenceConfig(
                patch_size=(128, 128),  # Smaller patches for faster processing
                overlap=16,
                batch_size=4,
                device='cpu',  # Use CPU for compatibility
                memory_limit_gb=2.0,
                enable_quality_metrics=True,
                preserve_metadata=True
            )
            
            logger.info("Inference configuration:")
            logger.info(f"  Patch size: {inference_config.patch_size}")
            logger.info(f"  Overlap: {inference_config.overlap}")
            logger.info(f"  Batch size: {inference_config.batch_size}")
            logger.info(f"  Device: {inference_config.device}")
            logger.info(f"  Memory limit: {inference_config.memory_limit_gb}GB")
            
            # Step 4: Initialize inference API
            api = ImageEnhancementAPI(inference_config)
            
            # Step 5: Load the trained model
            logger.info("Loading trained model...")
            success = api.load_model(model_path)
            
            if not success:
                logger.error("Failed to load model")
                return
            
            logger.info("Model loaded successfully")
            
            # Get model information
            model_info = api.get_system_info()
            logger.info(f"Using device: {model_info['device']}")
            logger.info(f"Model loaded: {model_info['model_loaded']}")
            
            # Step 6: Enhance the TIFF file
            logger.info("Enhancing TIFF file...")
            output_path = str(Path(temp_dir) / "enhanced_output.tif")
            
            # Create progress callback
            def progress_callback(message: str, progress: float):
                print(f"\rProgress: {message} ({progress*100:.1f}%)", end='', flush=True)
                if progress >= 1.0:
                    print()  # New line when complete
            
            # Perform enhancement
            result = api.enhance_3d_tiff(
                input_path, 
                output_path,
                progress_callback=progress_callback,
                calculate_metrics=True
            )
            
            # Step 7: Check results
            if result.success:
                logger.info(f"Enhancement completed successfully!")
                logger.info(f"Output saved: {output_path}")
                logger.info(f"Processing time: {result.processing_time:.2f}s")
                logger.info(f"Input shape: {result.input_shape}")
                
                # Display quality metrics
                if result.quality_metrics:
                    logger.info("Quality Metrics:")
                    logger.info(f"  PSNR: {result.quality_metrics.get('psnr', 'N/A'):.2f} dB")
                    logger.info(f"  MSE: {result.quality_metrics.get('mse', 'N/A'):.6f}")
                    logger.info(f"  MAE: {result.quality_metrics.get('mae', 'N/A'):.6f}")
                    logger.info(f"  Mean intensity: {result.quality_metrics.get('mean_intensity', 'N/A'):.3f}")
                    logger.info(f"  Dynamic range: {result.quality_metrics.get('dynamic_range', 'N/A'):.3f}")
                
                # Get comprehensive metrics
                comprehensive_metrics = api.get_enhancement_metrics(result)
                if 'processing_efficiency' in comprehensive_metrics:
                    efficiency = comprehensive_metrics['processing_efficiency']
                    logger.info("Processing Efficiency:")
                    logger.info(f"  Pixels per second: {efficiency.get('pixels_per_second', 'N/A'):.0f}")
                    logger.info(f"  Slices per second: {efficiency.get('slices_per_second', 'N/A'):.2f}")
                    logger.info(f"  Megapixels processed: {efficiency.get('megapixels_processed', 'N/A'):.2f}")
                
            else:
                logger.error(f"Enhancement failed: {result.error_message}")
                if result.warnings:
                    logger.warning("Warnings:")
                    for warning in result.warnings:
                        logger.warning(f"  {warning}")
                return
            
            # Step 8: Demonstrate array-based processing
            logger.info("\nDemonstrating array-based processing...")
            
            # Create test array
            test_array = np.random.rand(3, 128, 128).astype(np.float32)
            logger.info(f"Test array shape: {test_array.shape}")
            
            # Process array directly
            array_result = api.enhance_3d_array(test_array, calculate_metrics=True)
            
            if array_result.success:
                logger.info("Array processing completed successfully!")
                logger.info(f"Processing time: {array_result.processing_time:.2f}s")
                
                # Get enhanced array
                enhanced_array = array_result.quality_metrics['enhanced_array']
                logger.info(f"Enhanced array shape: {enhanced_array.shape}")
                logger.info(f"Enhanced array dtype: {enhanced_array.dtype}")
                
                # Compare input and output
                logger.info("Array Comparison:")
                logger.info(f"  Input range: [{test_array.min():.3f}, {test_array.max():.3f}]")
                logger.info(f"  Output range: [{enhanced_array.min():.3f}, {enhanced_array.max():.3f}]")
                
            else:
                logger.error(f"Array processing failed: {array_result.error_message}")
            
            # Step 9: Demonstrate configuration updates
            logger.info("\nDemonstrating runtime configuration updates...")
            
            # Update processing configuration
            api.set_processing_config(
                batch_size=2,
                memory_limit_gb=1.0
            )
            
            logger.info("Updated configuration:")
            updated_info = api.get_system_info()
            logger.info(f"  New batch size: {updated_info['config']['batch_size']}")
            logger.info(f"  New memory limit: {updated_info['config']['memory_limit_gb']}GB")
            
            logger.info("Basic inference example completed successfully!")
            
    except Exception as e:
        logger.error(f"Inference example failed: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary files")


if __name__ == '__main__':
    main()