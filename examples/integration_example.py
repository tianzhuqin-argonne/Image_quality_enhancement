#!/usr/bin/env python3
"""
Integration example showing how to integrate the 3D image enhancement system
into existing codebases and workflows.

This example demonstrates various integration patterns and best practices.
"""

import logging
from pathlib import Path
import tempfile
import shutil
import numpy as np
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.inference.api import ImageEnhancementAPI
    from src.core.config import InferenceConfig, EnhancementResult
    from src.core.data_models import SliceStack
    from src.testing.test_fixtures import TestDataFixtures
    from src.core.error_handling import ErrorHandler, with_error_handling
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running from the project root directory")
    exit(1)


class ImageEnhancementService:
    """
    Example service class showing how to integrate the enhancement system
    into a larger application or service.
    """
    
    def __init__(self, model_path: str, config: Optional[InferenceConfig] = None):
        """
        Initialize the enhancement service.
        
        Args:
            model_path: Path to trained model
            config: Optional inference configuration
        """
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.api = None
        self.error_handler = ErrorHandler()
        
        # Initialize API
        self._initialize_api()
    
    def _initialize_api(self) -> None:
        """Initialize the enhancement API."""
        try:
            self.api = ImageEnhancementAPI(self.config)
            
            if not self.api.load_model(self.model_path):
                raise RuntimeError(f"Failed to load model: {self.model_path}")
            
            logger.info("Enhancement service initialized successfully")
            
        except Exception as e:
            self.error_handler.handle_error(
                e, "service_initialization", 
                recoverable=False,
                recovery_suggestions=["Check model path", "Verify model compatibility"]
            )
            raise
    
    @with_error_handling(ErrorHandler(), "image_enhancement", recoverable=True)
    def enhance_image(
        self, 
        input_path: str, 
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Enhance a single image file.
        
        Args:
            input_path: Path to input image
            output_path: Path to save enhanced image
            progress_callback: Optional progress callback
            
        Returns:
            Dictionary with enhancement results and metrics
        """
        logger.info(f"Enhancing image: {input_path}")
        
        result = self.api.enhance_3d_tiff(
            input_path, output_path,
            progress_callback=progress_callback,
            calculate_metrics=True
        )
        
        if not result.success:
            raise RuntimeError(f"Enhancement failed: {result.error_message}")
        
        # Return structured results
        return {
            'success': True,
            'input_path': input_path,
            'output_path': output_path,
            'processing_time': result.processing_time,
            'quality_metrics': result.quality_metrics,
            'warnings': result.warnings
        }
    
    def enhance_batch(
        self, 
        input_files: List[str], 
        output_dir: str,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Enhance multiple images in batch.
        
        Args:
            input_files: List of input file paths
            output_dir: Output directory
            progress_callback: Optional progress callback
            
        Returns:
            List of enhancement results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        for i, input_file in enumerate(input_files):
            try:
                input_path = Path(input_file)
                output_file = output_path / f"{input_path.stem}_enhanced{input_path.suffix}"
                
                # Update progress
                if progress_callback:
                    progress_callback(f"Processing {i+1}/{len(input_files)}", i / len(input_files))
                
                # Enhance image
                result = self.enhance_image(str(input_path), str(output_file))
                results.append(result)
                
                logger.info(f"Enhanced {input_file} -> {output_file}")
                
            except Exception as e:
                error_result = {
                    'success': False,
                    'input_path': input_file,
                    'error': str(e)
                }
                results.append(error_result)
                logger.error(f"Failed to enhance {input_file}: {e}")
        
        if progress_callback:
            progress_callback("Batch processing complete", 1.0)
        
        return results
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get service status and diagnostics."""
        if not self.api:
            return {'status': 'not_initialized'}
        
        system_info = self.api.get_system_info()
        error_summary = self.error_handler.get_error_summary()
        
        return {
            'status': 'ready' if system_info['model_loaded'] else 'error',
            'model_path': self.model_path,
            'device': system_info['device'],
            'model_info': system_info.get('model_info', {}),
            'config': system_info['config'],
            'error_summary': error_summary
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.api:
            self.api.cleanup()
        logger.info("Enhancement service cleaned up")


class BatchProcessor:
    """
    Example batch processor for handling large-scale image enhancement tasks.
    """
    
    def __init__(self, service: ImageEnhancementService):
        """Initialize batch processor."""
        self.service = service
        self.results_log = []
    
    def process_directory(
        self, 
        input_dir: str, 
        output_dir: str,
        file_pattern: str = "*.tif",
        max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_pattern: File pattern to match
            max_files: Maximum number of files to process
            
        Returns:
            Processing summary
        """
        input_path = Path(input_dir)
        
        # Find files
        files = list(input_path.glob(file_pattern))
        if max_files:
            files = files[:max_files]
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files
        def progress_callback(message: str, progress: float):
            print(f"\rBatch Progress: {message} ({progress*100:.1f}%)", end='', flush=True)
            if progress >= 1.0:
                print()
        
        results = self.service.enhance_batch(
            [str(f) for f in files], 
            output_dir,
            progress_callback
        )
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        summary = {
            'total_files': len(files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(files) if files else 0,
            'results': results
        }
        
        logger.info(f"Batch processing complete: {successful}/{len(files)} successful")
        return summary


def demonstrate_basic_integration():
    """Demonstrate basic integration patterns."""
    logger.info("=== Basic Integration Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with TestDataFixtures(temp_dir) as fixtures:
            # Create test data and model
            test_volume = fixtures.get_small_test_volume()
            input_path = fixtures.save_test_tiff(test_volume, "test_input")
            model_path = fixtures.save_mock_model("integration_model.pth")
            
            # Initialize service
            config = InferenceConfig(
                patch_size=(128, 128),
                batch_size=4,
                device='cpu',
                enable_quality_metrics=True
            )
            
            service = ImageEnhancementService(model_path, config)
            
            # Check service status
            status = service.get_service_status()
            logger.info(f"Service status: {status['status']}")
            logger.info(f"Using device: {status['device']}")
            
            # Enhance single image
            output_path = str(Path(temp_dir) / "enhanced_output.tif")
            
            def progress_callback(message: str, progress: float):
                logger.info(f"Progress: {message} ({progress*100:.1f}%)")
            
            result = service.enhance_image(input_path, output_path, progress_callback)
            
            logger.info("Enhancement Results:")
            logger.info(f"  Success: {result['success']}")
            logger.info(f"  Processing time: {result['processing_time']:.2f}s")
            if result['quality_metrics']:
                logger.info(f"  PSNR: {result['quality_metrics'].get('psnr', 'N/A'):.2f}")
            
            # Cleanup
            service.cleanup()
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_batch_processing():
    """Demonstrate batch processing integration."""
    logger.info("\n=== Batch Processing Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with TestDataFixtures(temp_dir) as fixtures:
            # Create test dataset
            dataset_pairs = fixtures.create_test_dataset(
                num_pairs=5, volume_size="small", degradation_level="light"
            )
            
            input_files = [pair[0] for pair in dataset_pairs]
            model_path = fixtures.save_mock_model("batch_model.pth")
            
            # Initialize service and batch processor
            service = ImageEnhancementService(model_path)
            processor = BatchProcessor(service)
            
            # Process batch
            output_dir = str(Path(temp_dir) / "batch_outputs")
            summary = processor.process_directory(
                str(Path(dataset_pairs[0][0]).parent),  # Input directory
                output_dir,
                max_files=3
            )
            
            logger.info("Batch Processing Summary:")
            logger.info(f"  Total files: {summary['total_files']}")
            logger.info(f"  Successful: {summary['successful']}")
            logger.info(f"  Failed: {summary['failed']}")
            logger.info(f"  Success rate: {summary['success_rate']*100:.1f}%")
            
            # Cleanup
            service.cleanup()
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_error_handling():
    """Demonstrate error handling and recovery."""
    logger.info("\n=== Error Handling Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with TestDataFixtures(temp_dir) as fixtures:
            model_path = fixtures.save_mock_model("error_model.pth")
            
            # Initialize service
            service = ImageEnhancementService(model_path)
            
            # Test with non-existent file
            try:
                result = service.enhance_image("nonexistent.tif", "output.tif")
            except Exception as e:
                logger.info(f"Expected error caught: {type(e).__name__}")
            
            # Check error summary
            status = service.get_service_status()
            error_summary = status['error_summary']
            
            logger.info("Error Summary:")
            logger.info(f"  Total errors: {error_summary['total_errors']}")
            logger.info(f"  Recoverable errors: {error_summary['recoverable_errors']}")
            
            if error_summary['recent_errors']:
                recent = error_summary['recent_errors'][-1]
                logger.info(f"  Most recent error: {recent['error_type']}")
                logger.info(f"  Recovery suggestions: {recent.get('recovery_suggestions', [])}")
            
            # Cleanup
            service.cleanup()
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def demonstrate_custom_workflow():
    """Demonstrate custom workflow integration."""
    logger.info("\n=== Custom Workflow Example ===")
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        with TestDataFixtures(temp_dir) as fixtures:
            # Create test data
            test_volume = fixtures.get_small_test_volume()
            model_path = fixtures.save_mock_model("workflow_model.pth")
            
            # Custom workflow: preprocess -> enhance -> postprocess
            
            # Step 1: Preprocess (example: normalize intensity)
            logger.info("Step 1: Preprocessing")
            preprocessed_data = test_volume.to_numpy()
            preprocessed_data = (preprocessed_data - preprocessed_data.mean()) / preprocessed_data.std()
            preprocessed_data = np.clip(preprocessed_data, 0, 1)
            
            # Step 2: Enhance using API
            logger.info("Step 2: Enhancement")
            api = ImageEnhancementAPI(InferenceConfig(device='cpu'))
            api.load_model(model_path)
            
            result = api.enhance_3d_array(preprocessed_data.astype(np.float32))
            
            if result.success:
                enhanced_data = result.quality_metrics['enhanced_array']
                
                # Step 3: Postprocess (example: apply custom filter)
                logger.info("Step 3: Postprocessing")
                # Simple smoothing filter
                from scipy import ndimage
                postprocessed_data = ndimage.gaussian_filter(enhanced_data, sigma=0.5)
                
                logger.info("Custom workflow completed successfully")
                logger.info(f"  Original shape: {test_volume.shape}")
                logger.info(f"  Enhanced shape: {enhanced_data.shape}")
                logger.info(f"  Processing time: {result.processing_time:.2f}s")
                
                # Save final result
                final_volume = SliceStack(postprocessed_data)
                output_path = fixtures.save_test_tiff(final_volume, "workflow_output")
                logger.info(f"  Final output saved: {output_path}")
            
            # Cleanup
            api.cleanup()
            
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all integration examples."""
    logger.info("Starting integration examples")
    
    try:
        demonstrate_basic_integration()
        demonstrate_batch_processing()
        demonstrate_error_handling()
        demonstrate_custom_workflow()
        
        logger.info("\nAll integration examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration examples failed: {e}")
        raise


if __name__ == '__main__':
    main()