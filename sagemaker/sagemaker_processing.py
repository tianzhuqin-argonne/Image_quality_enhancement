#!/usr/bin/env python3
"""
SageMaker processing script for data preprocessing and batch inference.

This script can be used for SageMaker Processing jobs to preprocess
training data or perform batch inference on large datasets.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
import json

# Add the source directory to Python path
sys.path.append('/opt/ml/code')

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import Sigray ML Platform components
from src.inference.api import ImageEnhancementAPI
from src.core.config import InferenceConfig
from src.core.tiff_handler import TIFFDataHandler
from src.testing.synthetic_data import SyntheticDataGenerator

# SageMaker processing paths
SM_INPUT_DIR = os.environ.get('SM_INPUT_DIR', '/opt/ml/processing/input')
SM_OUTPUT_DIR = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/processing/output')
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '/opt/ml/processing/model')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SageMaker Processing for Sigray ML Platform')
    
    # Processing mode
    parser.add_argument('--mode', type=str, choices=['preprocess', 'inference', 'generate'], 
                       default='inference', help='Processing mode')
    
    # Inference parameters
    parser.add_argument('--model-path', type=str, help='Path to trained model (for inference mode)')
    parser.add_argument('--patch-size', type=str, default='256,256', help='Patch size (height,width)')
    parser.add_argument('--overlap', type=int, default=32, help='Patch overlap')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--memory-limit', type=float, default=8.0, help='Memory limit in GB')
    
    # Data generation parameters (for generate mode)
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to generate')
    parser.add_argument('--volume-size', type=str, default='medium', choices=['small', 'medium', 'large'],
                       help='Volume size for generated data')
    parser.add_argument('--degradation-level', type=str, default='moderate', 
                       choices=['light', 'moderate', 'heavy'], help='Degradation level')
    
    # I/O parameters
    parser.add_argument('--input-dir', type=str, default=SM_INPUT_DIR, help='Input directory')
    parser.add_argument('--output-dir', type=str, default=SM_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--model-dir', type=str, default=SM_MODEL_DIR, help='Model directory')
    
    return parser.parse_args()


def preprocess_data(input_dir, output_dir, args):
    """Preprocess training data."""
    logger.info(f"Preprocessing data from {input_dir} to {output_dir}")
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize TIFF handler
    tiff_handler = TIFFDataHandler()
    
    # Find all TIFF files
    tiff_files = list(input_path.rglob('*.tif*'))
    logger.info(f"Found {len(tiff_files)} TIFF files")
    
    processed_count = 0
    
    for tiff_file in tiff_files:
        try:
            logger.info(f"Processing: {tiff_file}")
            
            # Load and validate
            data = tiff_handler.load_3d_tiff(str(tiff_file))
            
            if not tiff_handler.validate_dimensions(data):
                logger.warning(f"Skipping file with invalid dimensions: {tiff_file}")
                continue
            
            # Get metadata
            metadata = tiff_handler.get_metadata(str(tiff_file))
            
            # Create output filename
            relative_path = tiff_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed file
            tiff_handler.save_3d_tiff(data, str(output_file))
            
            # Save metadata
            metadata_file = output_file.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {tiff_file}: {e}")
            continue
    
    logger.info(f"Preprocessing completed: {processed_count}/{len(tiff_files)} files processed")
    
    # Save processing summary
    summary = {
        'total_files': len(tiff_files),
        'processed_files': processed_count,
        'failed_files': len(tiff_files) - processed_count,
        'processing_mode': 'preprocess'
    }
    
    summary_file = output_path / 'processing_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def batch_inference(input_dir, output_dir, model_dir, args):
    """Perform batch inference on images."""
    logger.info(f"Running batch inference from {input_dir} to {output_dir}")
    
    # Parse patch size
    patch_size = tuple(map(int, args.patch_size.split(',')))
    
    # Create inference configuration
    config = InferenceConfig(
        patch_size=patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        device=args.device,
        memory_limit_gb=args.memory_limit,
        enable_quality_metrics=True,
        preserve_metadata=True
    )
    
    # Initialize API
    api = ImageEnhancementAPI(config)
    
    # Load model
    model_path = args.model_path or str(Path(model_dir) / 'model.pth')
    if not Path(model_path).exists():
        # Look for any .pth file in model directory
        model_files = list(Path(model_dir).glob('*.pth'))
        if model_files:
            model_path = str(model_files[0])
        else:
            raise FileNotFoundError(f"No model file found in {model_dir}")
    
    logger.info(f"Loading model from: {model_path}")
    success = api.load_model(model_path)
    if not success:
        raise RuntimeError("Failed to load model")
    
    # Find input files
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_path.rglob('*.tif*'))
    logger.info(f"Found {len(tiff_files)} files for inference")
    
    # Process each file
    results = []
    
    for i, tiff_file in enumerate(tiff_files):
        try:
            logger.info(f"Processing {i+1}/{len(tiff_files)}: {tiff_file}")
            
            # Create output filename
            relative_path = tiff_file.relative_to(input_path)
            output_file = output_path / relative_path.with_name(
                f"{relative_path.stem}_enhanced{relative_path.suffix}"
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Progress callback
            def progress_callback(message, progress):
                logger.info(f"  {message} ({progress*100:.1f}%)")
            
            # Enhance image
            result = api.enhance_3d_tiff(
                str(tiff_file),
                str(output_file),
                progress_callback=progress_callback,
                calculate_metrics=True
            )
            
            # Record result
            result_info = {
                'input_file': str(tiff_file),
                'output_file': str(output_file),
                'success': result.success,
                'processing_time': result.processing_time,
                'input_shape': result.input_shape,
                'error_message': result.error_message,
                'warnings': result.warnings
            }
            
            if result.success and result.quality_metrics:
                result_info['quality_metrics'] = {
                    k: v for k, v in result.quality_metrics.items() 
                    if k != 'enhanced_array'  # Exclude large arrays
                }
            
            results.append(result_info)
            
            # Save individual result
            result_file = output_file.with_suffix('.json')
            with open(result_file, 'w') as f:
                json.dump(result_info, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error processing {tiff_file}: {e}")
            results.append({
                'input_file': str(tiff_file),
                'success': False,
                'error_message': str(e)
            })
    
    # Save batch results summary
    successful = sum(1 for r in results if r['success'])
    summary = {
        'total_files': len(tiff_files),
        'successful': successful,
        'failed': len(tiff_files) - successful,
        'processing_mode': 'batch_inference',
        'model_path': model_path,
        'config': {
            'patch_size': patch_size,
            'overlap': args.overlap,
            'batch_size': args.batch_size,
            'device': args.device
        },
        'results': results
    }
    
    summary_file = output_path / 'batch_results.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Batch inference completed: {successful}/{len(tiff_files)} files processed successfully")


def generate_synthetic_data(output_dir, args):
    """Generate synthetic training data."""
    logger.info(f"Generating synthetic data to {output_dir}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data generator
    generator = SyntheticDataGenerator()
    
    # Create directories
    input_dir = output_path / 'input'
    target_dir = output_path / 'target'
    input_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    
    # Generate data
    logger.info(f"Generating {args.num_samples} samples...")
    
    for i in range(args.num_samples):
        try:
            # Generate volume pair
            clean_volume, degraded_volume = generator.create_training_pair(
                volume_size=args.volume_size,
                degradation_level=args.degradation_level
            )
            
            # Save files
            input_file = input_dir / f"input_{i:04d}.tif"
            target_file = target_dir / f"target_{i:04d}.tif"
            
            generator.save_volume(degraded_volume, str(input_file))
            generator.save_volume(clean_volume, str(target_file))
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{args.num_samples} samples")
                
        except Exception as e:
            logger.error(f"Error generating sample {i}: {e}")
            continue
    
    # Save generation summary
    summary = {
        'num_samples': args.num_samples,
        'volume_size': args.volume_size,
        'degradation_level': args.degradation_level,
        'processing_mode': 'generate_data',
        'output_structure': {
            'input_dir': str(input_dir),
            'target_dir': str(target_dir)
        }
    }
    
    summary_file = output_path / 'generation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data generation completed: {args.num_samples} samples created")


def main():
    """Main processing function."""
    try:
        args = parse_args()
        
        logger.info("Starting SageMaker processing job")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Arguments: {vars(args)}")
        
        if args.mode == 'preprocess':
            preprocess_data(args.input_dir, args.output_dir, args)
            
        elif args.mode == 'inference':
            batch_inference(args.input_dir, args.output_dir, args.model_dir, args)
            
        elif args.mode == 'generate':
            generate_synthetic_data(args.output_dir, args)
            
        else:
            raise ValueError(f"Unknown processing mode: {args.mode}")
        
        logger.info("Processing job completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing job failed: {e}")
        raise


if __name__ == '__main__':
    main()