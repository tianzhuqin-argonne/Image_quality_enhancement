#!/usr/bin/env python3
"""
Command-line interface for 3D image enhancement inference.

This script provides a comprehensive CLI for enhancing 3D TIFF images
using trained U-Net models with configurable parameters and monitoring.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..inference.api import ImageEnhancementAPI
from ..core.config import InferenceConfig
from ..core.error_handling import setup_global_error_handling


class InferenceCLI:
    """Command-line interface for inference pipeline."""
    
    def __init__(self):
        """Initialize inference CLI."""
        self.parser = self._create_parser()
        self.logger = logging.getLogger(__name__)
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser for inference CLI."""
        parser = argparse.ArgumentParser(
            description="Enhance 3D TIFF images using trained U-Net models",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic enhancement with default settings
  python -m src.cli.inference_cli --model model.pth --input image.tif --output enhanced.tif

  # Batch processing multiple files
  python -m src.cli.inference_cli --model model.pth --input-dir inputs/ --output-dir outputs/

  # Enhancement with custom patch size and GPU
  python -m src.cli.inference_cli --model model.pth --input image.tif --output enhanced.tif \\
    --patch-size 512 512 --device cuda --batch-size 8

  # Enhancement with quality metrics
  python -m src.cli.inference_cli --model model.pth --input image.tif --output enhanced.tif \\
    --calculate-metrics --save-metrics metrics.json

  # Memory-constrained processing
  python -m src.cli.inference_cli --model model.pth --input large_image.tif --output enhanced.tif \\
    --memory-limit 2.0 --batch-size 2
            """
        )
        
        # Required arguments
        required = parser.add_argument_group('required arguments')
        required.add_argument(
            '--model', type=str, required=True,
            help='Path to trained model file (.pth)'
        )
        
        # Input/output - either single file or batch processing
        io_group = parser.add_mutually_exclusive_group(required=True)
        io_group.add_argument(
            '--input', type=str,
            help='Input TIFF file to enhance'
        )
        io_group.add_argument(
            '--input-dir', type=str,
            help='Directory containing input TIFF files for batch processing'
        )
        
        parser.add_argument(
            '--output', type=str,
            help='Output path for enhanced TIFF file (required with --input)'
        )
        parser.add_argument(
            '--output-dir', type=str,
            help='Output directory for batch processing (required with --input-dir)'
        )
        
        # Processing parameters
        processing = parser.add_argument_group('processing parameters')
        processing.add_argument(
            '--patch-size', type=int, nargs=2, default=[256, 256], metavar=('H', 'W'),
            help='Patch size for processing (height width) (default: 256 256)'
        )
        processing.add_argument(
            '--overlap', type=int, default=32,
            help='Overlap between patches (default: 32)'
        )
        processing.add_argument(
            '--batch-size', type=int, default=32,
            help='Batch size for processing (default: 32)'
        )
        
        # Hardware and performance
        hardware = parser.add_argument_group('hardware and performance')
        hardware.add_argument(
            '--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], default='auto',
            help='Device to use for inference (default: auto)'
        )
        hardware.add_argument(
            '--memory-limit', type=float, default=8.0,
            help='Memory limit in GB (default: 8.0)'
        )
        
        # Quality and metrics
        quality = parser.add_argument_group('quality and metrics')
        quality.add_argument(
            '--calculate-metrics', action='store_true',
            help='Calculate quality metrics (PSNR, MSE, etc.)'
        )
        quality.add_argument(
            '--save-metrics', type=str,
            help='Save quality metrics to JSON file'
        )
        quality.add_argument(
            '--preserve-metadata', action='store_true', default=True,
            help='Preserve original TIFF metadata (default: True)'
        )
        quality.add_argument(
            '--no-preserve-metadata', action='store_false', dest='preserve_metadata',
            help='Do not preserve original TIFF metadata'
        )
        
        # Output options
        output_opts = parser.add_argument_group('output options')
        output_opts.add_argument(
            '--output-format', type=str, choices=['tiff'], default='tiff',
            help='Output format (default: tiff)'
        )
        output_opts.add_argument(
            '--compression', type=str, choices=['none', 'lzw', 'jpeg'],
            help='Output compression (default: none)'
        )
        output_opts.add_argument(
            '--overwrite', action='store_true',
            help='Overwrite existing output files'
        )
        
        # Progress and logging
        logging_group = parser.add_argument_group('progress and logging')
        logging_group.add_argument(
            '--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
            help='Logging level (default: INFO)'
        )
        logging_group.add_argument(
            '--progress', action='store_true', default=True,
            help='Show progress bar (default: True)'
        )
        logging_group.add_argument(
            '--no-progress', action='store_false', dest='progress',
            help='Hide progress bar'
        )
        logging_group.add_argument(
            '--quiet', action='store_true',
            help='Suppress all output except errors'
        )
        logging_group.add_argument(
            '--config-file', type=str,
            help='Load configuration from JSON file'
        )
        logging_group.add_argument(
            '--save-config', type=str,
            help='Save current configuration to JSON file'
        )
        
        # Batch processing options
        batch = parser.add_argument_group('batch processing options')
        batch.add_argument(
            '--max-files', type=int,
            help='Maximum number of files to process in batch mode'
        )
        batch.add_argument(
            '--file-pattern', type=str, default='*.tif',
            help='File pattern for batch processing (default: *.tif)'
        )
        batch.add_argument(
            '--recursive', action='store_true',
            help='Process files recursively in subdirectories'
        )
        
        return parser
    
    def _setup_logging(self, log_level: str, quiet: bool = False) -> None:
        """Setup logging configuration."""
        if quiet:
            log_level = 'ERROR'
        
        # Setup global error handling
        setup_global_error_handling()
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        self.logger.info(f"Logging configured: level={log_level}")
    
    def _load_config_from_file(self, config_file: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
            sys.exit(1)
    
    def _save_config_to_file(self, config: InferenceConfig, config_file: str) -> None:
        """Save configuration to JSON file."""
        try:
            config_dict = {
                'patch_size': config.patch_size,
                'overlap': config.overlap,
                'batch_size': config.batch_size,
                'device': config.device,
                'memory_limit_gb': config.memory_limit_gb,
                'enable_quality_metrics': config.enable_quality_metrics,
                'preserve_metadata': config.preserve_metadata,
                'output_format': config.output_format,
                'compression': config.compression
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_file}: {e}")
    
    def _create_inference_config(self, args: argparse.Namespace) -> InferenceConfig:
        """Create inference configuration from arguments."""
        # Load base config from file if specified
        if args.config_file:
            file_config = self._load_config_from_file(args.config_file)
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None and key in file_config:
                    file_config[key] = value
            
            # Convert to InferenceConfig
            config = InferenceConfig(
                patch_size=tuple(file_config.get('patch_size', [256, 256])),
                overlap=file_config.get('overlap', 32),
                batch_size=file_config.get('batch_size', 32),
                device=file_config.get('device', 'auto'),
                memory_limit_gb=file_config.get('memory_limit', 8.0),
                enable_quality_metrics=file_config.get('calculate_metrics', False),
                preserve_metadata=file_config.get('preserve_metadata', True),
                output_format=file_config.get('output_format', 'tiff'),
                compression=file_config.get('compression', None)
            )
        else:
            # Create config from command line arguments
            config = InferenceConfig(
                patch_size=tuple(args.patch_size),
                overlap=args.overlap,
                batch_size=args.batch_size,
                device=args.device,
                memory_limit_gb=args.memory_limit,
                enable_quality_metrics=args.calculate_metrics,
                preserve_metadata=args.preserve_metadata,
                output_format=args.output_format,
                compression=args.compression
            )
        
        return config
    
    def _find_input_files(self, input_dir: str, pattern: str, recursive: bool, max_files: Optional[int]) -> List[str]:
        """Find input files for batch processing."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")
        
        # Find files
        if recursive:
            files = list(input_path.rglob(pattern))
        else:
            files = list(input_path.glob(pattern))
        
        if not files:
            raise ValueError(f"No files matching pattern '{pattern}' found in {input_path}")
        
        # Sort for consistent ordering
        files = sorted([str(f) for f in files])
        
        # Limit number of files if specified
        if max_files:
            files = files[:max_files]
        
        self.logger.info(f"Found {len(files)} files for processing")
        return files
    
    def _create_progress_callback(self, show_progress: bool, filename: str = ""):
        """Create progress callback function."""
        if not show_progress:
            return None
        
        def progress_callback(message: str, progress: float):
            if filename:
                print(f"\r{filename}: {message} ({progress*100:.1f}%)", end='', flush=True)
            else:
                print(f"\r{message} ({progress*100:.1f}%)", end='', flush=True)
            
            if progress >= 1.0:
                print()  # New line when complete
        
        return progress_callback
    
    def _save_metrics(self, metrics: dict, metrics_file: str) -> None:
        """Save metrics to JSON file."""
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            self.logger.info(f"Saved metrics to {metrics_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics to {metrics_file}: {e}")
    
    def _process_single_file(self, api: ImageEnhancementAPI, input_path: str, output_path: str, 
                           show_progress: bool, calculate_metrics: bool) -> dict:
        """Process a single file."""
        self.logger.info(f"Processing: {input_path} -> {output_path}")
        
        # Create progress callback
        progress_callback = self._create_progress_callback(show_progress, Path(input_path).name)
        
        # Process file
        start_time = time.time()
        result = api.enhance_3d_tiff(
            input_path, output_path,
            progress_callback=progress_callback,
            calculate_metrics=calculate_metrics
        )
        processing_time = time.time() - start_time
        
        if result.success:
            self.logger.info(f"Successfully processed {input_path} in {processing_time:.2f}s")
            if result.quality_metrics:
                self.logger.info(f"Quality metrics: PSNR={result.quality_metrics.get('psnr', 'N/A'):.2f}")
        else:
            self.logger.error(f"Failed to process {input_path}: {result.error_message}")
        
        return api.get_enhancement_metrics(result)
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run inference CLI."""
        try:
            # Parse arguments
            parsed_args = self.parser.parse_args(args)
            
            # Validate arguments
            if parsed_args.input and not parsed_args.output:
                print("Error: --output is required when using --input")
                return 1
            if parsed_args.input_dir and not parsed_args.output_dir:
                print("Error: --output-dir is required when using --input-dir")
                return 1
            
            # Check PyTorch availability
            if not TORCH_AVAILABLE:
                print("Error: PyTorch is required for inference but not available.")
                print("Please install PyTorch: pip install torch torchvision")
                return 1
            
            # Setup logging
            self._setup_logging(parsed_args.log_level, parsed_args.quiet)
            
            if not parsed_args.quiet:
                self.logger.info("Starting 3D image enhancement inference")
                self.logger.info(f"PyTorch version: {torch.__version__}")
                self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
            
            # Create inference configuration
            config = self._create_inference_config(parsed_args)
            
            # Save configuration if requested
            if parsed_args.save_config:
                self._save_config_to_file(config, parsed_args.save_config)
            
            # Initialize API
            api = ImageEnhancementAPI(config)
            
            # Load model
            self.logger.info(f"Loading model: {parsed_args.model}")
            if not api.load_model(parsed_args.model):
                self.logger.error("Failed to load model")
                return 1
            
            # Get system info
            if not parsed_args.quiet:
                system_info = api.get_system_info()
                self.logger.info(f"Using device: {system_info['device']}")
                if system_info.get('model_loaded'):
                    model_info = system_info.get('model_info', {})
                    self.logger.info(f"Model parameters: {model_info.get('total_parameters', 'N/A')}")
            
            all_metrics = []
            
            # Process files
            if parsed_args.input:
                # Single file processing
                if Path(parsed_args.output).exists() and not parsed_args.overwrite:
                    self.logger.error(f"Output file exists: {parsed_args.output}. Use --overwrite to replace.")
                    return 1
                
                # Create output directory if needed
                Path(parsed_args.output).parent.mkdir(parents=True, exist_ok=True)
                
                metrics = self._process_single_file(
                    api, parsed_args.input, parsed_args.output,
                    parsed_args.progress, parsed_args.calculate_metrics
                )
                all_metrics.append(metrics)
                
            else:
                # Batch processing
                input_files = self._find_input_files(
                    parsed_args.input_dir, parsed_args.file_pattern,
                    parsed_args.recursive, parsed_args.max_files
                )
                
                # Create output directory
                output_dir = Path(parsed_args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process each file
                for i, input_file in enumerate(input_files):
                    input_path = Path(input_file)
                    output_path = output_dir / f"{input_path.stem}_enhanced{input_path.suffix}"
                    
                    if output_path.exists() and not parsed_args.overwrite:
                        self.logger.warning(f"Skipping existing file: {output_path}")
                        continue
                    
                    self.logger.info(f"Processing file {i+1}/{len(input_files)}")
                    
                    try:
                        metrics = self._process_single_file(
                            api, str(input_path), str(output_path),
                            parsed_args.progress, parsed_args.calculate_metrics
                        )
                        metrics['input_file'] = str(input_path)
                        metrics['output_file'] = str(output_path)
                        all_metrics.append(metrics)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {input_path}: {e}")
                        continue
            
            # Save metrics if requested
            if parsed_args.save_metrics and all_metrics:
                self._save_metrics(all_metrics, parsed_args.save_metrics)
            
            # Summary
            successful = sum(1 for m in all_metrics if m.get('success', False))
            total = len(all_metrics)
            
            if not parsed_args.quiet:
                self.logger.info(f"Processing complete: {successful}/{total} files successful")
                
                if parsed_args.calculate_metrics and all_metrics:
                    # Calculate average metrics
                    avg_psnr = sum(m.get('quality_metrics', {}).get('psnr', 0) for m in all_metrics if m.get('success')) / max(successful, 1)
                    avg_time = sum(m.get('processing_time', 0) for m in all_metrics if m.get('success')) / max(successful, 1)
                    
                    self.logger.info(f"Average PSNR: {avg_psnr:.2f}")
                    self.logger.info(f"Average processing time: {avg_time:.2f}s")
            
            return 0 if successful == total else 1
            
        except KeyboardInterrupt:
            self.logger.info("Processing interrupted by user")
            return 1
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return 1


def main():
    """Main entry point for inference CLI."""
    cli = InferenceCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())