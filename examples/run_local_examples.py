#!/usr/bin/env python3
"""
Run all local examples for the Sigray ML Platform.

This script demonstrates the complete workflow from training to inference
using the local development environment.
"""

import logging
import sys
import time
from pathlib import Path
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_installation():
    """Check if the platform is properly installed."""
    logger.info("🔍 Checking installation...")
    
    try:
        from src.inference.api import ImageEnhancementAPI
        from src.training.training_manager import TrainingManager
        from src.core.config import TrainingConfig, InferenceConfig
        from src.testing.test_fixtures import TestDataFixtures
        
        logger.info("✅ All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.error("Please run: pip install -e .")
        return False

def check_gpu_availability():
    """Check GPU availability and performance."""
    logger.info("🖥️ Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ CUDA GPU available: {gpu_name} ({gpu_count} devices)")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("✅ Apple Silicon GPU (MPS) available")
            return 'mps'
        else:
            logger.info("ℹ️ No GPU available, using CPU")
            return 'cpu'
            
    except ImportError:
        logger.error("❌ PyTorch not available")
        return None

def run_training_example(device='auto', quick_mode=True):
    """Run a training example."""
    logger.info("🚀 Starting training example...")
    
    try:
        from src.training.training_manager import TrainingManager
        from src.core.config import TrainingConfig
        from src.testing.test_fixtures import TestDataFixtures
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Using temporary directory: {temp_dir}")
        
        # Configure training (quick mode for demo)
        config = TrainingConfig(
            epochs=5 if quick_mode else 20,
            batch_size=4 if device == 'cpu' else 8,
            learning_rate=1e-3,
            device=device,
            patch_size=(128, 128) if quick_mode else (256, 256),
            overlap=16,
            early_stopping_patience=3,
            checkpoint_interval=2,
            use_augmentation=True
        )
        
        # Create training data
        with TestDataFixtures(temp_dir) as fixtures:
            logger.info("📊 Creating training dataset...")
            
            # Create training pairs
            dataset = fixtures.create_test_dataset(
                num_pairs=8 if quick_mode else 20,
                volume_size="small",
                degradation_level="moderate"
            )
            
            # Save training data
            input_paths = []
            target_paths = []
            
            for i, (input_vol, target_vol) in enumerate(dataset):
                input_path = fixtures.save_test_tiff(input_vol, f"input_{i:03d}")
                target_path = fixtures.save_test_tiff(target_vol, f"target_{i:03d}")
                input_paths.append(input_path)
                target_paths.append(target_path)
            
            logger.info(f"Created {len(input_paths)} training pairs")
            
            # Initialize training manager
            training_output = Path(temp_dir) / "training_output"
            trainer = TrainingManager(
                config=config,
                output_dir=str(training_output),
                use_tensorboard=False  # Disable for example
            )
            
            logger.info(f"Using device: {trainer.device}")
            
            # Prepare training data
            logger.info("📚 Preparing training data...")
            train_loader, val_loader = trainer.prepare_training_data(
                input_paths=input_paths,
                target_paths=target_paths
            )
            
            # Create a simple pre-trained model
            from src.models.model_manager import UNetModelManager
            model_manager = UNetModelManager()
            
            pretrained_model_path = Path(temp_dir) / "pretrained_model.pth"
            model = model_manager.create_unet_model()
            
            import torch
            torch.save({'model_state_dict': model.state_dict()}, pretrained_model_path)
            
            # Start training
            logger.info("🎯 Starting model training...")
            start_time = time.time()
            
            best_model_path = trainer.fine_tune_model(
                str(pretrained_model_path),
                train_loader,
                val_loader
            )
            
            training_time = time.time() - start_time
            
            # Get training summary
            summary = trainer.get_training_summary()
            
            logger.info("✅ Training completed!")
            logger.info(f"⏱️ Training time: {training_time:.1f} seconds")
            logger.info(f"📈 Total epochs: {summary.get('total_epochs', 'N/A')}")
            logger.info(f"🏆 Best epoch: {summary.get('best_epoch', 'N/A')}")
            logger.info(f"📉 Best validation loss: {summary.get('best_validation_loss', 'N/A'):.6f}")
            logger.info(f"💾 Best model saved: {best_model_path}")
            
            return best_model_path, temp_dir
            
    except Exception as e:
        logger.error(f"❌ Training example failed: {e}")
        return None, None

def run_inference_example(model_path=None, temp_dir=None, device='auto'):
    """Run an inference example."""
    logger.info("🔮 Starting inference example...")
    
    try:
        from src.inference.api import ImageEnhancementAPI
        from src.core.config import InferenceConfig
        from src.testing.test_fixtures import TestDataFixtures
        
        # Use provided temp_dir or create new one
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
            cleanup_temp = True
        else:
            cleanup_temp = False
        
        # Configure inference
        config = InferenceConfig(
            patch_size=(128, 128),
            overlap=16,
            batch_size=4 if device == 'cpu' else 16,
            device=device,
            memory_limit_gb=4.0,
            enable_quality_metrics=True,
            preserve_metadata=True
        )
        
        # Initialize API
        api = ImageEnhancementAPI(config)
        
        # Load model
        if model_path and Path(model_path).exists():
            logger.info(f"📥 Loading trained model: {model_path}")
            success = api.load_model(model_path)
        else:
            # Create a mock model for demonstration
            logger.info("📥 Creating mock model for demonstration...")
            with TestDataFixtures(temp_dir) as fixtures:
                mock_model_path = fixtures.save_mock_model("demo_model.pth")
                success = api.load_model(mock_model_path)
        
        if not success:
            logger.error("❌ Failed to load model")
            return False
        
        logger.info("✅ Model loaded successfully")
        
        # Get system info
        system_info = api.get_system_info()
        logger.info(f"🖥️ Using device: {system_info['device']}")
        
        # Test 1: Array-based inference
        logger.info("🧪 Test 1: Array-based inference...")
        
        import numpy as np
        test_array = np.random.rand(3, 128, 128).astype(np.float32)
        logger.info(f"Input array shape: {test_array.shape}")
        
        start_time = time.time()
        result = api.enhance_3d_array(test_array, calculate_metrics=True)
        inference_time = time.time() - start_time
        
        if result.success:
            enhanced_array = result.quality_metrics.get('enhanced_array')
            logger.info("✅ Array inference successful!")
            logger.info(f"⏱️ Processing time: {result.processing_time:.2f}s")
            logger.info(f"🔄 Total inference time: {inference_time:.2f}s")
            logger.info(f"📐 Enhanced array shape: {enhanced_array.shape}")
            
            # Show quality metrics
            if result.quality_metrics:
                logger.info("📊 Quality metrics:")
                for key, value in result.quality_metrics.items():
                    if key != 'enhanced_array' and isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")
        else:
            logger.error(f"❌ Array inference failed: {result.error_message}")
        
        # Test 2: File-based inference
        logger.info("🧪 Test 2: File-based inference...")
        
        with TestDataFixtures(temp_dir) as fixtures:
            # Create test TIFF file
            test_volume = fixtures.get_small_test_volume()
            input_path = fixtures.save_test_tiff(test_volume, "test_input")
            output_path = Path(temp_dir) / "enhanced_output.tif"
            
            logger.info(f"Input file: {input_path}")
            logger.info(f"Output file: {output_path}")
            
            # Progress callback
            def progress_callback(message, progress):
                if progress in [0.2, 0.5, 0.8, 1.0]:  # Show key milestones
                    logger.info(f"  📈 {message} ({progress*100:.0f}%)")
            
            start_time = time.time()
            result = api.enhance_3d_tiff(
                input_path,
                str(output_path),
                progress_callback=progress_callback,
                calculate_metrics=True
            )
            inference_time = time.time() - start_time
            
            if result.success:
                logger.info("✅ File inference successful!")
                logger.info(f"⏱️ Processing time: {result.processing_time:.2f}s")
                logger.info(f"🔄 Total inference time: {inference_time:.2f}s")
                logger.info(f"📐 Input shape: {result.input_shape}")
                logger.info(f"💾 Output saved: {output_path}")
                
                # Show quality metrics
                if result.quality_metrics:
                    logger.info("📊 Quality metrics:")
                    for key, value in result.quality_metrics.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {key}: {value:.4f}")
                
                # Show processing efficiency
                metrics = api.get_enhancement_metrics(result)
                if 'processing_efficiency' in metrics:
                    eff = metrics['processing_efficiency']
                    logger.info("⚡ Processing efficiency:")
                    logger.info(f"  Pixels/second: {eff.get('pixels_per_second', 0):.0f}")
                    logger.info(f"  Slices/second: {eff.get('slices_per_second', 0):.2f}")
                    logger.info(f"  Megapixels processed: {eff.get('megapixels_processed', 0):.2f}")
            else:
                logger.error(f"❌ File inference failed: {result.error_message}")
        
        # Test 3: Configuration updates
        logger.info("🧪 Test 3: Runtime configuration updates...")
        
        api.set_processing_config(
            batch_size=8,
            memory_limit_gb=2.0
        )
        
        updated_info = api.get_system_info()
        logger.info("✅ Configuration updated successfully!")
        logger.info(f"  New batch size: {updated_info['config']['batch_size']}")
        logger.info(f"  New memory limit: {updated_info['config']['memory_limit_gb']}GB")
        
        # Cleanup
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference example failed: {e}")
        return False

def run_cli_examples():
    """Demonstrate CLI usage."""
    logger.info("💻 CLI Examples:")
    
    logger.info("\n📚 Training CLI:")
    logger.info("python -m src.cli.train_cli \\")
    logger.info("  --input-dir data/inputs \\")
    logger.info("  --target-dir data/targets \\")
    logger.info("  --output-dir training_output \\")
    logger.info("  --epochs 50 \\")
    logger.info("  --batch-size 16 \\")
    logger.info("  --device auto")
    
    logger.info("\n🔮 Inference CLI:")
    logger.info("# Single file:")
    logger.info("python -m src.cli.inference_cli \\")
    logger.info("  --model model.pth \\")
    logger.info("  --input image.tif \\")
    logger.info("  --output enhanced.tif \\")
    logger.info("  --calculate-metrics")
    
    logger.info("\n# Batch processing:")
    logger.info("python -m src.cli.inference_cli \\")
    logger.info("  --model model.pth \\")
    logger.info("  --input-dir input_folder/ \\")
    logger.info("  --output-dir enhanced_folder/ \\")
    logger.info("  --batch-size 32")

def main():
    """Run all examples."""
    logger.info("🎉 Welcome to Sigray ML Platform Local Examples!")
    logger.info("=" * 60)
    
    # Check installation
    if not check_installation():
        sys.exit(1)
    
    # Check GPU
    device = check_gpu_availability()
    if device is None:
        sys.exit(1)
    
    logger.info("=" * 60)
    
    # Ask user what to run
    logger.info("🎯 What would you like to run?")
    logger.info("1. Quick demo (training + inference, ~2 minutes)")
    logger.info("2. Training example only (~1 minute)")
    logger.info("3. Inference example only (~30 seconds)")
    logger.info("4. Show CLI examples")
    logger.info("5. Run all examples")
    
    try:
        choice = input("\nEnter your choice (1-5): ").strip()
    except KeyboardInterrupt:
        logger.info("\n👋 Goodbye!")
        sys.exit(0)
    
    start_time = time.time()
    
    if choice == "1":
        # Quick demo
        logger.info("\n🚀 Running quick demo...")
        model_path, temp_dir = run_training_example(device, quick_mode=True)
        if model_path:
            run_inference_example(model_path, temp_dir, device)
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    elif choice == "2":
        # Training only
        logger.info("\n🚀 Running training example...")
        model_path, temp_dir = run_training_example(device, quick_mode=False)
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    elif choice == "3":
        # Inference only
        logger.info("\n🔮 Running inference example...")
        run_inference_example(device=device)
    
    elif choice == "4":
        # CLI examples
        run_cli_examples()
    
    elif choice == "5":
        # All examples
        logger.info("\n🚀 Running all examples...")
        model_path, temp_dir = run_training_example(device, quick_mode=True)
        if model_path:
            run_inference_example(model_path, temp_dir, device)
        run_cli_examples()
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    else:
        logger.info("❌ Invalid choice. Please run again and select 1-5.")
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"🎊 Examples completed in {total_time:.1f} seconds!")
    logger.info("\n📚 Next steps:")
    logger.info("1. Try with your own TIFF data")
    logger.info("2. Experiment with different parameters")
    logger.info("3. Check out the SageMaker integration for cloud scaling")
    logger.info("4. Read the full documentation in README.md")
    logger.info("\n💡 For help: python examples/run_local_examples.py")

if __name__ == "__main__":
    main()