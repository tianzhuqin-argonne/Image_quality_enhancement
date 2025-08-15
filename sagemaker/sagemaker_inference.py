#!/usr/bin/env python3
"""
SageMaker inference script for Sigray ML Platform.

This script handles model loading and inference for SageMaker endpoints
and batch transform jobs.
"""

import json
import logging
import os
import sys
from pathlib import Path
import io
import base64

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model
model_api = None
config = None


def model_fn(model_dir):
    """
    Load the model for inference.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded model API
    """
    global model_api, config
    
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Load training configuration if available
        config_path = Path(model_dir) / 'training_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            
            # Create inference config from training config
            config = InferenceConfig(
                patch_size=tuple(training_config.get('patch_size', [256, 256])),
                overlap=training_config.get('overlap', 32),
                batch_size=32,  # Optimize for inference
                device='auto',
                memory_limit_gb=8.0,
                enable_quality_metrics=True,
                preserve_metadata=True
            )
        else:
            # Use default configuration
            config = InferenceConfig()
        
        # Initialize API
        model_api = ImageEnhancementAPI(config)
        
        # Load model
        model_path = Path(model_dir) / 'model.pth'
        if not model_path.exists():
            # Fallback to other possible model files
            model_files = list(Path(model_dir).glob('*.pth'))
            if model_files:
                model_path = model_files[0]
            else:
                raise FileNotFoundError(f"No model file found in {model_dir}")
        
        success = model_api.load_model(str(model_path))
        if not success:
            raise RuntimeError("Failed to load model")
        
        logger.info("Model loaded successfully")
        return model_api
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body, request_content_type):
    """
    Parse input data for inference.
    
    Args:
        request_body: Raw request body
        request_content_type: Content type of the request
        
    Returns:
        Parsed input data
    """
    logger.info(f"Processing input with content type: {request_content_type}")
    
    try:
        if request_content_type == 'application/json':
            # JSON input - expect base64 encoded image data
            input_data = json.loads(request_body)
            
            if 'image_data' in input_data:
                # Base64 encoded image
                image_bytes = base64.b64decode(input_data['image_data'])
                # Convert to numpy array (assuming it's a serialized numpy array)
                image_array = np.frombuffer(image_bytes, dtype=np.float32)
                
                # Reshape based on provided shape or infer
                if 'shape' in input_data:
                    shape = tuple(input_data['shape'])
                    image_array = image_array.reshape(shape)
                else:
                    # Try to infer shape (this is a simplified example)
                    # In practice, you'd need more sophisticated shape inference
                    raise ValueError("Shape must be provided for numpy array input")
                
                return {
                    'array': image_array,
                    'type': 'array',
                    'metadata': input_data.get('metadata', {})
                }
            
            elif 'file_path' in input_data:
                # File path input (for batch transform)
                return {
                    'file_path': input_data['file_path'],
                    'type': 'file',
                    'metadata': input_data.get('metadata', {})
                }
            
            else:
                raise ValueError("JSON input must contain 'image_data' or 'file_path'")
        
        elif request_content_type == 'application/x-npy':
            # Direct numpy array input
            image_array = np.frombuffer(request_body, dtype=np.float32)
            # Note: Shape information would need to be provided separately
            # This is a simplified example
            return {
                'array': image_array,
                'type': 'array',
                'metadata': {}
            }
        
        elif request_content_type in ['image/tiff', 'application/x-tiff']:
            # TIFF file input
            # Save to temporary file and process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                tmp_file.write(request_body)
                tmp_file_path = tmp_file.name
            
            return {
                'file_path': tmp_file_path,
                'type': 'file',
                'metadata': {},
                'temp_file': True
            }
        
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")
    
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise


def predict_fn(input_data, model):
    """
    Run inference on the input data.
    
    Args:
        input_data: Parsed input data
        model: Loaded model API
        
    Returns:
        Prediction results
    """
    logger.info(f"Running inference on {input_data['type']} input")
    
    try:
        if input_data['type'] == 'array':
            # Array-based inference
            result = model.enhance_3d_array(
                input_data['array'],
                calculate_metrics=True
            )
            
            if result.success:
                enhanced_array = result.quality_metrics.get('enhanced_array')
                return {
                    'success': True,
                    'enhanced_array': enhanced_array,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics,
                    'input_shape': result.input_shape,
                    'warnings': result.warnings
                }
            else:
                return {
                    'success': False,
                    'error_message': result.error_message,
                    'warnings': result.warnings
                }
        
        elif input_data['type'] == 'file':
            # File-based inference
            import tempfile
            
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='_enhanced.tif', delete=False) as tmp_output:
                output_path = tmp_output.name
            
            result = model.enhance_3d_tiff(
                input_data['file_path'],
                output_path,
                calculate_metrics=True
            )
            
            if result.success:
                # Read enhanced file and encode as base64
                with open(output_path, 'rb') as f:
                    enhanced_file_data = f.read()
                
                enhanced_file_b64 = base64.b64encode(enhanced_file_data).decode('utf-8')
                
                # Clean up temporary files
                os.unlink(output_path)
                if input_data.get('temp_file'):
                    os.unlink(input_data['file_path'])
                
                return {
                    'success': True,
                    'enhanced_file_data': enhanced_file_b64,
                    'processing_time': result.processing_time,
                    'quality_metrics': result.quality_metrics,
                    'input_shape': result.input_shape,
                    'output_path': result.output_path,
                    'warnings': result.warnings
                }
            else:
                # Clean up temporary files
                if os.path.exists(output_path):
                    os.unlink(output_path)
                if input_data.get('temp_file'):
                    os.unlink(input_data['file_path'])
                
                return {
                    'success': False,
                    'error_message': result.error_message,
                    'warnings': result.warnings
                }
        
        else:
            raise ValueError(f"Unknown input type: {input_data['type']}")
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {
            'success': False,
            'error_message': str(e),
            'warnings': []
        }


def output_fn(prediction, accept):
    """
    Format the prediction output.
    
    Args:
        prediction: Prediction results
        accept: Requested output format
        
    Returns:
        Formatted output
    """
    logger.info(f"Formatting output for accept type: {accept}")
    
    try:
        if accept == 'application/json':
            # Return JSON response
            # Convert numpy arrays to lists for JSON serialization
            json_prediction = {}
            for key, value in prediction.items():
                if isinstance(value, np.ndarray):
                    json_prediction[key] = value.tolist()
                else:
                    json_prediction[key] = value
            
            return json.dumps(json_prediction, default=str)
        
        elif accept == 'application/x-npy':
            # Return numpy array directly (for enhanced_array)
            if prediction.get('success') and 'enhanced_array' in prediction:
                enhanced_array = prediction['enhanced_array']
                return enhanced_array.tobytes()
            else:
                # Return error as JSON even if numpy was requested
                return json.dumps(prediction, default=str)
        
        elif accept in ['image/tiff', 'application/x-tiff']:
            # Return TIFF file data
            if prediction.get('success') and 'enhanced_file_data' in prediction:
                return base64.b64decode(prediction['enhanced_file_data'])
            else:
                # Return error as JSON
                return json.dumps(prediction, default=str)
        
        else:
            # Default to JSON
            json_prediction = {}
            for key, value in prediction.items():
                if isinstance(value, np.ndarray):
                    json_prediction[key] = value.tolist()
                else:
                    json_prediction[key] = value
            
            return json.dumps(json_prediction, default=str)
    
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        error_response = {
            'success': False,
            'error_message': f"Output formatting error: {str(e)}",
            'warnings': []
        }
        return json.dumps(error_response)


# For batch transform jobs
def transform_fn(model, request_body, request_content_type, accept):
    """
    Complete transform function for batch jobs.
    
    This combines input_fn, predict_fn, and output_fn for batch processing.
    """
    logger.info("Running batch transform")
    
    try:
        # Parse input
        input_data = input_fn(request_body, request_content_type)
        
        # Run prediction
        prediction = predict_fn(input_data, model)
        
        # Format output
        output = output_fn(prediction, accept)
        
        return output
    
    except Exception as e:
        logger.error(f"Error in transform function: {e}")
        error_response = {
            'success': False,
            'error_message': str(e),
            'warnings': []
        }
        return json.dumps(error_response)