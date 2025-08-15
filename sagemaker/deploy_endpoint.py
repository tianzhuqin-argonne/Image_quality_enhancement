#!/usr/bin/env python3
"""
Script to deploy Sigray ML Platform model as SageMaker endpoint.

This script provides utilities for deploying trained models as
real-time inference endpoints with auto-scaling and monitoring.
"""

import argparse
import boto3
import json
import logging
import time
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy Sigray ML Platform endpoint')
    
    # Model parameters
    parser.add_argument('--model-data', type=str, required=True,
                       help='S3 path to model artifacts (model.tar.gz)')
    parser.add_argument('--endpoint-name', type=str, required=True,
                       help='Name for the SageMaker endpoint')
    
    # Instance configuration
    parser.add_argument('--instance-type', type=str, default='ml.p3.2xlarge',
                       help='Instance type for endpoint')
    parser.add_argument('--initial-instance-count', type=int, default=1,
                       help='Initial number of instances')
    
    # Auto-scaling configuration
    parser.add_argument('--enable-autoscaling', action='store_true',
                       help='Enable auto-scaling for the endpoint')
    parser.add_argument('--min-capacity', type=int, default=1,
                       help='Minimum number of instances for auto-scaling')
    parser.add_argument('--max-capacity', type=int, default=10,
                       help='Maximum number of instances for auto-scaling')
    parser.add_argument('--target-invocations', type=int, default=100,
                       help='Target invocations per minute for scaling')
    
    # Monitoring configuration
    parser.add_argument('--enable-data-capture', action='store_true',
                       help='Enable data capture for monitoring')
    parser.add_argument('--data-capture-percentage', type=int, default=10,
                       help='Percentage of requests to capture')
    
    # Other options
    parser.add_argument('--wait-for-deployment', action='store_true',
                       help='Wait for deployment to complete')
    parser.add_argument('--test-endpoint', action='store_true',
                       help='Test endpoint after deployment')
    
    return parser.parse_args()


def create_model(model_data, role):
    """Create SageMaker model."""
    logger.info(f"Creating model from: {model_data}")
    
    model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point='sagemaker_inference.py',
        source_dir='sagemaker',
        framework_version='1.12.0',
        py_version='py38',
        env={
            'PYTHONPATH': '/opt/ml/code',
            'MMS_DEFAULT_RESPONSE_TIMEOUT': '900'  # 15 minutes timeout
        }
    )
    
    logger.info("✅ Model created successfully")
    return model


def deploy_endpoint(model, endpoint_name, instance_type, initial_instance_count, 
                   enable_data_capture=False, data_capture_percentage=10):
    """Deploy model as SageMaker endpoint."""
    logger.info(f"Deploying endpoint: {endpoint_name}")
    
    # Configure data capture if enabled
    data_capture_config = None
    if enable_data_capture:
        sagemaker_session = sagemaker.Session()
        bucket = sagemaker_session.default_bucket()
        
        data_capture_config = sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=data_capture_percentage,
            destination_s3_uri=f's3://{bucket}/sigray-ml-data-capture/{endpoint_name}/'
        )
        logger.info(f"Data capture enabled: {data_capture_percentage}% sampling")
    
    # Deploy endpoint
    predictor = model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        data_capture_config=data_capture_config
    )
    
    logger.info(f"✅ Endpoint deployed: {endpoint_name}")
    return predictor


def setup_autoscaling(endpoint_name, min_capacity, max_capacity, target_invocations):
    """Set up auto-scaling for the endpoint."""
    logger.info(f"Setting up auto-scaling for: {endpoint_name}")
    
    # Initialize auto-scaling client
    autoscaling_client = boto3.client('application-autoscaling')
    
    # Register scalable target
    resource_id = f'endpoint/{endpoint_name}/variant/AllTraffic'
    
    try:
        autoscaling_client.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_capacity,
            MaxCapacity=max_capacity
        )
        logger.info(f"Registered scalable target: {min_capacity}-{max_capacity} instances")
        
        # Create scaling policy
        policy_name = f'{endpoint_name}-scaling-policy'
        
        autoscaling_client.put_scaling_policy(
            PolicyName=policy_name,
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': target_invocations,
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': 300,  # 5 minutes
                'ScaleInCooldown': 300    # 5 minutes
            }
        )
        logger.info(f"Created scaling policy: target {target_invocations} invocations/minute")
        
    except Exception as e:
        logger.error(f"Failed to set up auto-scaling: {e}")
        raise


def test_endpoint(predictor):
    """Test the deployed endpoint."""
    logger.info("Testing endpoint...")
    
    try:
        import numpy as np
        import base64
        
        # Create test data
        test_image = np.random.rand(3, 128, 128).astype(np.float32)
        
        # Prepare payload
        image_bytes = test_image.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            'image_data': image_b64,
            'shape': test_image.shape,
            'metadata': {'test': True}
        }
        
        # Set content types
        predictor.content_type = 'application/json'
        predictor.accept = 'application/json'
        
        # Make prediction
        start_time = time.time()
        result = predictor.predict(payload)
        inference_time = time.time() - start_time
        
        # Check result
        if result.get('success'):
            logger.info(f"✅ Endpoint test successful!")
            logger.info(f"Inference time: {inference_time:.2f}s")
            logger.info(f"Processing time: {result.get('processing_time', 'N/A')}s")
            
            if 'quality_metrics' in result:
                metrics = result['quality_metrics']
                logger.info("Quality metrics:")
                for key, value in metrics.items():
                    if key != 'enhanced_array':
                        logger.info(f"  {key}: {value}")
        else:
            logger.error(f"❌ Endpoint test failed: {result.get('error_message')}")
            
    except Exception as e:
        logger.error(f"❌ Endpoint test failed: {e}")


def setup_cloudwatch_alarms(endpoint_name):
    """Set up CloudWatch alarms for monitoring."""
    logger.info(f"Setting up CloudWatch alarms for: {endpoint_name}")
    
    cloudwatch = boto3.client('cloudwatch')
    
    # High error rate alarm
    try:
        cloudwatch.put_metric_alarm(
            AlarmName=f'{endpoint_name}-high-error-rate',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='ModelLatency',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Average',
            Threshold=30000.0,  # 30 seconds
            ActionsEnabled=True,
            AlarmDescription='High latency on SageMaker endpoint',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': 'AllTraffic'
                }
            ],
            Unit='Milliseconds'
        )
        logger.info("Created high latency alarm")
        
        # High invocation rate alarm
        cloudwatch.put_metric_alarm(
            AlarmName=f'{endpoint_name}-high-invocation-rate',
            ComparisonOperator='GreaterThanThreshold',
            EvaluationPeriods=2,
            MetricName='Invocations',
            Namespace='AWS/SageMaker',
            Period=300,
            Statistic='Sum',
            Threshold=1000.0,
            ActionsEnabled=True,
            AlarmDescription='High invocation rate on SageMaker endpoint',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': 'AllTraffic'
                }
            ]
        )
        logger.info("Created high invocation rate alarm")
        
    except Exception as e:
        logger.error(f"Failed to create CloudWatch alarms: {e}")


def main():
    """Main deployment function."""
    try:
        args = parse_args()
        
        logger.info("Starting endpoint deployment")
        logger.info(f"Model data: {args.model_data}")
        logger.info(f"Endpoint name: {args.endpoint_name}")
        logger.info(f"Instance type: {args.instance_type}")
        
        # Get execution role
        role = get_execution_role()
        logger.info(f"Using role: {role}")
        
        # Create model
        model = create_model(args.model_data, role)
        
        # Deploy endpoint
        predictor = deploy_endpoint(
            model=model,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            initial_instance_count=args.initial_instance_count,
            enable_data_capture=args.enable_data_capture,
            data_capture_percentage=args.data_capture_percentage
        )
        
        # Wait for deployment if requested
        if args.wait_for_deployment:
            logger.info("Waiting for deployment to complete...")
            predictor.wait_for_deployment()
            logger.info("✅ Deployment completed")
        
        # Set up auto-scaling if enabled
        if args.enable_autoscaling:
            setup_autoscaling(
                endpoint_name=args.endpoint_name,
                min_capacity=args.min_capacity,
                max_capacity=args.max_capacity,
                target_invocations=args.target_invocations
            )
        
        # Set up monitoring
        setup_cloudwatch_alarms(args.endpoint_name)
        
        # Test endpoint if requested
        if args.test_endpoint:
            test_endpoint(predictor)
        
        # Print deployment summary
        logger.info("🎉 Deployment completed successfully!")
        logger.info(f"Endpoint name: {args.endpoint_name}")
        logger.info(f"Endpoint URL: https://runtime.sagemaker.{boto3.Session().region_name}.amazonaws.com/endpoints/{args.endpoint_name}/invocations")
        
        # Save deployment info
        deployment_info = {
            'endpoint_name': args.endpoint_name,
            'model_data': args.model_data,
            'instance_type': args.instance_type,
            'initial_instance_count': args.initial_instance_count,
            'autoscaling_enabled': args.enable_autoscaling,
            'data_capture_enabled': args.enable_data_capture,
            'deployment_time': time.time()
        }
        
        with open(f'{args.endpoint_name}_deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"Deployment info saved to: {args.endpoint_name}_deployment_info.json")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == '__main__':
    main()