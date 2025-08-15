# Running Sigray ML Platform on AWS SageMaker

This guide shows how to run the Sigray Machine Learning Platform on AWS SageMaker for scalable training and inference.

## Overview

AWS SageMaker provides several ways to run your ML workloads:

1. **SageMaker Notebooks** - Interactive development and experimentation
2. **SageMaker Training Jobs** - Scalable model training
3. **SageMaker Processing Jobs** - Data preprocessing and batch inference
4. **SageMaker Endpoints** - Real-time inference
5. **SageMaker Batch Transform** - Batch inference

## Quick Start

### 1. SageMaker Notebook Instance

The easiest way to get started is with a SageMaker Notebook instance:

```python
# In a SageMaker notebook cell
!git clone https://github.com/tianzhuqin-argonne/Image_quality_enhancement.git
%cd Image_quality_enhancement
!pip install -e .

# Test the installation
from src.inference.api import ImageEnhancementAPI
print("✅ Sigray ML Platform installed successfully!")
```

### 2. Training on SageMaker

Use the provided training script for SageMaker training jobs:

```python
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create PyTorch estimator
estimator = PyTorch(
    entry_point='sagemaker_train.py',
    source_dir='sagemaker',
    role=role,
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='1.12.0',
    py_version='py38',
    hyperparameters={
        'epochs': 50,
        'batch-size': 16,
        'learning-rate': 1e-4
    }
)

# Start training
estimator.fit({'training': 's3://your-bucket/training-data/'})
```

### 3. Batch Inference

Process multiple images using SageMaker Batch Transform:

```python
from sagemaker.pytorch import PyTorchModel

# Create model from training job
model = PyTorchModel(
    model_data=estimator.model_data,
    role=role,
    entry_point='sagemaker_inference.py',
    source_dir='sagemaker',
    framework_version='1.12.0',
    py_version='py38'
)

# Create batch transform job
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    output_path='s3://your-bucket/enhanced-images/'
)

# Start batch transform
transformer.transform(
    data='s3://your-bucket/input-images/',
    content_type='application/x-tiff'
)
```

## Detailed Setup Instructions

### Prerequisites

1. AWS Account with SageMaker access
2. IAM role with SageMaker permissions
3. S3 bucket for data storage
4. (Optional) ECR repository for custom containers

### Environment Setup

#### Option 1: SageMaker Notebook Instance

1. **Create Notebook Instance**:
   - Instance type: `ml.t3.medium` (CPU) or `ml.p3.2xlarge` (GPU)
   - Volume size: 20GB+
   - IAM role with S3 and SageMaker permissions

2. **Install Platform**:
   ```bash
   # In terminal
   git clone https://github.com/tianzhuqin-argonne/Image_quality_enhancement.git
   cd Image_quality_enhancement
   pip install -e .
   ```

#### Option 2: SageMaker Studio

1. **Create SageMaker Domain**
2. **Launch Studio**
3. **Create new notebook** with PyTorch kernel
4. **Install platform** as shown above

### Data Preparation

Upload your training data to S3:

```bash
# Upload training data
aws s3 cp local_training_data/ s3://your-bucket/training-data/ --recursive

# Upload inference data
aws s3 cp local_inference_data/ s3://your-bucket/inference-data/ --recursive
```

## Usage Examples

### Interactive Development

See `sagemaker_notebook_example.ipynb` for a complete interactive example.

### Training Job

```python
# sagemaker_training_example.py
import sagemaker
from sagemaker.pytorch import PyTorch

def run_training_job():
    # Configuration
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
    
    # Hyperparameters
    hyperparameters = {
        'epochs': 100,
        'batch-size': 16,
        'learning-rate': 1e-4,
        'patch-size': '256,256',
        'device': 'cuda',
        'early-stopping-patience': 15
    }
    
    # Create estimator
    estimator = PyTorch(
        entry_point='sagemaker_train.py',
        source_dir='sagemaker',
        role=role,
        instance_type='ml.p3.2xlarge',
        instance_count=1,
        volume_size=100,  # GB
        max_run=86400,    # 24 hours
        framework_version='1.12.0',
        py_version='py38',
        hyperparameters=hyperparameters,
        environment={
            'PYTHONPATH': '/opt/ml/code'
        }
    )
    
    # Training data channels
    training_input = sagemaker.inputs.TrainingInput(
        s3_data='s3://your-bucket/training-data/',
        distribution='FullyReplicated'
    )
    
    # Start training
    estimator.fit({'training': training_input})
    
    return estimator

if __name__ == '__main__':
    estimator = run_training_job()
    print(f"Model artifacts: {estimator.model_data}")
```

### Real-time Endpoint

```python
# Deploy model as real-time endpoint
from sagemaker.pytorch import PyTorchModel

def deploy_endpoint(model_data_url):
    model = PyTorchModel(
        model_data=model_data_url,
        role=sagemaker.get_execution_role(),
        entry_point='sagemaker_inference.py',
        source_dir='sagemaker',
        framework_version='1.12.0',
        py_version='py38'
    )
    
    # Deploy endpoint
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.p3.2xlarge',
        endpoint_name='sigray-ml-endpoint'
    )
    
    return predictor

# Use endpoint
predictor = deploy_endpoint(model_data_url)

# Make prediction
import numpy as np
test_image = np.random.rand(5, 256, 256).astype(np.float32)
result = predictor.predict(test_image)
```

## Cost Optimization

### Instance Types

**Training:**
- `ml.p3.2xlarge` - GPU training (recommended)
- `ml.p3.8xlarge` - Multi-GPU training (large datasets)
- `ml.c5.4xlarge` - CPU training (budget option)

**Inference:**
- `ml.p3.2xlarge` - GPU inference (fast)
- `ml.c5.2xlarge` - CPU inference (cost-effective)
- `ml.inf1.xlarge` - AWS Inferentia (optimized inference)

### Spot Instances

Use spot instances for training to save up to 90%:

```python
estimator = PyTorch(
    # ... other parameters
    use_spot_instances=True,
    max_wait=7200,  # Wait time for spot instances
    checkpoint_s3_uri='s3://your-bucket/checkpoints/'
)
```

### Auto Scaling

Configure auto-scaling for endpoints:

```python
# Auto scaling configuration
predictor.update_endpoint(
    initial_instance_count=1,
    instance_type='ml.p3.2xlarge'
)

# Configure auto scaling
import boto3
autoscaling = boto3.client('application-autoscaling')

autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{predictor.endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)
```

## Monitoring and Logging

### CloudWatch Metrics

Monitor your SageMaker jobs:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Get training job metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='TrainingJobCPUUtilization',
    Dimensions=[
        {
            'Name': 'TrainingJobName',
            'Value': 'your-training-job-name'
        }
    ],
    StartTime=datetime.utcnow() - timedelta(hours=1),
    EndTime=datetime.utcnow(),
    Period=300,
    Statistics=['Average']
)
```

### Custom Metrics

Log custom metrics from your training script:

```python
# In sagemaker_train.py
import json
import os

def log_metrics(epoch, train_loss, val_loss):
    metrics = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # SageMaker will automatically capture these
    print(f'METRICS: {json.dumps(metrics)}')
```

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```python
   # Reduce batch size in hyperparameters
   hyperparameters = {
       'batch-size': 8,  # Reduced from 16
       'memory-limit': 4.0  # GB
   }
   ```

2. **S3 Access Issues**:
   ```python
   # Ensure IAM role has S3 permissions
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:ListBucket"
               ],
               "Resource": [
                   "arn:aws:s3:::your-bucket/*",
                   "arn:aws:s3:::your-bucket"
               ]
           }
       ]
   }
   ```

3. **Training Job Failures**:
   ```python
   # Check CloudWatch logs
   import boto3
   
   logs_client = boto3.client('logs')
   response = logs_client.describe_log_streams(
       logGroupName='/aws/sagemaker/TrainingJobs',
       logStreamNamePrefix='your-training-job-name'
   )
   ```

### Performance Optimization

1. **Use GPU instances** for training and inference
2. **Enable mixed precision** training
3. **Use appropriate instance types** for your workload
4. **Optimize data loading** with multiple workers
5. **Use spot instances** for cost savings

## Security Best Practices

1. **Use IAM roles** with minimal required permissions
2. **Enable VPC** for network isolation
3. **Encrypt data** at rest and in transit
4. **Use KMS keys** for encryption
5. **Enable CloudTrail** for audit logging

## Next Steps

1. Try the notebook example: `sagemaker_notebook_example.ipynb`
2. Run a training job with your data
3. Deploy a real-time endpoint
4. Set up batch processing pipeline
5. Monitor and optimize performance

For more advanced usage, see the individual script files in this directory.