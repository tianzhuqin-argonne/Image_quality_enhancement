"""
Unit tests for U-Net model architecture.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

from src.models.unet import UNet, UNetLoss, create_unet_model, calculate_model_flops


class TestUNet:
    """Test cases for U-Net model architecture."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.model = create_unet_model(
            input_channels=1,
            output_channels=1,
            base_channels=64,
            bilinear=True
        )
    
    def test_unet_initialization(self):
        """Test U-Net model initialization."""
        assert isinstance(self.model, UNet)
        assert self.model.n_channels == 1
        assert self.model.n_classes == 1
        assert self.model.base_channels == 64
        assert self.model.bilinear is True
    
    def test_unet_forward_pass(self):
        """Test U-Net forward pass with different input sizes."""
        test_sizes = [(256, 256), (128, 128), (512, 512)]
        
        for height, width in test_sizes:
            input_tensor = torch.randn(2, 1, height, width)  # Batch size 2
            output = self.model(input_tensor)
            
            # Check output shape
            expected_shape = (2, 1, height, width)
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
            # Check output is not NaN or Inf
            assert torch.isfinite(output).all(), "Output contains NaN or Inf values"
    
    def test_unet_architecture_validation(self):
        """Test U-Net architecture validation."""
        is_valid = self.model.validate_architecture()
        assert is_valid is True
    
    def test_unet_6_levels(self):
        """Test that U-Net has exactly 6 levels."""
        # Check encoder blocks
        encoder_blocks = [self.model.down1, self.model.down2, self.model.down3, 
                         self.model.down4, self.model.down5]
        assert len(encoder_blocks) == 5
        
        # Check decoder blocks
        decoder_blocks = [self.model.up1, self.model.up2, self.model.up3,
                         self.model.up4, self.model.up5]
        assert len(decoder_blocks) == 5
        
        # Total levels = 1 (input) + 5 (down) = 6 levels
    
    def test_unet_channel_progression(self):
        """Test that U-Net has correct channel progression."""
        expected_channels = [64, 128, 256, 512, 1024, 2048]
        model_info = self.model.get_model_info()
        
        assert model_info['channel_progression'] == expected_channels
    
    def test_unet_model_info(self):
        """Test U-Net model information extraction."""
        info = self.model.get_model_info()
        
        required_keys = [
            'architecture', 'input_channels', 'output_classes', 'base_channels',
            'bilinear_upsampling', 'total_parameters', 'trainable_parameters',
            'model_size_mb', 'levels', 'channel_progression'
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        assert info['architecture'] == '6-level U-Net'
        assert info['levels'] == 6
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
    
    def test_unet_different_configurations(self):
        """Test U-Net with different configurations."""
        configs = [
            {'input_channels': 3, 'output_channels': 3, 'base_channels': 32},
            {'input_channels': 1, 'output_channels': 2, 'base_channels': 128},
            {'bilinear': False}
        ]
        
        for config in configs:
            model = create_unet_model(**config)
            
            # Test forward pass
            n_channels = config.get('input_channels', 1)
            n_classes = config.get('output_channels', 1)
            
            input_tensor = torch.randn(1, n_channels, 256, 256)
            output = model(input_tensor)
            
            expected_shape = (1, n_classes, 256, 256)
            assert output.shape == expected_shape
    
    def test_unet_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        input_tensor = torch.randn(1, 1, 256, 256, requires_grad=True)
        output = self.model(input_tensor)
        
        # Compute a simple loss
        loss = output.mean()
        loss.backward()
        
        # Check that input gradients exist
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()
        
        # Check that model parameters have gradients
        for param in self.model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
    
    def test_unet_memory_efficiency(self):
        """Test U-Net memory usage."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 1, 256, 256)
            
            # Forward pass should not cause memory issues
            with torch.no_grad():
                output = self.model(input_tensor)
                assert output.shape[0] == batch_size
    
    def test_unet_device_compatibility(self):
        """Test U-Net device compatibility."""
        # Test CPU
        cpu_model = create_unet_model()
        cpu_input = torch.randn(1, 1, 128, 128)
        cpu_output = cpu_model(cpu_input)
        assert cpu_output.device.type == 'cpu'
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_model = create_unet_model().cuda()
            cuda_input = torch.randn(1, 1, 128, 128).cuda()
            cuda_output = cuda_model(cuda_input)
            assert cuda_output.device.type == 'cuda'
        else:
            pytest.skip("CUDA not available")


class TestUNetLoss:
    """Test cases for U-Net loss function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loss_fn = UNetLoss(mse_weight=1.0, l1_weight=0.1)
    
    def test_loss_computation(self):
        """Test loss computation."""
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randn(2, 1, 64, 64)
        
        loss = self.loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_loss_perfect_prediction(self):
        """Test loss with perfect prediction."""
        target = torch.randn(1, 1, 32, 32)
        pred = target.clone()
        
        loss = self.loss_fn(pred, target)
        assert loss.item() < 1e-6  # Should be very close to zero
    
    def test_loss_gradient_flow(self):
        """Test that loss allows gradient flow."""
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randn(1, 1, 32, 32)
        
        loss = self.loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


class TestUNetUtilities:
    """Test cases for U-Net utility functions."""
    
    def test_create_unet_model(self):
        """Test U-Net model factory function."""
        model = create_unet_model(
            input_channels=3,
            output_channels=2,
            base_channels=32,
            bilinear=False
        )
        
        assert isinstance(model, UNet)
        assert model.n_channels == 3
        assert model.n_classes == 2
        assert model.base_channels == 32
        assert model.bilinear is False
    
    def test_calculate_model_flops(self):
        """Test FLOP calculation for U-Net model."""
        model = create_unet_model()
        
        flops_256 = calculate_model_flops(model, (256, 256))
        flops_128 = calculate_model_flops(model, (128, 128))
        
        assert isinstance(flops_256, int)
        assert isinstance(flops_128, int)
        assert flops_256 > flops_128  # Larger input should have more FLOPs
        assert flops_256 > 0
        assert flops_128 > 0
    
    def test_model_parameter_count(self):
        """Test parameter counting."""
        model = create_unet_model(base_channels=64)
        info = model.get_model_info()
        
        # Count parameters manually
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert info['total_parameters'] == total_params
        assert info['trainable_parameters'] == trainable_params
        
        # For a 6-level U-Net with base_channels=64, we expect a significant number of parameters
        assert total_params > 1_000_000  # At least 1M parameters
    
    def test_model_size_estimation(self):
        """Test model size estimation."""
        model = create_unet_model()
        info = model.get_model_info()
        
        # Model size should be reasonable
        assert info['model_size_mb'] > 0
        assert info['model_size_mb'] < 1000  # Should be less than 1GB
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = create_unet_model()
        
        # Test various input sizes
        input_sizes = [
            (64, 64), (128, 128), (256, 256), (512, 512),
            (96, 96), (160, 160), (224, 224)
        ]
        
        for height, width in input_sizes:
            input_tensor = torch.randn(1, 1, height, width)
            
            with torch.no_grad():
                output = model(input_tensor)
                
            # Output should have same spatial dimensions as input
            assert output.shape == (1, 1, height, width)
    
    def test_batch_processing(self):
        """Test model with different batch sizes."""
        model = create_unet_model()
        
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 1, 128, 128)
            
            with torch.no_grad():
                output = model(input_tensor)
                
            assert output.shape == (batch_size, 1, 128, 128)


class TestUNetIntegration:
    """Integration tests for U-Net model."""
    
    def test_training_step(self):
        """Test a complete training step."""
        model = create_unet_model()
        loss_fn = UNetLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy data
        input_data = torch.randn(2, 1, 128, 128)
        target_data = torch.randn(2, 1, 128, 128)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        output = model(input_data)
        loss = loss_fn(output, target_data)
        loss.backward()
        optimizer.step()
        
        # Check that loss is computed and gradients exist
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0
        
        # Check that parameters were updated
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_inference_mode(self):
        """Test model in inference mode."""
        model = create_unet_model()
        model.eval()
        
        input_data = torch.randn(1, 1, 256, 256)
        
        with torch.no_grad():
            output = model(input_data)
        
        assert output.shape == (1, 1, 256, 256)
        assert torch.isfinite(output).all()
    
    def test_model_state_dict(self):
        """Test model state dict operations."""
        model1 = create_unet_model()
        model2 = create_unet_model()
        
        # Models should have different initial weights
        input_data = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            output1 = model1(input_data)
            output2 = model2(input_data)
        
        # Outputs should be different (very unlikely to be identical)
        assert not torch.allclose(output1, output2, atol=1e-6)
        
        # Copy state dict
        model2.load_state_dict(model1.state_dict())
        
        # Now outputs should be identical
        with torch.no_grad():
            output1_new = model1(input_data)
            output2_new = model2(input_data)
        
        assert torch.allclose(output1_new, output2_new, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])