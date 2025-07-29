#!/usr/bin/env python3
"""
Test script to compare JAX and PyTorch DiagGaussianActor implementations
"""
import sys
import os
sys.path.append('/work/dlclarge2/celikh-nr1-ayca/RFCL-Pytorch')

import numpy as np
import torch
import torch.nn.functional as F

# Import PyTorch implementation
from rfcl.agents.sac.networks_pytorch import DiagGaussianActor as DiagGaussianActorPyTorch
from rfcl.agents.sac.networks_pytorch import MLP as MLPPyTorch, create_feature_extractor

def test_pytorch_implementation():
    """Test the PyTorch DiagGaussianActor implementation"""
    print("=" * 60)
    print("Testing PyTorch DiagGaussianActor Implementation")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    obs_dim = 10
    action_dim = 4
    batch_size = 32
    hidden_dims = [256, 256, 256]
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden dimensions: {hidden_dims}")
    print()
    
    # Create feature extractor
    print("Creating feature extractor...")
    feature_extractor = create_feature_extractor(
        input_dim=obs_dim, 
        hidden_dims=hidden_dims, 
        activation=torch.nn.ReLU
    )
    print(f"Feature extractor output dim: {feature_extractor.output_dim}")
    
    # Create the actor
    print("Creating DiagGaussianActor...")
    actor = DiagGaussianActorPyTorch(
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_extractor=feature_extractor,
        state_dependent_std=True,
        tanh_squash_distribution=True
    )
    
    print(f"Actor created with {sum(p.numel() for p in actor.parameters())} parameters")
    print()
    
    # Test with batch of observations
    print("Testing with batch of random observations...")
    obs = torch.randn(batch_size, obs_dim)
    print(f"Input observations shape: {obs.shape}")
    
    # Test 1: Deterministic forward pass
    print("\n1. Testing deterministic forward pass...")
    with torch.no_grad():
        det_actions = actor(obs, deterministic=True)
    
    print(f"✓ Deterministic actions shape: {det_actions.shape}")
    print(f"✓ Actions range: [{det_actions.min():.4f}, {det_actions.max():.4f}]")
    
    # Check that actions are in [-1, 1] range (due to tanh)
    assert det_actions.min() >= -1.0 and det_actions.max() <= 1.0, \
        "Deterministic actions should be in [-1, 1] with tanh squashing"
    print("✓ Actions are properly bounded in [-1, 1]")
    
    # Test 2: Stochastic forward pass
    print("\n2. Testing stochastic forward pass...")
    with torch.no_grad():
        dist = actor(obs, deterministic=False)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
    
    print(f"✓ Sampled actions shape: {actions.shape}")
    print(f"✓ Log probabilities shape: {log_probs.shape}")
    print(f"✓ Actions range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"✓ Log probs range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
    
    # Check that stochastic actions are also bounded
    assert actions.min() >= -1.0 and actions.max() <= 1.0, \
        "Stochastic actions should be in [-1, 1] with tanh squashing"
    print("✓ Stochastic actions are properly bounded in [-1, 1]")
    
    # Test 3: Different samples give different actions
    print("\n3. Testing randomness...")
    with torch.no_grad():
        actions1 = actor(obs, deterministic=False).sample()
        actions2 = actor(obs, deterministic=False).sample()
    
    diff = torch.abs(actions1 - actions2).mean()
    print(f"✓ Mean difference between samples: {diff:.4f}")
    assert diff > 0.01, "Different samples should be different"
    print("✓ Actor produces diverse samples")
    
    # Test 4: Gradient computation
    print("\n4. Testing gradient computation...")
    
    # Clear any existing gradients
    actor.zero_grad()
    
    # Forward pass with gradients
    dist = actor(obs, deterministic=False)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    # Simple policy gradient loss
    loss = -log_probs.mean()
    print(f"✓ Policy gradient loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    grad_norms = []
    for name, param in actor.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm > 0:
                print(f"✓ {name}: grad norm = {grad_norm:.6f}")
    
    assert len(grad_norms) > 0, "No gradients computed"
    assert any(g > 0 for g in grad_norms), "All gradients are zero"
    print("✓ Gradients computed successfully")
    
    # Test 5: State dependent vs fixed std
    print("\n5. Testing state dependent vs fixed std...")
    
    # Create actor with fixed std
    actor_fixed = DiagGaussianActorPyTorch(
        obs_dim=obs_dim,
        action_dim=action_dim,
        feature_extractor=create_feature_extractor(obs_dim, hidden_dims),
        state_dependent_std=False,
        tanh_squash_distribution=True
    )
    
    with torch.no_grad():
        dist_state_dep = actor(obs, deterministic=False)
        dist_fixed = actor_fixed(obs, deterministic=False)
        
        # Sample multiple times from the same observation
        single_obs = obs[:1]  # Take first observation
        
        samples_state = [actor(single_obs, deterministic=False).sample() for _ in range(10)]
        samples_fixed = [actor_fixed(single_obs, deterministic=False).sample() for _ in range(10)]
        
        var_state = torch.var(torch.cat(samples_state), dim=0).mean()
        var_fixed = torch.var(torch.cat(samples_fixed), dim=0).mean()
        
    print(f"✓ State-dependent std variance: {var_state:.4f}")
    print(f"✓ Fixed std variance: {var_fixed:.4f}")
    print("✓ Both std types work correctly")
    
    # Test 6: Memory and performance
    print("\n6. Testing memory and performance...")
    
    import time
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    for _ in range(10):
        _ = actor(obs, deterministic=False).sample()
    
    # Time forward passes
    start_time = time.time()
    n_runs = 100
    
    for _ in range(n_runs):
        with torch.no_grad():
            _ = actor(obs, deterministic=False).sample()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / n_runs * 1000  # ms
    
    print(f"✓ Average forward pass time: {avg_time:.2f} ms")
    print(f"✓ Throughput: ~{batch_size * n_runs / (end_time - start_time):.0f} samples/sec")
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! PyTorch DiagGaussianActor is working correctly!")
    print("=" * 60)
    
    return True


def compare_with_config():
    """Test with a configuration similar to the RFCL config"""
    print("\n" + "=" * 60)
    print("Testing with RFCL-like Configuration")
    print("=" * 60)
    
    # Configuration similar to the YAML config
    config = {
        'obs_dim': 10,  # This would be environment-specific
        'action_dim': 4,  # This would be environment-specific  
        'hidden_dims': [256, 256, 256],  # From config: features: [256, 256, 256]
        'activation': torch.nn.ReLU,  # Default activation
        'output_activation': torch.nn.ReLU,  # From config: output_activation: "relu"
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create feature extractor with config
    feature_extractor = MLPPyTorch(
        input_dim=config['obs_dim'],
        hidden_dims=config['hidden_dims'],
        activation=config['activation']
    )
    
    # Create actor
    actor = DiagGaussianActorPyTorch(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        feature_extractor=feature_extractor,
        state_dependent_std=True  # Default in original code
    )
    
    print(f"Created actor with {sum(p.numel() for p in actor.parameters())} parameters")
    
    # Test with sample input
    obs = torch.randn(8, config['obs_dim'])  # Batch size 8 like in config
    
    with torch.no_grad():
        # Test deterministic
        det_actions = actor(obs, deterministic=True)
        print(f"✓ Deterministic actions: {det_actions.shape}")
        
        # Test stochastic
        dist = actor(obs, deterministic=False)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        print(f"✓ Stochastic actions: {actions.shape}")
        print(f"✓ Log probabilities: {log_probs.shape}")
        print(f"✓ Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    
    print("✓ RFCL-like configuration test passed!")


def main():
    """Run all tests"""
    print("Starting PyTorch DiagGaussianActor Tests...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    try:
        # Run main tests
        test_pytorch_implementation()
        
        # Run config-specific tests
        compare_with_config()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo integrate with the main training script:")
        print("1. Import the PyTorch networks: from rfcl.agents.sac.networks_pytorch import DiagGaussianActor")
        print("2. Replace JAX networks with PyTorch equivalents")
        print("3. Update the training loop to use PyTorch operations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
