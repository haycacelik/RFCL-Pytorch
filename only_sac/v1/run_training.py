#!/usr/bin/env python3
"""
Simple script to train SAC on LunarLander-v3
"""

from only_sac.v1.train import train_sac

if __name__ == "__main__":
    # Quick training configuration (for testing)
    config = {
        "env_name": "LunarLander-v3",
        "total_steps": 100000,  # Reduced for quick training
        "seed": 42,
        "eval_freq": 5000,
        "eval_episodes": 5,
        "save_freq": 25000,
        "log_freq": 1000,
        
        # SAC hyperparameters (tuned for LunarLander continuous)
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-5,  # Much lower learning rate for temperature
        "tau": 0.005,
        "gamma": 0.99,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "num_qs": 2,
        "learnable_temp": True,
        "grad_updates_per_step": 1,
        "actor_update_freq": 2,  # Update actor less frequently
        "seed_steps": 5000,  # Fewer random steps
        "hidden_dim": 256,
        "target_entropy": -2.0,  # Explicit target entropy for 2D action space
        "initial_alpha": 0.1,  # Lower initial temperature
    }
    
    print("Starting SAC training on LunarLander-v3")
    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    agent, metrics = train_sac(**config)
    print("Training completed!")
