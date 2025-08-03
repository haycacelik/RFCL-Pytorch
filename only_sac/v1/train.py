import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from only_sac.v1.sac_agent import SACAgent


def make_env(env_name="LunarLander-v3"):
    """Create and configure environment"""
    if "LunarLander" in env_name and "v3" in env_name:
        env = gym.make(env_name, continuous=True)
    else:
        env = gym.make(env_name)
    return env


def train_sac(
    env_name="LunarLander-v3",
    total_steps=500000,
    seed=42,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_freq=5000,
    eval_episodes=5,
    save_freq=50000,
    log_freq=1000,
    use_tensorboard=True,
    tensorboard_log_dir="runs",
    **sac_kwargs
):
    """Train SAC agent on LunarLander-v3"""
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environments
    env = make_env(env_name)
    eval_env = make_env(env_name)
    
    # Set environment seeds (gymnasium style)
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 1)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Environment: {env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    
    # Initialize agent
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        **sac_kwargs
    )
    
    # Initialize TensorBoard
    writer = None
    if use_tensorboard:
        log_dir = os.path.join(tensorboard_log_dir, f"{env_name}_seed{seed}")
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")
    
    # Create save directory
    save_dir = f"models/{env_name}_seed{seed}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training metrics
    metrics = defaultdict(list)
    step_metrics = defaultdict(list)
    
    # Training loop
    pbar = tqdm(total=total_steps, desc="Training SAC")
    
    while agent.total_env_steps < total_steps:
        # Perform training step
        update_info = agent.train_step(env)
        
        # Store metrics
        for k, v in update_info.items():
            step_metrics[k].append(v)
        
        # Evaluation
        if agent.total_env_steps % eval_freq == 0 and agent.total_env_steps > 0:
            print("step", agent.total_env_steps)
            try:
                print(update_info["entropy"])
            except:
                print("wait")

            eval_results = agent.evaluate(eval_env, eval_episodes)
            
            for k, v in eval_results.items():
                metrics[k].append(v)
            
            # Log episode metrics if available
            if agent.train_metrics["episode_reward"]:
                metrics["train_reward"].append(np.mean(agent.train_metrics["episode_reward"][-10:]))
                metrics["train_length"].append(np.mean(agent.train_metrics["episode_length"][-10:]))
            
            # Print progress
            if metrics["eval_reward"]:
                print(f"\nStep {agent.total_env_steps:>6}: "
                      f"Eval Reward: {eval_results['eval_reward']:>8.2f}, "
                      f"Train Reward: {metrics['train_reward'][-1] if metrics['train_reward'] else 0:>8.2f}, "
                      f"Alpha: {agent.log_alpha.exp().item():.4f}")
            
            # Log to TensorBoard
            if writer is not None:
                # Log evaluation metrics
                for k, v in eval_results.items():
                    writer.add_scalar(f"eval/{k}", v, agent.total_env_steps)
                
                # Log training metrics if available
                if metrics["train_reward"]:
                    writer.add_scalar("train/reward", metrics["train_reward"][-1], agent.total_env_steps)
                    writer.add_scalar("train/length", metrics["train_length"][-1], agent.total_env_steps)
                
                # Log recent update metrics
                for k, v in step_metrics.items():
                    if v:  # Only log if we have values
                        writer.add_scalar(f"update/{k}", np.mean(v[-100:]), agent.total_env_steps)
                
                # Log hyperparameters
                writer.add_scalar("hyperparams/alpha", agent.log_alpha.exp().item(), agent.total_env_steps)
                writer.add_scalar("hyperparams/buffer_size", len(agent.replay_buffer), agent.total_env_steps)
        
        # Save model
        if agent.total_env_steps % save_freq == 0 and agent.total_env_steps > 0:
            save_path = os.path.join(save_dir, f"model_{agent.total_env_steps}.pt")
            agent.save(save_path)
            print(f"Model saved to {save_path}")
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "Episodes": len(agent.train_metrics["episode_reward"]),
            "Alpha": f"{agent.log_alpha.exp().item():.4f}",
            "Buffer": len(agent.replay_buffer)
        })
    
    pbar.close()
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model.pt")
    agent.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_eval = agent.evaluate(eval_env, episodes=20)
    print(f"Final evaluation reward: {final_eval['eval_reward']:.2f}")
    
    # Log final evaluation to TensorBoard
    if writer is not None:
        writer.add_scalar("final/eval_reward", final_eval['eval_reward'], agent.total_env_steps)
        writer.close()
        print("TensorBoard logging completed.")
    
    # Plot training curves
    plot_training_curves(metrics, save_dir)
    
    env.close()
    eval_env.close()
    
    return agent, metrics


def plot_training_curves(metrics, save_dir):
    """Plot and save training curves"""
    if not metrics["eval_reward"]:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Evaluation reward
    axes[0, 0].plot(metrics["eval_reward"])
    axes[0, 0].set_title("Evaluation Reward")
    axes[0, 0].set_xlabel("Evaluation Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True)
    
    # Training reward (if available)
    if metrics["train_reward"]:
        axes[0, 1].plot(metrics["train_reward"])
        axes[0, 1].set_title("Training Reward")
        axes[0, 1].set_xlabel("Evaluation Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].grid(True)
    
    # Evaluation length
    axes[1, 0].plot(metrics["eval_length"])
    axes[1, 0].set_title("Evaluation Episode Length")
    axes[1, 0].set_xlabel("Evaluation Episode")
    axes[1, 0].set_ylabel("Length")
    axes[1, 0].grid(True)
    
    # Training length (if available)
    if metrics["train_length"]:
        axes[1, 1].plot(metrics["train_length"])
        axes[1, 1].set_title("Training Episode Length")
        axes[1, 1].set_xlabel("Evaluation Episode")
        axes[1, 1].set_ylabel("Length")
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"))
    plt.close()


def test_agent(model_path, env_name="LunarLander-v3", episodes=10, render=False):
    """Test a trained agent"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create environment
    env = make_env(env_name)
    if render:
        env = gym.wrappers.RecordVideo(env, "videos", force=True)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Create and load agent
    agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(model_path)
    
    # Test episodes
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            if render:
                env.render()
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            done = done or truncated  # Handle gymnasium's done/truncated split
            episode_reward += reward
        
        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
    
    print(f"\nAverage reward over {episodes} episodes: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on LunarLander-v3")
    parser.add_argument("--env", default="LunarLander-v3", help="Environment name")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--save-freq", type=int, default=50000, help="Model save frequency")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Use TensorBoard logging")
    parser.add_argument("--log-dir", default="runs", help="TensorBoard log directory")
    parser.add_argument("--test", type=str, help="Test model from path")
    parser.add_argument("--render", action="store_true", help="Render during testing")
    
    # SAC hyperparameters
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="Critic learning rate")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="Alpha learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient")
    parser.add_argument("--buffer-size", type=int, default=1000000, help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed-steps", type=int, default=10000, help="Random exploration steps")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.test:
        # Test mode
        test_agent(args.test, args.env, render=args.render)
    else:
        # Training mode
        sac_kwargs = {
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "alpha_lr": args.alpha_lr,
            "gamma": args.gamma,
            "tau": args.tau,
            "buffer_size": args.buffer_size,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "seed_steps": args.seed_steps,
        }
        
        agent, metrics = train_sac(
            env_name=args.env,
            total_steps=args.steps,
            seed=args.seed,
            device=device,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            use_tensorboard=args.tensorboard,
            tensorboard_log_dir=args.log_dir,
            **sac_kwargs
        )