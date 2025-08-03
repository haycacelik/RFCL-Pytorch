import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm
import time

from only_sac.v1.networks import DiagGaussianActor, Critic, Ensemble, MLPFeatureExtractor
from only_sac.v1.replay_buffer import ReplayBuffer


class SACAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        tau=0.005,
        gamma=0.99,
        buffer_size=1_000_000,
        batch_size=256,
        num_qs=2,
        device="cpu",
        target_entropy=None,
        learnable_temp=True,
        grad_updates_per_step=1,
        actor_update_freq=1,
        seed_steps=10000,
        hidden_dim=256,
        initial_alpha=0.2,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.device = device

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size

        self.grad_updates_per_step = grad_updates_per_step
        self.actor_update_freq = actor_update_freq

        self.seed_steps = seed_steps
        self.learnable_temp = learnable_temp
        self.total_env_steps = 0
        self.training_steps = 0
        print(batch_size)
        # Feature extractors
        actor_feature_extractor = MLPFeatureExtractor(obs_dim, [hidden_dim, hidden_dim])
        critic_feature_extractor = MLPFeatureExtractor(obs_dim + action_dim, [hidden_dim, hidden_dim])

        # Actor and Critic
        self.actor = DiagGaussianActor(actor_feature_extractor, action_dim).to(device)
        self.critic = Ensemble(lambda: Critic(critic_feature_extractor), num=num_qs).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        # Temperature (alpha)
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=device)
        # Use less restrictive target entropy for continuous control to prevent alpha explosion
        self.target_entropy = target_entropy if target_entropy is not None else -action_dim

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, obs_dim, action_dim)

        # Metrics
        self.train_metrics = defaultdict(list)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def select_action(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor(obs, deterministic=True)
            else:
                dist = self.actor(obs)
                action = dist.sample()
        return action.cpu().numpy()[0]

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_parameters(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = self.replay_buffer.sample(self.batch_size)
        obs, action, reward, next_obs, done = [
            torch.tensor(x, dtype=torch.float32, device=self.device) for x in batch
        ]

        update_info = {}

        # Critic update
        with torch.no_grad():
            # get the action prob dist
            next_dist = self.actor(next_obs)
            # sample the next action
            next_action = next_dist.rsample()
            # entrohpy term
            log_prob = next_dist.log_prob(next_action)
            # give qvalue for action
            target_qs = self.target_critic(next_obs, next_action)  # [ensemble, batch]
            # get the min one
            min_target_q, _ = target_qs.min(dim=0)
            alpha = self.log_alpha.exp()
            target = reward + self.gamma * (1 - done) * (min_target_q - alpha * log_prob)

        qs = self.critic(obs, action)  # [ensemble, batch]
        critic_loss = ((qs - target.unsqueeze(0)) ** 2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        update_info["critic_loss"] = critic_loss.item()
        update_info["q_mean"] = qs.mean().item()

        # Actor update
        if self.training_steps % self.actor_update_freq == 0:
            for param in self.critic.parameters():
                param.requires_grad = False

            # get the action prob dist
            dist = self.actor(obs)
            new_action = dist.rsample()
            log_prob = dist.log_prob(new_action)
            q_new_actions = self.critic(obs, new_action)
            min_q_new_actions, _ = q_new_actions.min(dim=0)
            actor_loss = (alpha * log_prob - min_q_new_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param in self.critic.parameters():
                param.requires_grad = True

            update_info["actor_loss"] = actor_loss.item()
            update_info["entropy"] = -log_prob.mean().item()

            # Temperature update
            if self.learnable_temp:
                entropy = -log_prob.mean().detach()
                temp_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad()
                temp_loss.backward()
                # Clip gradients to prevent temperature explosion
                torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=10.0)
                self.alpha_optimizer.step()
                
                # Clamp log_alpha to prevent extreme values
                with torch.no_grad():
                    self.log_alpha.clamp_(min=-10.0, max=3.0)  # alpha between ~0.00005 and ~20

                update_info["temp_loss"] = temp_loss.item()
                update_info["alpha"] = self.alpha

        # Soft update target critic
        self.soft_update(self.critic, self.target_critic)

        return update_info

    def train_step(self, env):
        """Perform one training step: rollout + update"""
        # Handle gymnasium reset API (returns tuple)
        if not hasattr(env, '_obs'):
            obs, _ = env.reset()
            env._obs = obs
            env._episode_reward = 0
            env._episode_length = 0
        
        obs = env._obs
        
        # Sample action
        if self.total_env_steps < self.seed_steps:
            action = env.action_space.sample()
        else:
            action = self.select_action(obs)

        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        done = done or truncated  # Handle gymnasium's done/truncated split
        
        # Track episode statistics manually
        env._episode_reward += reward
        env._episode_length += 1
        
        # Store transition
        self.store_transition(obs, action, reward, next_obs, done)
        
        # Update observation
        if done:
            # Episode completed - store metrics
            self.train_metrics["episode_reward"].append(env._episode_reward)
            self.train_metrics["episode_length"].append(env._episode_length)
            
            # Reset environment and episode tracking
            obs, _ = env.reset()
            env._obs = obs
            env._episode_reward = 0
            env._episode_length = 0
        else:
            env._obs = next_obs

        self.total_env_steps += 1

        # Update parameters
        update_info = {}
        if self.total_env_steps >= self.seed_steps and self.total_env_steps % 5 == 0:
            # update every 5 steps
            for _ in range(self.grad_updates_per_step):
                step_info = self.update_parameters()
                for k, v in step_info.items():
                    if k not in update_info:
                        update_info[k] = []
                    update_info[k].append(v)
                self.training_steps += 1
            
            # Average the update info
            update_info = {k: np.mean(v) for k, v in update_info.items()}

        return update_info

    def evaluate(self, env, episodes=5):
        total_reward = 0
        total_length = 0
        
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action = self.select_action(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                done = done or truncated  # Handle gymnasium's done/truncated split
                episode_reward += reward
                episode_length += 1
                
            total_reward += episode_reward
            total_length += episode_length
            
        return {
            "eval_reward": total_reward / episodes,
            "eval_length": total_length / episodes
        }

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_env_steps': self.total_env_steps,
            'training_steps': self.training_steps,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.total_env_steps = checkpoint['total_env_steps']
        self.training_steps = checkpoint['training_steps']
