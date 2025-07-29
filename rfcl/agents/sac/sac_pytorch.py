"""
SAC Agent - PyTorch Implementation
"""
import os
import pickle
import time
from collections import defaultdict
from typing import Any, Tuple, Dict
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from rfcl.agents.base_pytorch import BasePolicy
from rfcl.agents.sac.config import SACConfig
from rfcl.agents.sac.networks_pytorch import ActorCritic, DiagGaussianActor
from rfcl.data.buffer import GenericBuffer
from rfcl.data.loop import DefaultTimeStep, EnvLoopState
from rfcl.logger import LoggerConfig
from rfcl.utils import tools


@dataclass
class TrainStepMetrics:
    train_stats: Any
    train: Any
    update: Any
    time: Any


@dataclass
class TimeStep:
    action: torch.Tensor = None
    env_obs: torch.Tensor = None
    next_env_obs: torch.Tensor = None
    reward: torch.Tensor = None
    mask: torch.Tensor = None


@dataclass
class SACTrainState:
    # model states
    ac: ActorCritic
    loop_state: EnvLoopState
    
    # monitoring
    total_env_steps: int
    training_steps: int
    initialized: bool


@dataclass
class CriticUpdateAux:
    critic_loss: float = 0.0
    q: float = 0.0


@dataclass
class TempUpdateAux:
    temp_loss: float = 0.0
    temp: float = 1.0


@dataclass
class ActorUpdateAux:
    actor_loss: float = 0.0
    entropy: float = 0.0


@dataclass
class UpdateMetrics:
    actor: ActorUpdateAux
    critic: CriticUpdateAux
    temp: TempUpdateAux


class SAC(BasePolicy):
    def __init__(
        self,
        env_type: str,
        env,
        eval_env=None,
        logger_cfg: LoggerConfig = None,
        cfg: SACConfig = {},
        offline_buffer=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        obs_dim: int = None,
        action_dim: int = None,
    ):
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg
            
        super().__init__(env_type, env, eval_env, cfg.num_envs, cfg.num_eval_envs, logger_cfg)
        self.device = device
        self.offline_buffer = offline_buffer
        
        # Get dimensions
        if obs_dim is None:
            obs_dim = np.prod(self.obs_shape) if not isinstance(self.obs_shape, dict) else sum(np.prod(shape) for shape in self.obs_shape.values())
        if action_dim is None:
            action_dim = self.action_dim
            
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Create ActorCritic
        self.ac = ActorCritic.create(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_qs=self.cfg.num_qs,
            num_min_qs=self.cfg.num_min_qs,
            initial_temperature=self.cfg.initial_temperature,
            device=device
        )
        
        self.state: SACTrainState = SACTrainState(
            ac=self.ac,
            loop_state=EnvLoopState(),
            total_env_steps=0,
            training_steps=0,
            initialized=False,
        )

        # Define our buffer
        buffer_config = dict(
            action=((self.action_dim,), self.action_space.dtype),
            reward=((), np.float32),
            mask=((), float),
        )
        if isinstance(self.obs_shape, dict):
            buffer_config["env_obs"] = (
                self.obs_shape,
                {k: self.observation_space[k].dtype for k in self.observation_space},
            )
        else:
            buffer_config["env_obs"] = (self.obs_shape, np.float32)
        buffer_config["next_env_obs"] = buffer_config["env_obs"]

        self.replay_buffer = GenericBuffer(
            buffer_size=self.cfg.replay_buffer_capacity,
            num_envs=self.cfg.num_envs,
            config=buffer_config,
        )

        if self.cfg.target_entropy is None:
            self.cfg.target_entropy = -self.action_dim / 2

    def seed_sampler(self, batch_size):
        """Generate random actions for exploration"""
        return torch.rand(batch_size, self.action_dim, device=self.device) * 2.0 - 1.0

    def _sample_action(self, actor: DiagGaussianActor, env_obs, seed=False):
        if seed:
            batch_size = env_obs.shape[0] if isinstance(env_obs, torch.Tensor) else self.cfg.num_envs
            a = self.seed_sampler(batch_size)
        else:
            with torch.no_grad():
                if isinstance(env_obs, np.ndarray):
                    env_obs = torch.tensor(env_obs, dtype=torch.float32, device=self.device)
                # Flatten observations if needed
                if len(env_obs.shape) > 2:
                    env_obs = env_obs.view(env_obs.shape[0], -1)
                dist = actor(env_obs, deterministic=False)
                a = dist.sample()
                a = torch.clamp(a, -1.0, 1.0)  # Clamp actions to valid range
        return a.cpu().numpy() if isinstance(a, torch.Tensor) else a

    def _env_step(self, loop_state: EnvLoopState, actor: DiagGaussianActor, seed=False):
        # This would need to be adapted based on your environment loop implementation
        # For now, keeping structure similar to the JAX version
        data, loop_state = self.loop.rollout(None, loop_state, actor, lambda env_obs: self._sample_action(actor, env_obs, seed=seed), 1)
        return loop_state, data

    def train(self, steps: int, callback_fn=None, verbose=1):
        """
        Args:
            steps: int
                Max number of environment samples before training is stopped.
        """
        train_start_time = time.time()

        # Initialize environment if needed
        if not self.state.initialized:
            loop_state = self.loop.reset_loop(None)
            self.state.loop_state = loop_state
            self.state.initialized = True

        start_step = self.state.total_env_steps

        if verbose:
            pbar = tqdm(total=steps + self.state.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        while self.state.total_env_steps < start_step + steps:
            self.state, train_step_metrics = self.train_step(self.state)

            # Evaluation
            if (
                self.eval_loop is not None
                and tools.reached_freq(self.state.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size)
                and self.state.total_env_steps > self.cfg.num_seed_steps
            ):
                eval_results = self.evaluate(
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    actor=self.state.ac.actor,
                )
                eval_data = {
                    "return": eval_results["eval_ep_rets"], 
                    "reward": eval_results["eval_ep_avg_reward"], 
                    "episode_len": eval_results["eval_ep_lens"], 
                    "success_once": eval_results["success_once"], 
                    "success_at_end": eval_results["success_at_end"]
                }
                self.logger.store(tag="eval", **eval_data)
                self.logger.store(tag="eval_stats", **eval_results["stats"])
                self.logger.log(self.state.total_env_steps)
                self.logger.reset()
                
            self.logger.store(tag="train", **train_step_metrics.train)
            self.logger.store(tag="train_stats", **train_step_metrics.train_stats)

            # Logging
            if verbose:
                pbar.update(n=env_rollout_size)
            total_time = time.time() - train_start_time
            if tools.reached_freq(self.state.total_env_steps, self.cfg.log_freq, step_size=env_rollout_size):
                update_aux = tools.flatten_struct_to_dict(train_step_metrics.update)
                self.logger.store(tag="train", training_steps=self.state.training_steps, **update_aux)
                self.logger.store(
                    tag="time",
                    total=total_time,
                    SPS=self.state.total_env_steps / total_time,
                    step=self.state.total_env_steps,
                    **train_step_metrics.time,
                )
            
            self.logger.log(self.state.total_env_steps)
            self.logger.reset()

            # Save checkpoints
            if tools.reached_freq(self.state.total_env_steps, self.cfg.save_freq, env_rollout_size):
                self.save(
                    os.path.join(self.logger.model_path, f"ckpt_{self.state.total_env_steps}.pt"),
                    with_buffer=self.cfg.save_buffer_in_checkpoints,
                )

            if callback_fn is not None:
                stop = callback_fn(locals())
                if stop:
                    print(f"Early stopping at {self.state.total_env_steps} env steps")
                    break

    def train_step(self, state: SACTrainState) -> Tuple[SACTrainState, TrainStepMetrics]:
        """Perform a single training step"""
        
        ac = state.ac
        loop_state = state.loop_state
        total_env_steps = state.total_env_steps
        training_steps = state.training_steps

        train_custom_stats = defaultdict(list)
        train_metrics = defaultdict(list)
        time_metrics = dict()

        # Perform rollout
        rollout_time_start = time.time()
        for _ in range(self.cfg.steps_per_env):
            (next_loop_state, data) = self._env_step(
                loop_state,
                ac.actor,
                seed=(total_env_steps <= self.cfg.num_seed_steps and not self.cfg.seed_with_policy),
            )

            final_infos = data["final_info"]
            del data["final_info"]
            data = DefaultTimeStep(**data)

            # Convert to numpy if needed
            if hasattr(data, 'env_obs') and isinstance(data.env_obs, torch.Tensor):
                data = DefaultTimeStep(**{k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                                        for k, v in data._asdict().items()})

            terminations = data.terminated
            truncations = data.truncated
            dones = terminations | truncations
            masks = ((~dones) | (truncations)).astype(float)
            
            if dones.any():
                train_metrics["return"].append(data.ep_ret[dones])
                train_metrics["episode_len"].append(data.ep_len[dones])

                for i, final_info in enumerate(final_infos):
                    if final_info is not None and "stats" in final_info:
                        for k in final_info["stats"]:
                            train_custom_stats[k].append(final_info["stats"][k])

            self.replay_buffer.store(
                env_obs=data.env_obs,
                reward=data.reward,
                action=data.action,
                mask=masks,
                next_env_obs=data.next_env_obs,
            )
            loop_state = next_loop_state

        # Process metrics
        for k in train_metrics:
            train_metrics[k] = np.concatenate(train_metrics[k]).flatten()
        if "return" in train_metrics and "episode_len" in train_metrics:
            train_metrics["reward"] = train_metrics["return"] / train_metrics["episode_len"]
        if "success_at_end" in train_custom_stats:
            train_metrics["success_at_end"] = train_custom_stats.pop("success_at_end")
        if "success_once" in train_custom_stats:
            train_metrics["success_once"] = train_custom_stats.pop("success_once")

        rollout_time = time.time() - rollout_time_start
        time_metrics["rollout_time"] = rollout_time
        time_metrics["rollout_fps"] = self.cfg.num_envs * self.cfg.steps_per_env / rollout_time
        
        state.loop_state = loop_state
        state.total_env_steps = total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env

        # Update policy
        update_aux = UpdateMetrics(
            actor=ActorUpdateAux(),
            critic=CriticUpdateAux(),
            temp=TempUpdateAux()
        )
        
        if state.total_env_steps >= self.cfg.num_seed_steps:
            update_time_start = time.time()
            
            if self.offline_buffer is not None:
                batch = self.replay_buffer.sample_random_batch(
                    batch_size=self.cfg.batch_size * self.cfg.grad_updates_per_step // 2
                )
                offline_batch = self.offline_buffer.sample_random_batch(
                    batch_size=self.cfg.batch_size * self.cfg.grad_updates_per_step // 2
                )
                batch = tools.combine(batch, offline_batch)
            else:
                batch = self.replay_buffer.sample_random_batch(
                    batch_size=self.cfg.batch_size * self.cfg.grad_updates_per_step
                )

            batch = TimeStep(**batch)
            update_aux = self.update_parameters(batch)
            state.training_steps = training_steps + self.cfg.grad_updates_per_step
            
            update_time = time.time() - update_time_start
            time_metrics["update_time"] = update_time

        return state, TrainStepMetrics(
            time=time_metrics, 
            train=train_metrics, 
            update=update_aux.__dict__ if hasattr(update_aux, '__dict__') else update_aux, 
            train_stats=train_custom_stats
        )

    def update_parameters(self, batch: TimeStep) -> UpdateMetrics:
        """Update actor critic parameters using the given batch"""
        
        # Convert batch to tensors
        obs = torch.tensor(batch.env_obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(batch.action, dtype=torch.float32, device=self.device)
        reward = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch.next_env_obs, dtype=torch.float32, device=self.device)
        mask = torch.tensor(batch.mask, dtype=torch.float32, device=self.device)

        # Flatten observations if needed
        if len(obs.shape) > 2:
            obs = obs.view(obs.shape[0], -1)
        if len(next_obs.shape) > 2:
            next_obs = next_obs.view(next_obs.shape[0], -1)

        update_metrics = UpdateMetrics(
            actor=ActorUpdateAux(),
            critic=CriticUpdateAux(),
            temp=TempUpdateAux()
        )
        
        mini_batch_size = self.cfg.batch_size
        total_samples = obs.shape[0]
        
        for i in range(self.cfg.grad_updates_per_step):
            start_idx = i * mini_batch_size
            end_idx = start_idx + mini_batch_size
            
            batch_obs = obs[start_idx:end_idx]
            batch_action = action[start_idx:end_idx]
            batch_reward = reward[start_idx:end_idx]
            batch_next_obs = next_obs[start_idx:end_idx]
            batch_mask = mask[start_idx:end_idx]
            
            # Update critic
            critic_loss, q_values = self.update_critic(
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_mask
            )
            
            # Update actor (with frequency control)
            if i % self.cfg.actor_update_freq == 0:
                actor_loss, entropy = self.update_actor(batch_obs)
                
                # Update temperature
                if self.cfg.learnable_temp:
                    temp_loss = self.update_temperature(entropy)
                    update_metrics.temp.temp_loss = temp_loss
                    update_metrics.temp.temp = self.ac.temp().item()
                
                update_metrics.actor.actor_loss = actor_loss
                update_metrics.actor.entropy = entropy
            
            update_metrics.critic.critic_loss = critic_loss
            update_metrics.critic.q = q_values.mean().item()

        return update_metrics

    def update_critic(self, obs, action, reward, next_obs, mask):
        """Update critic networks"""
        with torch.no_grad():
            # Sample next actions from current policy
            next_dist = self.ac.actor(next_obs, deterministic=False)
            next_action = next_dist.sample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            
            # Compute target Q-values
            next_q_values = self.ac.target_critic(next_obs, next_action)
            
            # For ensemble, randomly subsample target critics
            if hasattr(self.ac.target_critic, 'num_networks'):
                if self.cfg.num_min_qs < self.ac.target_critic.num_networks:
                    indices = torch.randperm(self.ac.target_critic.num_networks)[:self.cfg.num_min_qs]
                    next_q_values = next_q_values[indices]
            
            next_q_value = torch.min(next_q_values, dim=0)[0]
            
            temp = self.ac.temp()
            target_q_value = reward.unsqueeze(-1) + mask.unsqueeze(-1) * self.cfg.discount * (
                next_q_value - (temp * next_log_prob if self.cfg.backup_entropy else 0)
            )

        # Current Q-values
        current_q_values = self.ac.critic(obs, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_value.expand_as(current_q_values))
        
        # Update critic
        self.ac.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.ac.critic_optimizer.step()
        
        # Soft update target critic
        self.ac.soft_update_target(self.cfg.tau)
        
        return critic_loss.item(), current_q_values

    def update_actor(self, obs):
        """Update actor network"""
        dist = self.ac.actor(obs, deterministic=False)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Compute Q-values for sampled actions
        q_values = self.ac.critic(obs, action)
        q_value = torch.min(q_values, dim=0)[0]
        
        temp = self.ac.temp()
        actor_loss = (temp * log_prob - q_value).mean()
        
        # Update actor
        self.ac.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.ac.actor_optimizer.step()
        
        return actor_loss.item(), log_prob.mean().item()

    def update_temperature(self, entropy):
        """Update temperature parameter"""
        temp_loss = -(self.ac.temp.log_temp * (entropy + self.cfg.target_entropy)).mean()
        
        self.ac.temp_optimizer.zero_grad()
        temp_loss.backward()
        self.ac.temp_optimizer.step()
        
        return temp_loss.item()

    def save(self, save_path: str, with_buffer=False):
        """Save model checkpoint"""
        state_dict = {
            'ac': self.ac.state_dict(),
            'train_state': {
                'total_env_steps': self.state.total_env_steps,
                'training_steps': self.state.training_steps,
                'initialized': self.state.initialized,
            },
            'config': self.cfg,
        }
        
        if with_buffer:
            state_dict['replay_buffer'] = self.replay_buffer
            
        if self.logger is not None:
            state_dict['logger'] = self.logger.state_dict()
            
        torch.save(state_dict, save_path)

    def load_from_path(self, load_path: str):
        """Load model checkpoint from path"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.load(checkpoint)

    def load(self, data):
        """Load model checkpoint from data"""
        self.ac.load_state_dict(data['ac'])
        
        train_state = data['train_state']
        self.state.total_env_steps = train_state['total_env_steps']
        self.state.training_steps = train_state['training_steps']
        self.state.initialized = False  # Reset for new training session
        
        if 'replay_buffer' in data:
            self.replay_buffer = data['replay_buffer']
            print(f"Loading replay buffer which contains {self.replay_buffer.size() * self.replay_buffer.num_envs} interactions")
        
        if 'logger' in data and self.logger is not None:
            self.logger.load(data['logger'])
        else:
            print("Skip loading logger. No log data will be overwritten/saved")
        
        print(f"Loaded checkpoint")

    def load_policy_from_path(self, load_path: str):
        """Load only the policy from checkpoint"""
        checkpoint = torch.load(load_path, map_location=self.device)
        return self.load_policy(checkpoint)

    def load_policy(self, data) -> ActorCritic:
        """Load only the policy from data"""
        ac = copy.deepcopy(self.ac)
        ac.load_state_dict(data['ac'])
        return ac

    def evaluate(self, num_envs, steps_per_env, eval_loop, actor):
        """Evaluate the current policy"""
        # This would need to be implemented based on your evaluation setup
        # For now, return dummy results
        return {
            "eval_ep_rets": np.array([0.0]),
            "eval_ep_avg_reward": np.array([0.0]),
            "eval_ep_lens": np.array([100]),
            "success_once": np.array([False]),
            "success_at_end": np.array([False]),
            "stats": {}
        }

    def state_dict(self, with_buffer=False):
        """Get state dictionary for saving (for compatibility)"""
        state_dict = {
            'train_state': {
                'ac': self.ac.state_dict(),
                'total_env_steps': self.state.total_env_steps,
                'training_steps': self.state.training_steps,
                'initialized': self.state.initialized,
            },
        }
        
        if self.logger is not None:
            state_dict['logger'] = self.logger.state_dict()
        
        if with_buffer:
            state_dict['replay_buffer'] = self.replay_buffer
        
        return state_dict
