"""
Models for SAC - PyTorch Implementation
"""
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Type, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from dataclasses import dataclass


def default_init(scale: Optional[float] = np.sqrt(2)):
    """Default orthogonal initialization"""
    def init_fn(tensor):
        return nn.init.orthogonal_(tensor, gain=scale)
    return init_fn


class MLP(nn.Module):
    """Multi-layer perceptron feature extractor"""
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [256, 256], 
        activation: nn.Module = nn.ReLU,
        output_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim if output_dim is not None else prev_dim
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            default_init()(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class Ensemble(nn.Module):
    """Ensemble of networks"""
    def __init__(self, network_class: Type[nn.Module], num_networks: int, *args, **kwargs):
        super().__init__()
        self.num_networks = num_networks
        self.networks = nn.ModuleList([
            network_class(*args, **kwargs) for _ in range(num_networks)
        ])
    
    def forward(self, *args, **kwargs):
        outputs = [net(*args, **kwargs) for net in self.networks]
        return torch.stack(outputs, dim=0)  # Shape: (num_networks, batch_size, ...)


class Critic(nn.Module):
    """Q-value critic network"""
    def __init__(
        self, 
        obs_dim: int, 
        action_dim: int, 
        feature_extractor: nn.Module = None,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU
    ):
        super().__init__()
        
        if feature_extractor is None:
            self.feature_extractor = MLP(
                input_dim=obs_dim + action_dim,
                hidden_dims=hidden_dims,
                activation=activation
            )
        else:
            self.feature_extractor = feature_extractor
        
        self.value_head = nn.Linear(self.feature_extractor.output_dim, 1)
        
        # Initialize value head
        default_init()(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        features = self.feature_extractor(x)
        value = self.value_head(features)
        return value.squeeze(-1)


class DiagGaussianActor(nn.Module):
    """Diagonal Gaussian actor with optional tanh squashing"""
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_extractor: nn.Module = None,
        hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU,
        tanh_squash_distribution: bool = True,
        state_dependent_std: bool = True,
        log_std_range: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.tanh_squash_distribution = tanh_squash_distribution
        self.state_dependent_std = state_dependent_std
        self.log_std_range = log_std_range
        
        if feature_extractor is None:
            self.feature_extractor = MLP(
                input_dim=obs_dim,
                hidden_dims=hidden_dims,
                activation=activation
            )
        else:
            self.feature_extractor = feature_extractor
        
        # Action mean head
        self.action_head = nn.Linear(self.feature_extractor.output_dim, action_dim)
        default_init(1.0)(self.action_head.weight)
        nn.init.constant_(self.action_head.bias, 0)
        
        # Log std head or parameter
        if self.state_dependent_std:
            self.log_std_head = nn.Linear(self.feature_extractor.output_dim, action_dim)
            default_init(1.0)(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, 0)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.feature_extractor(obs)
        mean = self.action_head(features)
        
        if not self.tanh_squash_distribution:
            mean = torch.tanh(mean)
        
        if deterministic:
            return torch.tanh(mean)
        
        # Compute log std
        if self.state_dependent_std:
            log_std = self.log_std_head(features)
            log_std = torch.tanh(log_std)
        else:
            log_std = self.log_std.expand_as(mean)
        
        # Scale log_std to the specified range
        log_std = self.log_std_range[0] + 0.5 * (
            self.log_std_range[1] - self.log_std_range[0]
        ) * (log_std + 1)
        
        std = torch.exp(log_std)
        
        # Create distribution
        normal_dist = distributions.Normal(mean, std)
        
        if self.tanh_squash_distribution:
            # Apply tanh squashing using TransformedDistribution
            tanh_transform = distributions.TanhTransform(cache_size=1)
            dist = distributions.TransformedDistribution(normal_dist, [tanh_transform])
        else:
            dist = distributions.Independent(normal_dist, 1)
        
        return dist


class Temperature(nn.Module):
    """Temperature parameter for SAC"""
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(np.log(initial_temperature)))
    
    def forward(self):
        return torch.exp(self.log_temp)


@dataclass
class ActorCritic:
    """Actor-Critic model container"""
    actor: nn.Module
    critic: nn.Module
    target_critic: nn.Module
    temp: nn.Module
    
    # Optimizers
    actor_optimizer: torch.optim.Optimizer
    critic_optimizer: torch.optim.Optimizer
    temp_optimizer: torch.optim.Optimizer
    
    @classmethod
    def create(
        cls,
        obs_dim: int,
        action_dim: int,
        actor_hidden_dims: list = [256, 256],
        critic_hidden_dims: list = [256, 256],
        activation: nn.Module = nn.ReLU,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        initial_temperature: float = 1.0,
        num_qs: int = 10,
        num_min_qs: int = 2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> "ActorCritic":
        
        # Create actor
        actor = DiagGaussianActor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation
        ).to(device)
        
        # Create critic ensemble
        critic = Ensemble(
            Critic,
            num_networks=num_qs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation
        ).to(device)
        
        # Create target critic ensemble
        target_critic = Ensemble(
            Critic,
            num_networks=num_min_qs or num_qs,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation
        ).to(device)
        
        # Copy critic parameters to target critic
        target_critic.load_state_dict(critic.state_dict())
        
        # Create temperature
        temp = Temperature(initial_temperature).to(device)
        
        # Create optimizers
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        temp_optimizer = torch.optim.Adam(temp.parameters(), lr=temp_lr)
        
        return cls(
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temp_optimizer=temp_optimizer
        )
    
    def act(self, obs: torch.Tensor, deterministic: bool = True):
        """Sample actions deterministically"""
        with torch.no_grad():
            return self.actor(obs, deterministic=deterministic)
    
    def sample(self, obs: torch.Tensor):
        """Sample actions from distribution"""
        with torch.no_grad():
            dist = self.actor(obs, deterministic=False)
            return dist.sample()
    
    def soft_update_target(self, tau: float = 0.005):
        """Soft update target critic parameters"""
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving"""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'temp': self.temp.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'temp_optimizer': self.temp_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any], load_critic: bool = True):
        """Load state dictionary"""
        self.actor.load_state_dict(state_dict['actor'])
        if load_critic:
            self.critic.load_state_dict(state_dict['critic'])
            self.target_critic.load_state_dict(state_dict['target_critic'])
        self.temp.load_state_dict(state_dict['temp'])
        
        if 'actor_optimizer' in state_dict:
            self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        if 'critic_optimizer' in state_dict and load_critic:
            self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        if 'temp_optimizer' in state_dict:
            self.temp_optimizer.load_state_dict(state_dict['temp_optimizer'])
    
    def save(self, save_path: str):
        """Save model to file"""
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)
    
    def load(self, load_path: str, load_critic: bool = True):
        """Load model from file"""
        state_dict = torch.load(load_path, map_location='cpu')
        self.load_state_dict(state_dict, load_critic=load_critic)
        return self
    
    def to(self, device):
        """Move all models to device"""
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        self.target_critic = self.target_critic.to(device)
        self.temp = self.temp.to(device)
        return self


# Utility functions for compatibility
def create_feature_extractor(input_dim: int, hidden_dims: list = [256, 256], activation: nn.Module = nn.ReLU):
    """Create a standard MLP feature extractor"""
    return MLP(input_dim=input_dim, hidden_dims=hidden_dims, activation=activation)


def create_actor(obs_dim: int, action_dim: int, **kwargs):
    """Create a DiagGaussianActor"""
    return DiagGaussianActor(obs_dim=obs_dim, action_dim=action_dim, **kwargs)


def create_critic(obs_dim: int, action_dim: int, **kwargs):
    """Create a Critic"""
    return Critic(obs_dim=obs_dim, action_dim=action_dim, **kwargs)


def create_critic_ensemble(obs_dim: int, action_dim: int, num_critics: int = 2, **kwargs):
    """Create an ensemble of critics"""
    return Ensemble(Critic, num_critics, obs_dim=obs_dim, action_dim=action_dim, **kwargs)
