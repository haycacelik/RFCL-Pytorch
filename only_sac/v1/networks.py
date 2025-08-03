
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, TransformedDistribution, TanhTransform
import numpy as np

# Helper for orthogonal initialization
def default_init(layer, scale=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=scale)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 256], activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            default_init(layer)
            layers.extend([layer, activation()])
            prev_dim = hidden_dim
        
        # Remove the last activation
        self.network = nn.Sequential(*layers[:-1])
    
    def forward(self, x):
        return self.network(x)

class Ensemble(nn.Module):
    def __init__(self, net_cls, num=2, *args, **kwargs):
        super().__init__()
        self.nets = nn.ModuleList([net_cls(*args, **kwargs) for _ in range(num)])

    def forward(self, *args):
        return torch.stack([net(*args) for net in self.nets], dim=0)

class Critic(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.value_head = nn.Linear(feature_extractor.output_dim, 1)
        default_init(self.value_head, 1.0)

    def forward(self, obs, acts):
        x = torch.cat([obs, acts], dim=-1)
        features = self.feature_extractor(x)
        value = self.value_head(features)
        return value.squeeze(-1)

class DiagGaussianActor(nn.Module):
    def __init__(
        self,
        feature_extractor,
        act_dims,
        output_activation=F.relu,
        tanh_squash_distribution=True,
        state_dependent_std=True,
        log_std_range=(-5.0, 2.0),
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.act_dims = act_dims
        self.output_activation = output_activation
        self.tanh_squash_distribution = tanh_squash_distribution
        self.state_dependent_std = state_dependent_std
        self.log_std_range = log_std_range

        self.action_head = nn.Linear(feature_extractor.output_dim, act_dims)
        default_init(self.action_head, 1.0)
        if state_dependent_std:
            self.log_std_head = nn.Linear(feature_extractor.output_dim, act_dims)
            default_init(self.log_std_head, 1.0)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dims))

    def forward(self, x, deterministic=False):
        x = self.feature_extractor(x)
        mean = self.action_head(x)
        if not self.tanh_squash_distribution:
            mean = torch.tanh(mean)
        # for evaluation
        if deterministic:
            return torch.tanh(mean)
        if self.state_dependent_std:
            log_std = self.log_std_head(x)
            log_std = torch.tanh(log_std)
        else:
            log_std = self.log_std
        min_log_std, max_log_std = self.log_std_range
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1)
        std = torch.exp(log_std)
        # MultivariateNormal expects covariance_matrix or scale_tril, so we use diag_embed for diagonal covariance
        dist = MultivariateNormal(mean, covariance_matrix=torch.diag_embed(std**2))
        if self.tanh_squash_distribution:
            dist = TransformedDistribution(dist, [TanhTransform(cache_size=1)])
        return dist
