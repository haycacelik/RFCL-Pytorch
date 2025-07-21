"""
Models for SAC
"""
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Type
import numpy as np
import torch
import torch.nn as nn
# import flax
# import flax.linen as nn
# import jax
# import jax.numpy as jnp
# import optax
# from chex import Array, PRNGKey
# from flax import struct
#from tensorflow_probability.substrates import jax as tfp

from rfcl.models import Model# to do
from rfcl.models.model import Params# to do


tfd = tfp.distributions
tfb = tfp.bijectors

#what does this even do? Made basic changes
class Ensemble(nn.Module):

    def __init__(self,net_cls,num):
        super().__init__
        self.net_cls=net_cls
        self.num=num
        
        
    def forward(self, *args):
        ensemble = torch.vmap(
            self.net_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num,
        )
        return ensemble()(*args)


class Critic(nn.Module):
    
    def __init__(self,feature_extractor:nn.Module, 
                 in_dims: int
        ):
        super().__init__()
        self.feature_extractor=feature_extractor
        self.fc = nn.Linear(in_dims,1)

    #maybe have np array as input and to tensor()?
    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, acts],dim= -1)
        features = self.feature_extractor(x)
        value = self.fc(features)
        return value.squeeze(-1)


#to do check where this is called?
def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DiagGaussianActor(nn.Module):

    #maybve add data types of input?
    #currently in_dims are unknwon, maybe state dim? READ THE CODE!
    def __init__(self,feature_extractor,act_dims,output_activation,tanh_squash_distribution,
                 state_dependent_std,log_std_range,in_dims):
        super().__init__()
        
        self.feature_extractor=feature_extractor
        self.act_dims=act_dims
        self.output_activation=output_activation
        self.tanh_squash_distribution=tanh_squash_distribution
        self.state_dependent_std=state_dependent_std
        self.log_std_range=log_std_range
        self.in_dems = in_dems
        
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
           # self.log_std = nn.Dense(self.act_dims, kernel_init=default_init(1))
            #check if some custom init needed, probably not since default was used
            self.log_std = nn.Linear(self.in_dems,self.act_dims)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.act_dims))

        # scale of orthgonal initialization is recommended to be (high - low) / 2.
        # We always assume envs use normalized actions [-1, 1] so we init with 1
        
        #might require different input dimensions!
        self.action_head = nn.Linear(self.in_dems,self.act_dims)

    def forward(self, x, deterministic=False):
        x = self.feature_extractor(x)
        a = self.action_head(x)
        
        #why are their two calls if determinsitc?
        if not self.tanh_squash_distribution:
            a = nn.Tanh(a)
        if deterministic:
            return nn.Tanh(a)
        if self.state_dependent_std:
            log_std = self.log_std(x)
            log_std = nn.tanh(log_std)
        else:
            log_std = self.log_std
        log_std = self.log_std_range[0] + 0.5 * (self.log_std_range[1] - self.log_std_range[0]) * (log_std + 1)
        dist = torch.distribution.MultivariateNormalNormal(a, torch.exp(log_std))
        # distrax has some numerical imprecision bug atm where calling sample then log_prob can raise NaNs. tfd is more stable at the moment
        # dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=1))
            #dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
            raise NotImplementedError()
            #torch doesn't have one built in, add this in a later step!0
        return dist


class Temperature(nn.Module):

    def __init__(self,initial_temperature=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(initial_temperature))

    def forward(self):
        return torch.exp(self.log_temp)


@dataclass
class ActorCritic:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    
    def __init__(self,actor:nn. Module,
                 critic: nn.Module,
                 target_critic: nn.Module,
                 temp: nn.Module,
                 rng_key,#see if this is being used,
                 sample_obs: np.array,
                 sample_acts: np.array,
                 actor_optim_lr = 3e-4,
                 critic_optim_lr = 3e-4,
                 initial_temperature: float = 1.0,
                 temperature_optim_lr = 3e-4,
                 num_qs: int=10,
                 num_min_qs: int =2,
                 device="cpu"
    ):
    # Shoud probably create adam optimizers instead of taking optim inputs in the beginning
        self.actor=actor
        self.critic=critic
        self.target_critic=target_critic
        self.temp=temp
        self.rng_key=rng_key
        self.sample_obs=sample_obs
        self.sample_acts=sample_acts
        self.actor_optim=torch.optim.Adam(self.actor.parameters(),lr=actor_optim_lr)
        self.critic_optim=torch.optim.Adam(self.critic.parameters(),lr=critic_optim_lr)
        self.initial_temperature=initial_temperature
        self.temperature_optim=torch.optim.Adam(self.temp.parameters(),lr=temperature_optim_lr)
        self.num_qs=num_qs
        self.num_min_qs=num_min_qs
        self.device=device

    def act(self, obs):
        """Sample actions deterministicly"""
        return actor(obs, deterministic=True), {}

    def sample(self, obs):
        """Sample actions from distribution"""
        return actor(obs).sample(), {}

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),
            target_critic=self.target_critic.state_dict(),
            temp=self.temp.state_dict(),
        )

    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.state_dict()))

    #changed to not return the actor and critic models as they are now kept in the 
    # ActorCritic Class
    def load(self, params_dict: Params, load_critic=True):
        self.actor.load_state_dict(params_dict["actor"])
        if load_critic:
            critic = self.critic.load_state_dict(params_dict["critic"])
            target_critic = self.target_critic.load_state_dict(params_dict["target_critic"])
        temp = self.temp.load_state_dict(params_dict["temp"])

    def load_from_path(self, load_path: str, load_critic=True):
        with open(load_path, "rb") as f:
            params_dict = flax.serialization.from_bytes(self.state_dict(), f.read())
        self.load(params_dict, load_critic=load_critic)
        return self

    def load_from_path(self, load_path: str, load_critic=True):
        params_dict = torch.load(load_path, map_location=self.device)  # or "cuda" if needed
        self.load(params_dict, load_critic=load_critic)
        return self