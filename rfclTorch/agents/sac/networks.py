"""
Models for SAC
"""
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Type
import random
import copy
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


from rfclTorch.agents.sac.config import TimeStep


# Chatgpt wrote this, idk if works properly
class TanhTransform(torch.distributions.Transform):
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        # Forward transform: tanh
        return x.tanh()

    def _inverse(self, y):
        # Inverse transform: arctanh
        # Clamp input to avoid numerical issues outside (-1, 1)
        y = y.clamp(min=-0.999999, max=0.999999)
        return 0.5 * (torch.log1p(y) - torch.log1p(-y))

    def log_abs_det_jacobian(self, x, y):
        # Jacobian of tanh is 1 - tanh(x)^2 = 1 - y^2
        return 2.0 * (math.log(2) - x - torch.nn.functional.softplus(-2.0 * x))
        # Alternative simpler:
        # return torch.log1p(-y.pow(2)).abs()
#what does this even do? Made basic changes
#less efficient than JAX version, improve later
class Ensemble(nn.Module):

    def __init__(self,mod:nn.Module,num:int):
        super().__init__()
        self.module = mod
        self.ensemble = nn.ModuleList([copy.deepcopy(self.module) for i in range(num)])
        self.num = num
        
    def forward(self, *args):
        out = [mod(*args) for mod in self.ensemble]
        return torch.stack(out,dim=0)
    
    def sample(self,numSample:int):
        return random.sample(self.ensemble,numSample)
    
        


class Critic(nn.Module):
    
    def __init__(self,feature_extractor:nn.Module, 
                 in_dims: int
        ):
        super().__init__()
        self.feature_extractor=feature_extractor
        self.fc = nn.Linear(in_dims,1)

    #maybe have np array as input and to tensor()?
    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        obs = torch.tensor(obs,dtype=torch.float32)
        acts = torch.tensor(acts,dtype=torch.float32)
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
    def __init__(self,
                 feature_extractor: nn.Module,
                 act_dims: int,
                 in_dims: int,
                 output_activation = nn.ReLU,
                 tanh_squash_distribution: bool = True,
                 state_dependent_std: bool = True,
                 log_std_range: Tuple[float,float] = (-5.0,2.0)
        ):
        super().__init__()
        
        self.feature_extractor=feature_extractor
        self.act_dims=act_dims
        self.output_activation=output_activation
        self.tanh_squash_distribution=tanh_squash_distribution
        self.state_dependent_std=state_dependent_std
        self.log_std_range=log_std_range
        self.in_dims = in_dims
        
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
           # self.log_std = nn.Dense(self.act_dims, kernel_init=default_init(1))
            #check if some custom init needed, probably not since default was used
            self.log_std = nn.Linear(self.in_dims,self.act_dims)
        else:
            self.log_std = nn.Parameter(torch.zeros(self.act_dims))

        # scale of orthgonal initialization is recommended to be (high - low) / 2.
        # We always assume envs use normalized actions [-1, 1] so we init with 1
        
        #might require different input dimensions!
        self.action_head = nn.Linear(self.in_dims,self.act_dims)

    def forward(self, x, deterministic=False):

        x = self.feature_extractor(x)
        a = self.action_head(x)
        
        #why are their two calls if determinsitc?
        if not self.tanh_squash_distribution:
            a = nn.Tanh()(a)
        if deterministic:
            return nn.Tanh()(a)
        if self.state_dependent_std:
            log_std = self.log_std(x)
            log_std = nn.Tanh()(log_std)
        else:
            log_std = self.log_std
        log_std = self.log_std_range[0] + 0.5 * (self.log_std_range[1] - self.log_std_range[0]) * (log_std + 1)
        dist = torch.distributions.Independent(torch.distributions.Normal(loc=a, scale=torch.exp(log_std)),1)
        # distrax has some numerical imprecision bug atm where calling sample then log_prob can raise NaNs. tfd is more stable at the moment
        # dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=1))
            #dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
            dist = torch.distributions.TransformedDistribution(dist,TanhTransform())
            #torch doesn't have one built in, add this in a later step!0
        return dist


class Temperature(nn.Module):

    def __init__(self,initial_temperature=1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(initial_temperature)))

    def forward(self):
        return torch.exp(self.log_temp)



class ActorCritic(nn.Module):

    
    def __init__(self,actor:nn. Module,
                 critic: nn.Module,
                 target_critic: nn.Module,
                 temp: nn.Module,
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
        super().__init__()
    # Shoud probably create adam optimizers instead of taking optim inputs in the beginning
        self.actor=actor
        self.critic=Ensemble(critic,num_qs)
        self.target_critic=Ensemble(critic,num_qs or num_min_qs)
        target_critic.load_state_dict(critic.state_dict())
        self.temp=temp
        self.sample_obs=sample_obs
        self.sample_acts=sample_acts
        self.actor_optim=torch.optim.Adam(self.actor.parameters(),lr=actor_optim_lr)
        self.critic_optim=torch.optim.Adam(self.critic.parameters(),lr=critic_optim_lr)
        self.initial_temperature=initial_temperature
        self.temperature_optim=torch.optim.Adam(self.temp.parameters(),lr=temperature_optim_lr)
        self.num_qs=num_qs
        self.num_min_qs=num_min_qs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"using device {self.device} ")
        
        self.criticLoss = nn.MSELoss()
        
        self.to(self.device)

    def act(self, obs, deterministic=False):
        
        obs = torch.tensor(obs,device=self.device,dtype=torch.float32)
        """Sample actions deterministicly"""
        
        return self.actor(obs, deterministic=deterministic)


    def sample(self, obs):
        """Sample actions from distribution"""
        return self.actor(obs).sample()


    def save(self, save_path: str):
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)

    #changed to not return the actor and critic models as they are now kept in the 
    # ActorCritic Class
    def load(self, params_dict):
        self.load_state_dict(params_dict)


    def load_from_path(self, load_path: str, load_critic=True):
        params_dict = torch.load(load_path, map_location=self.device)  # or "cuda" if needed
        self.load(params_dict)
        return self
    
    def updateCritic(self, batch: TimeStep, discount: float, backup_entropy: bool, numSample:int):
        
        batch_next_obs = torch.tensor(batch.next_env_obs,device=self.device,dtype=torch.float32)
        batch_reward = torch.tensor(batch.reward,device=self.device,dtype=torch.float32)
        batch_mask = torch.tensor(batch.mask,device=self.device,dtype=torch.float32)
        
        self.critic_optim.zero_grad()
        
        dist = self.actor(batch_next_obs)
        next_actions = dist.sample()
        next_log_probs = dist.log_prob(next_actions)#idk if this line will work
        
        randomTargetsCritic =  self.target_critic.sample(numSample)
        nextQs = [mod(batch_next_obs,next_actions) for mod in randomTargetsCritic]
        nextQs = torch.stack(nextQs,dim=0)
        
        nextQ,_ = torch.min(nextQs,dim=0)
        
        targetQ = batch.reward + discount * batch_mask * next_q
        

        if backup_entropy:
            targetQ -= discount * batch_mask * self.temp() * next_log_probs
          
        Qs = self.critic(batch_next_obs,next_actions)  

        loss = self.criticLoss(Qs,targetQ)#might have issues with shape here!
        loss.backward()
        self.critic_optim.step()
        

    def updateActor(self,batch: TimeStep):
        
        self.actor_optim.zero_grad()
        dist = self.actor(batch.env_obs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        Qs = self.critic(batch.env_obs,actions)
        Q = torch.mean(Qs,dim=0)
        
        loss = torch.mean(log_probs * ac.temp() - Q)
        loss.backward()
        self.actor_optim.step()
        return -log_probs.mean()
        
    def updateTemp(self,entropy: float, target_entropy:float):
        self.temperature_optim.zero_grad()
        temperature = self.temp()
        loss = temperature * (entropy - target_entropy).mean()
        loss.backward()
        self.temperature_optim.step()
        
    def updateTarget(self, tau:float):
         for target_param, main_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)
        
    def updateParams(self):#use a function here to update the model!
        raise NotImplementedError