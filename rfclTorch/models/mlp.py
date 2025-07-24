"""MLP class"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
# import jax.numpy as jnp
import numpy as np

from .types import NetworkConfig


@dataclass
class MLPArchConfig:
    features: List[int]
    inFeatures: int
    activation: Union[Callable, str] = "relu"
    output_activation: Union[Callable, str] = None
    use_layer_norm: bool = False



@dataclass
class MLPConfig(NetworkConfig):
    type = "mlp"
    arch_cfg: MLPArchConfig


def default_init(scale: Optional[float] = np.sqrt(2)):#check where this is used!
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    """
    Parameters
    ----------
    features - hidden units in each layer

    activation - internal activation

    output_activation - activation after final layer, default is None
    """

    # features: Sequence[int]
    # activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    # output_activation: Callable[[jnp.ndarray], jnp.ndarray] = None
    # final_ortho_scale: float = np.sqrt(2)
    # use_layer_norm: bool = False

    def __init__(self,features: List[int],
                 inFeatures:int,
                 activation = nn.ReLU,
                 output_activation = None,
                 use_layer_norm: bool = False,
                 final_ortho_scale: float = np.sqrt(2)
        ):
        #missing ortho scale and kernel inits!
        
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        self.activations = nn.ModuleList()
        self.fc = nn.ModuleList()
        if use_layer_norm:
            self.lN = nn.ModuleList()
            
        prevFeature = inFeatures
        for feature in features[:-1]:
            self.fc.append(nn.Linear(prevFeature,feature))
            self.activations.append(activation())
            
            if self.use_layer_norm:
                self.lN.append(nn.LayerNorm(feature))
            prevFeature = feature
        
        self.fc.append(nn.Linear(prevFeature,features[-1]))
        
        if self.use_layer_norm:
                self.lN.append(nn.LayerNorm(features[-1]))
        
        if output_activation is not None:
            self.outputActivation = output_activation()
        else:
            self.outputActivation = None
            
        for layer in self.fc[:-1]:
            nn.init.orthogonal_(layer.weight,gain=np.sqrt(2))
            
        nn.init.orthogonal_(self.fc[-1].weight,gain=final_ortho_scale)
            
            
    def forward(self,x):
        if self.use_layer_norm:
            for fc,act,ln in zip(self.fc[:-1],self.lN[:-1],self.activations):
                x = fc(x)
                x = ln(x)
                x = act(x)
        else:
            for fc,act in zip(self.fc[:-1],self.activations):
                x = fc(x)
                x = act(x)
                
        x = self.fc[-1](x)
        
            
        if self.outputActivation is not None:
            if self.use_layer_norm:
                x = self.lN[-1](x)
            x = self.outputActivation(x)
            
        return x
             
        
    # @nn.compact
    # def __call__(self, x):
    #     for feat in self.features[:-1]:
    #         x = nn.Dense(feat, kernel_init=default_init())(x)
    #         if self.use_layer_norm:
    #             x = nn.LayerNorm()(x)
    #         x = self.activation(x)
    #     x = nn.Dense(self.features[-1], kernel_init=default_init(self.final_ortho_scale))(x)
    #     if self.output_activation is not None:
    #         if self.use_layer_norm:
    #             x = nn.LayerNorm()(x)
    #         x = self.output_activation(x)


    #     return x
