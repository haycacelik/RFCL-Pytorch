"""
Build various base models with configurations
"""
from dataclasses import asdict
from typing import Callable

import torch.nn as nn
from dacite import from_dict

from .mlp import MLP, MLPConfig
from .types import NetworkConfig


ACTIVATIONS = dict(relu=nn.ReLU, gelu=nn.GELU, tanh=nn.Tanh, sigmoid=nn.Sigmoid, log_softmax=lambda : nn.LogSoftmax(dim=-1))#check args for softmax


def activation_to_fn(activation: str) -> Callable:
    if activation is None:
        return nn.ReLU
    if activation in ACTIVATIONS:
        return ACTIVATIONS[activation]
    else:
        raise ValueError(f"{activation} is not handled as an activation. Handled activations are {list(ACTIVATIONS.keys())}")


def build_network_from_cfg(cfg: NetworkConfig):
    inFeatures = cfg.arch_cfg["inFeatures"]
    if cfg.type == "mlp":
        #cleanup how data is unpacked here for the MLP, seems unnecessary
        cfg = from_dict(data_class=MLPConfig, data=asdict(cfg))
        cfg.arch_cfg.activation = activation_to_fn(cfg.arch_cfg.activation)
        cfg.arch_cfg.output_activation = activation_to_fn(cfg.arch_cfg.output_activation)
        return MLP(
            features = cfg.arch_cfg.features,
            inFeatures = cfg.arch_cfg.inFeatures,
            activation = cfg.arch_cfg.activation,
            output_activation = cfg.arch_cfg.output_activation,
            use_layer_norm = cfg.arch_cfg.use_layer_norm
        )
