import flax.linen as nn
from typing import Callable, Tuple, Optional
from tensorflow_probability.substrates import jax as tfp # jax compatible tebsorflow probability distributions
tfd = tfp.distributions
import jax.numpy as jnp
tfb = tfp.bijectors

def default_init(scale: Optional[float] = np.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DiagGaussianActor(nn.Module):
    feature_extractor: nn.Module
    act_dims: int
    output_activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    tanh_squash_distribution: bool = True

    state_dependent_std: bool = True
    log_std_range: Tuple[float, float] = (-5.0, 2.0)

    def setup(self) -> None:
        if self.state_dependent_std:
            # Add final dense layer initialization scale and orthogonal init
            self.log_std = nn.Dense(self.act_dims, kernel_init=default_init(1))
        else:
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.act_dims,))

        # scale of orthgonal initialization is recommended to be (high - low) / 2.
        # We always assume envs use normalized actions [-1, 1] so we init with 1
        self.action_head = nn.Dense(self.act_dims, kernel_init=default_init(1))

    def __call__(self, x, deterministic=False):
        x = self.feature_extractor(x)
        a = self.action_head(x)
        if not self.tanh_squash_distribution:
            a = nn.tanh(a)
        if deterministic:
            return nn.tanh(a)
        if self.state_dependent_std:
            log_std = self.log_std(x)
            log_std = nn.tanh(log_std)
        else:
            log_std = self.log_std
        log_std = self.log_std_range[0] + 0.5 * (self.log_std_range[1] - self.log_std_range[0]) * (log_std + 1)
        dist = tfd.MultivariateNormalDiag(a, jnp.exp(log_std))
        # distrax has some numerical imprecision bug atm where calling sample then log_prob can raise NaNs. tfd is more stable at the moment
        # dist = distrax.MultivariateNormalDiag(a, jnp.exp(log_std))
        if self.tanh_squash_distribution:
            # dist = distrax.Transformed(distribution=dist, bijector=distrax.Block(distrax.Tanh(), ndims=1))
            dist = tfd.TransformedDistribution(distribution=dist, bijector=tfb.Tanh())
        return dist
