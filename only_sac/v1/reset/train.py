import sys
import numpy as np
from dataclasses import dataclass
from typing import Optional

from rfcl.utils.parse import parse_cfg


@dataclass
class TrainConfig:
    steps: int
    actor_lr: float
    critic_lr: float
    dataset_path: str
    shuffle_demos: bool
    num_demos: int

    data_action_scale: Optional[float]

    # reverse curriculum wrapper configs
    reverse_step_size: int
    curriculum_method: str
    start_step_sampler: str
    per_demo_buffer_size: int
    demo_horizon_to_max_steps_ratio: float
    train_on_demo_actions: bool

    # forward curriculum configs
    forward_curriculum: str
    staleness_transform: str
    staleness_coef: float
    staleness_temperature: float
    score_transform: str
    score_temperature: float
    num_seeds: int

    # stage 2 training configs
    load_actor: bool
    load_critic: bool
    load_as_offline_buffer: bool
    load_as_online_buffer: bool

    # other configs that are generally used for experimentation
    use_orig_env_for_eval: bool = True
    eval_start_of_demos: bool = False


@dataclass
class SACNetworkConfig:
    actor: NetworkConfig
    critic: NetworkConfig


@dataclass
class SACExperiment:
    seed: int
    sac: SACConfig
    env: EnvConfig
    eval_env: EnvConfig
    train: TrainConfig
    network: SACNetworkConfig
    logger: Optional[LoggerConfig]
    verbose: int
    algo: str = "sac"
    stage_1_model_path: str = None  # if not None, will load pretrained stage 1 model and skip to stage 2 of training
    save_eval_video: bool = True  # whether to save eval videos
    stage_1_only: bool = False  # stop training after reverse curriculum completes
    stage_2_only: bool = False # skip stage 1 training
    demo_seed: int = None  # fix a seed to fix which demonstrations are sampled from a dataset


def main(cfg: SACExperiment):
    np.random.seed(cfg.seed)

    ### Setup the experiment parameters ###

    # Setup training and evaluation environment configs
    env_cfg = cfg.env
    if "env_kwargs" not in env_cfg:
        env_cfg["env_kwargs"] = dict()
    cfg.eval_env = {**env_cfg, **cfg.eval_env}
    cfg = from_dict(data_class=SACExperiment, data=OmegaConf.to_container(cfg))
    env_cfg = cfg.env
    eval_env_cfg = cfg.eval_env



if __name__ == "__main__":
    cfg = parse_cfg(default_cfg_path=sys.argv[1])
    cfg
    # main(cfg)
