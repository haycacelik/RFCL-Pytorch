import os
import pickle
import time
from collections import defaultdict
from functools import partial
from typing import Any, Tuple
from dataclasses import dataclass, replace, asdict
from pathlib import Path
import torch
import numpy as np

from tqdm import tqdm


from rfclTorch.agents.sac.config import SACConfig, TimeStep
from rfclTorch.agents.base import BasePolicy
from rfclTorch.agents.sac.networks import ActorCritic, DiagGaussianActor
from rfclTorch.data.buffer import GenericBuffer
from rfclTorch.data.loop import DefaultTimeStep, EnvLoopState
from rfclTorch.logger import LoggerConfig
from rfclTorch.utils import tools


@dataclass
class TrainStepMetrics:
    train_stats: Any
    train: Any
    update: Any
    time: Any


@dataclass
class SACTrainState:
    # model states

    loop_state: EnvLoopState
    # rng

    # monitoring
    total_env_steps: int
    """
    Total env steps sampled so far
    """
    training_steps: int
    """
    Total training steps so far
    """
    initialized: bool
    """
    When False, will automatically reset the loop state. This is usually false when starting training. When resuming training
    it will try to proceed from the previous loop state
    """


class SAC(BasePolicy):
    def __init__(
        self,
        env_type: str,
        ac: ActorCritic,
        env,
        eval_env=None,
        logger_cfg: LoggerConfig = None,
        cfg: SACConfig = {},
        offline_buffer=None,
    ):
        if isinstance(cfg, dict):
            self.cfg = SACConfig(**cfg)
        else:
            self.cfg = cfg
        super().__init__(env_type, env, eval_env, cfg.num_envs, cfg.num_eval_envs, logger_cfg)
        self.offline_buffer = offline_buffer
        self.ActorCritic = ac
        self.state: SACTrainState = SACTrainState(
            loop_state=EnvLoopState(),
            total_env_steps=0,
            training_steps=0,
            initialized=False,
        )

        def seed_sampler():
            shape = (self.cfg.num_envs, *self.env.single_action_space.shape)
            return torch.distributions.Uniform(-1.0, 1.0).sample(shape)

        self.seed_sampler = seed_sampler

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


    def _sample_action(self, env_obs):
 
        dist = self.ActorCritic.act(env_obs) #define it as a torch distribution(add type guard thing whatever it's called :Type)
        a = dist.sample()
        return a, {}
    
    def _det_action(self,env_obs):
        return self.ActorCritic.act(env_obs,True), {}

    def _env_step(self, loop_state: EnvLoopState, actor: DiagGaussianActor, seed=False):
        data, loop_state = self.loop.rollout(loop_state, partial(self._sample_action), 1)
        return loop_state, data

    def train(self, steps: int, callback_fn=None, verbose=1):
        """
        Args :
                Random key to seed the training with. It is only used if train() was never called before, otherwise the code uses self.state.rng_key
            steps : int
                Max number of environment samples before training is stopped.
        """
        train_start_time = time.time()

        # if env_obs is None, then this is the first time calling train and we prepare the environment
        if not self.state.initialized:
            loop_state = self.loop.reset_loop()
            self.state = replace(
                self.state,
                loop_state=loop_state,
                initialized=True,
            )

        start_step = self.state.total_env_steps

        if verbose:
            pbar = tqdm(total=steps + self.state.total_env_steps, initial=start_step)

        env_rollout_size = self.cfg.steps_per_env * self.cfg.num_envs

        while self.state.total_env_steps < start_step + steps:

            self.state, train_step_metrics = self.train_step(self.state)

            # evaluate the current trained actor periodically
            if (
                self.eval_loop is not None
                and tools.reached_freq(self.state.total_env_steps, self.cfg.eval_freq, step_size=env_rollout_size)
                and self.state.total_env_steps > self.cfg.num_seed_steps
            ):

                eval_results = self.evaluate(
                    num_envs=self.cfg.num_eval_envs,
                    steps_per_env=self.cfg.eval_steps,
                    eval_loop=self.eval_loop,
                    apply_fn = self._det_action
                )
                
                eval_data = {
                    "return": eval_results["eval_ep_rets"], "reward": eval_results["eval_ep_avg_reward"], 
                    "episode_len": eval_results["eval_ep_lens"], "success_once": eval_results["success_once"], "success_at_end": eval_results["success_at_end"]
                }
                self.logger.store(
                    tag="eval",
                    **eval_data
                )
                self.logger.store(tag="eval_stats", **eval_results["stats"])
                self.logger.log(self.state.total_env_steps)
                self.logger.reset()
            self.logger.store(tag="train", **train_step_metrics.train)
            self.logger.store(tag="train_stats", **train_step_metrics.train_stats)

            ### Log Metrics ###
            if verbose:
                pbar.update(n=env_rollout_size)
            total_time = time.time() - train_start_time
            if tools.reached_freq(self.state.total_env_steps, self.cfg.log_freq, step_size=env_rollout_size):
                update_aux = train_step_metrics.update #it's already a dict?#tools.flatten_struct_to_dict(train_step_metrics.update)
                self.logger.store(tag="train", training_steps=self.state.training_steps, **update_aux)
                self.logger.store(
                    tag="time",
                    total=total_time,
                    SPS=self.state.total_env_steps / total_time,
                    step=self.state.total_env_steps,
                    **train_step_metrics.time,
                )
            # log and export the metrics
            self.logger.log(self.state.total_env_steps)
            self.logger.reset()

            # save checkpoints. Note that the logger auto saves upon metric improvements
            if tools.reached_freq(self.state.total_env_steps, self.cfg.save_freq, env_rollout_size):
                self.save(
                    os.path.join(self.logger.model_path, f"ckpt_{self.state.total_env_steps}.jx"),
                    with_buffer=self.cfg.save_buffer_in_checkpoints,
                )

            if callback_fn is not None:
                stop = callback_fn(locals())
                if stop:
                    print(f"Early stopping at {self.state.total_env_steps} env steps")
                    break

    def train_step(self, state: SACTrainState) -> Tuple[SACTrainState, TrainStepMetrics]:
        """
        Perform a single training step

        In SAC this is composed of collecting cfg.steps_per_env * cfg.num_envs of interaction data with a random sample or policy (depending on cfg.num_seed_steps)
        then performing gradient updates

        """

        ac = self.ActorCritic
        loop_state = state.loop_state
        total_env_steps = state.total_env_steps
        training_steps = state.training_steps

        train_custom_stats = defaultdict(list)
        train_metrics = defaultdict(list)
        time_metrics = dict()

        # perform a rollout
        # TODO make this buffer collection jittable
        rollout_time_start = time.time()
        for _ in range(self.cfg.steps_per_env):
            (next_loop_state, data) = self._env_step(
                loop_state,
                ac.actor,
                seed=(total_env_steps <= self.cfg.num_seed_steps and not (self.cfg.seed_with_policy)),
            )

            final_infos = data[
                "final_info"
            ]  
            del data["final_info"]


            # move data to numpy
            data: DefaultTimeStep = DefaultTimeStep(**(tools.tree_map(lambda x: np.array(x)[:, 0] if np.array(x).ndim >= 2 else x,data)))
            terminations = data.terminated
            truncations = data.truncated
            dones = terminations | truncations
            masks = ((~dones) | (truncations)).astype(float)
            if dones.any():
                # note for continuous task wrapped envs where there is no early done, all envs finish at the same time unless
                # they are staggered. So masks is never false.
                # if you want to always value bootstrap set masks to true.
                train_metrics["return"].append(data.ep_ret[dones])
                train_metrics["episode_len"].append(data.ep_len[dones])
                
                for i, final_info in enumerate(final_infos):
                    if final_info is not None:
                        if "stats" in final_info:
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

        # log time metrics

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
        state = replace(
            self.state,
            loop_state=loop_state,
            total_env_steps=total_env_steps + self.cfg.num_envs * self.cfg.steps_per_env,
        )
        # update policy
        #print(f"rollout done, time {rollout_time}")
        update_aux = dict()
        if self.state.total_env_steps >= self.cfg.num_seed_steps:
            update_time_start = time.time()
            
            if self.offline_buffer is not None:
                #removed key argument as it isnt used anyways!
                batch = self.replay_buffer.sample_random_batch(self.cfg.batch_size * self.cfg.grad_updates_per_step // 2)
                offline_batch = self.offline_buffer.sample_random_batch(self.cfg.batch_size * self.cfg.grad_updates_per_step // 2)
                batch = tools.combine(batch, offline_batch)
            else:
                batch = self.replay_buffer.sample_random_batch( self.cfg.batch_size * self.cfg.grad_updates_per_step)

            batch = TimeStep(**batch)
            self.update_parameters(batch)
            #shouldn't need this since the sae actor-critic should update it's weights
            state.training_steps = training_steps + self.cfg.grad_updates_per_step
            
            update_time = time.time() - update_time_start
            time_metrics["update_time"] = update_time
            #print(f"done training step, time : {update_time}")

        
        return state, TrainStepMetrics(time=time_metrics, train=train_metrics, update=update_aux, train_stats=train_custom_stats)

    
    def update_parameters(self, batch: TimeStep):
        def split_batch(batch: TimeStep, num_splits: int) -> list[dict[str, np.ndarray]]:
            batch_dict = asdict(batch)
            split_dicts = [
                {k: np.array_split(v, num_splits)[i] for k, v in batch_dict.items()}
                for i in range(num_splits)
            ]
            return split_dicts
        
        mini_batch_size = self.cfg.batch_size
        assert mini_batch_size * self.cfg.grad_updates_per_step == batch.action.shape[0]
        assert self.cfg.grad_updates_per_step % self.cfg.actor_update_freq == 0
        
        # Split the batch into individual mini-batches for each gradient update
        mini_batches = split_batch(batch, self.cfg.grad_updates_per_step)
        
        # Process critic updates for all mini-batches
        for i, mini_batch_dict in enumerate(mini_batches):
            mini_batch = TimeStep(**mini_batch_dict)
            
            # Update critic on every mini-batch
            self.ActorCritic.updateCritic(mini_batch, self.cfg.discount, self.cfg.backup_entropy, self.cfg.num_min_qs)
            self.ActorCritic.updateTarget(self.cfg.tau)
            
            # Update actor only at specified frequency
            if (i + 1) % self.cfg.actor_update_freq == 0:
                # Collect batches for actor update (from current position back by actor_update_freq)
                actor_batch_start = max(0, i + 1 - self.cfg.actor_update_freq)
                actor_batches = mini_batches[actor_batch_start:i + 1]
                
                # Combine the batches for actor update
                combined_batch = {}
                for key in actor_batches[0].keys():
                    combined_batch[key] = np.concatenate([batch_dict[key] for batch_dict in actor_batches], axis=0)
                
                entropy = self.ActorCritic.updateActor(TimeStep(**combined_batch))
                entropy = entropy.detach()
                
                if self.cfg.learnable_temp:
                    self.ActorCritic.updateTemp(entropy, self.cfg.target_entropy)
        

    def state_dict(self, with_buffer=False):
        state_dict = {
            "actor_critic": self.ActorCritic.state_dict(),
            "logger": self.logger.state_dict() if self.logger else None,
        }
        if with_buffer:
            state_dict["replay_buffer"] = self.replay_buffer
        return state_dict

    def save(self, save_path: str, with_buffer=False):
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict(with_buffer=with_buffer)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            # TODO replace pickle with something more efficient for replay buffers?
        # print(f"Saving Checkpoint {save_path}.", "Time:", time.time() - stime)
        
    def load_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Checkpoint {load_path}")
        self.load(state_dict)

    def load(self, data):
        self.ActorCritic.load_state_dict(data["actor_critic"])
        if self.logger and data["logger"] is not None:
            self.logger.load_state_dict(data["logger"])
        else:
            print("Skip loading logger. No log data will be overwritten/saved")
        if "replay_buffer" in data:
            self.replay_buffer = data["replay_buffer"]
            print(f"Loaded replay buffer with {self.replay_buffer.size() * self.replay_buffer.num_envs} transitions")
            

    def load_policy_from_path(self, load_path: str):
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        print(f"Loading Policy from {load_path}")
        return self.load_policy(state_dict)

    def load_policy(self, data):
        self.ActorCritic.load_state_dict(data["actor_critic"])
        return self.ActorCritic
