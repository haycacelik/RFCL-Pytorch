import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        
        # Pre-allocate memory
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices]
        )
        return batch
    
    def __len__(self):
        return self.size
