import gymnasium as gym
import numpy as np
from typing import *
import jax
from collections import deque
from jaxtyping import PyTree


class SequentialReplayBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               seed: Optional[int] = None,
               ):
    """
    Sequential replay buffer with support for parallel environments. 

    To simplify the implementation and speed up sampling, episode boundaries are NOT respected. i.e., the sampled subsequences may span multiple episodes. Any code using this buffer should handle this with termination/truncation signals

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store PER PARALLEL ENVIRONMENT
    dummy_input : Dict
        Example input from the environment. Used to determine the shape and dtype of the data to store
    num_envs : int, optional
        Number of parallel environments used for data collection, by default 1
    seed : Optional[int], optional
        Seed for sampling, by default None
    """
    print("[BUFFER] Creating buffer with priority sampling")
    self.capacity = capacity
    self.num_envs = num_envs
    self.data = jax.tree.map(lambda x: np.zeros(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    # Size and index counter for each environment buffer
    self.sizes = np.zeros(num_envs, dtype=int)
    self.current_inds = np.zeros(num_envs, dtype=int)

    self.np_random = np.random.default_rng(seed=seed)

  def insert(self,
             data: PyTree,
             env_mask: Optional[np.ndarray] = None
             ) -> None:
    """
    Insert data into the buffer

    Parameters
    ----------
    data : PyTree
        Data to insert
    env_mask : Optional[np.ndarray], optional
        A boolean mask of size self.num_envs, which specifies which env buffers receive new data. If None, all envs receive data, by default None
    """
    # Insert data for the specified envs
    if env_mask is None:
      env_mask = np.ones(self.num_envs, dtype=bool)

    def masked_set(x, y):
      x[self.current_inds, env_mask] = y[env_mask]
    jax.tree.map(masked_set, self.data, data)

    # Update buffer state
    self.current_inds[env_mask] = (
        self.current_inds[env_mask] + 1
    ) % self.capacity
    self.sizes[env_mask] = np.clip(self.sizes[env_mask] + 1, 0, self.capacity)

  def sample(self, batch_size: int, sequence_length: int, priority_ratio: float = 0.5) -> PyTree:
    """
    Sample a batch of sequences from the buffer with priority for good experiences.

    Sequences are drawn partly uniformly, partly prioritized from each environment buffer, and they may cross episode boundaries.

    Parameters
    ----------
    batch_size : int
    sequence_length : int

    Returns
    -------
    PyTree
        A batch of sequences with shape (sequence_length, batch_size, *)
    """
    # Regular random sampling for part of the batch
    regular_size = int(batch_size * (1 - priority_ratio))
    env_inds_regular = self.np_random.integers(low=0, high=self.num_envs, size=regular_size)
    start_inds_regular = self.np_random.integers(
        low=0, high=self.sizes[env_inds_regular] - sequence_length,
        size=regular_size, endpoint=True,
    )
    
    # Priority sampling for the rest
    priority_size = batch_size - regular_size
    if priority_size > 0:
        # Compute reward sums for each possible sequence
        reward_sums = np.zeros((self.num_envs, max(self.sizes)))
        for env_idx in range(self.num_envs):
            for i in range(sequence_length-1, self.sizes[env_idx]):
                # Sum rewards in the sequence
                seq_start = max(0, i - sequence_length + 1)
                reward_sums[env_idx, seq_start] = np.sum(
                    self.data['reward'][np.arange(seq_start, i+1) % self.capacity, env_idx]
                )
        
        # Flatten and normalize to create a probability distribution
        flat_rewards = reward_sums.flatten()
        flat_rewards = np.maximum(flat_rewards, 0)  # Only consider positive rewards
        if np.sum(flat_rewards) > 0:
            probs = flat_rewards / np.sum(flat_rewards)
            # Sample indices according to reward probabilities
            flat_indices = self.np_random.choice(
                len(flat_rewards), size=priority_size, p=probs, replace=True
            )
            # Convert flat indices back to env_idx and start_idx
            env_inds_priority = flat_indices // max(self.sizes)
            start_inds_priority = flat_indices % max(self.sizes)
        else:
            # Fallback to random if no positive rewards
            env_inds_priority = self.np_random.integers(low=0, high=self.num_envs, size=priority_size)
            start_inds_priority = self.np_random.integers(
                low=0, high=self.sizes[env_inds_priority] - sequence_length,
                size=priority_size, endpoint=True,
            )
        
        # Combine regular and priority samples
        env_inds = np.concatenate([env_inds_regular, env_inds_priority])
        start_inds = np.concatenate([start_inds_regular, start_inds_priority])
    else:
        env_inds = env_inds_regular
        start_inds = start_inds_regular
    
    # Handle wrapping
    start_inds = (
        start_inds - (self.sizes[env_inds] - self.current_inds[env_inds])
    ) % self.capacity
    
    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = start_inds[:, None] + np.arange(sequence_length)
    batch = jax.tree.map(
        lambda x: np.swapaxes(x[sequence_inds % self.capacity, env_inds[:, None]], 0, 1),
        self.data
    )
    
    return batch

  def get_state(self) -> Dict:
    return {
        'current_inds': self.current_inds,
        'sizes': self.sizes,
        'data': self.data,
    }

  def restore(self, state: Dict) -> None:
    self.current_inds = state['current_inds']
    self.sizes = state['sizes']
    self.data = state['data']