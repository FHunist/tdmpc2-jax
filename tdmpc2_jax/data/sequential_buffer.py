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

  def sample(self, batch_size: int, sequence_length: int) -> PyTree:
    """
    Sample a batch of sequences from the buffer using stratified sampling.
    
    This approach samples more heavily from recent experiences which tend to have
    better performance as training progresses, while still maintaining some
    exploration of older experiences.
    
    Parameters
    ----------
    batch_size : int
        Number of sequences to sample
    sequence_length : int
        Length of each sequence
        
    Returns
    -------
    PyTree
        A batch of sequences with shape (sequence_length, batch_size, *)
    """
    # Divide the buffer into recent and older segments
    recent_ratio = 0.3  # Consider the most recent 30% as "recent"
    recent_size = min(int(self.capacity * recent_ratio), 
                     np.max(self.sizes))
    
    # Sample more heavily from recent experiences
    recent_batch_ratio = 0.7  # 70% of samples from recent data
    recent_batch_size = int(batch_size * recent_batch_ratio)
    older_batch_size = batch_size - recent_batch_size
    
    # Sample from recent experiences
    recent_env_inds = self.np_random.integers(
        low=0, 
        high=self.num_envs,
        size=recent_batch_size
    )
    
    recent_max_positions = np.minimum(self.sizes[recent_env_inds], recent_size)
    recent_min_positions = np.maximum(0, recent_max_positions - recent_size)
    
    # Adjust to ensure we have enough history for the sequence length
    recent_min_positions = np.minimum(
        recent_min_positions,
        np.maximum(0, recent_max_positions - sequence_length)
    )
    
    # Sample start positions for recent experiences
    if np.all(recent_max_positions > recent_min_positions):
        recent_start_offsets = self.np_random.integers(
            low=0,
            high=np.maximum(1, recent_max_positions - recent_min_positions - sequence_length + 1),
            size=recent_batch_size
        )
        recent_start_inds = recent_min_positions + recent_start_offsets
    else:
        # Fallback if we don't have enough data yet
        recent_start_inds = self.np_random.integers(
            low=0,
            high=np.maximum(1, self.sizes[recent_env_inds] - sequence_length + 1),
            size=recent_batch_size
        )
    
    # Sample from older experiences if we have enough history
    if older_batch_size > 0 and np.any(self.sizes > recent_size + sequence_length):
        # Find envs with enough old data
        valid_envs = np.where(self.sizes > recent_size + sequence_length)[0]
        if len(valid_envs) == 0:
            # Fallback to all envs if none have enough old data
            valid_envs = np.arange(self.num_envs)
        
        older_env_inds = self.np_random.choice(valid_envs, size=older_batch_size)
        
        # Calculate valid range for older samples
        max_older_positions = np.maximum(0, self.sizes[older_env_inds] - recent_size)
        
        # Ensure we have enough history for sequence_length
        older_start_inds = self.np_random.integers(
            low=0,
            high=np.maximum(1, max_older_positions - sequence_length + 1),
            size=older_batch_size
        )
        
        # Combine samples
        env_inds = np.concatenate([recent_env_inds, older_env_inds])
        start_inds = np.concatenate([recent_start_inds, older_start_inds])
    else:
        # Fallback: sample the rest from recent to keep batch_size
        extra = batch_size - recent_batch_size
        extra_env_inds = self.np_random.integers(
            low=0,
            high=self.num_envs,
            size=extra
        )
        # reuse the same offset logic as above for recent
        extra_max_pos = np.minimum(self.sizes[extra_env_inds], recent_size)
        extra_min_pos = np.maximum(0, extra_max_pos - recent_size)
        extra_min_pos = np.minimum(
            extra_min_pos,
            np.maximum(0, extra_max_pos - sequence_length)
        )
        extra_start_offsets = self.np_random.integers(
            low=0,
            high=np.maximum(1, extra_max_pos - extra_min_pos - sequence_length + 1),
            size=extra
        )
        extra_start_inds = extra_min_pos + extra_start_offsets

        env_inds   = np.concatenate([recent_env_inds, extra_env_inds])
        start_inds = np.concatenate([recent_start_inds, extra_start_inds])
    
    # Handle wrapping: For wrapped buffers, we need to adjust indices relative to current pointer
    curr_positions = (self.current_inds[env_inds] - 1) % self.capacity  # Current position is the last inserted
    buffer_start_inds = (curr_positions - (self.sizes[env_inds] - start_inds - 1)) % self.capacity
    
    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = (buffer_start_inds[:, None] + np.arange(sequence_length)) % self.capacity
    
    buffer_indices = np.arange(self.capacity)
    is_valid_sequence = np.all(
        np.diff(buffer_indices[sequence_inds], axis=1) == 1, axis=1
    )
    
    # If we have invalid sequences, resample them
    if not np.all(is_valid_sequence):
        invalid_mask = ~is_valid_sequence
        invalid_count = np.sum(invalid_mask)
        
        # Simple fix: just shift the invalid sequences to ensure they're consecutive
        for i in np.where(invalid_mask)[0]:
            # Find a safe consecutive region
            safe_start = (buffer_start_inds[i] - sequence_length + 1) % self.capacity
            buffer_start_inds[i] = safe_start
            sequence_inds[i] = (safe_start + np.arange(sequence_length)) % self.capacity
    
    # Extract the data
    batch = jax.tree.map(
        lambda x: np.swapaxes(x[sequence_inds, env_inds[:, None]], 0, 1),
        self.data
    )
    
    return batch
     

  def prio_sample(self, batch_size: int, sequence_length: int, priority_ratio: float = 0.5) -> PyTree:
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