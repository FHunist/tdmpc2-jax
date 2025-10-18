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
        To simplify the implementation and speed up sampling, episode boundaries are NOT respected. 
        i.e., the sampled subsequences may span multiple episodes. Any code using this buffer should 
        handle this with termination/truncation signals
        
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
        self.capacity = capacity
        self.num_envs = num_envs
        
        # Create storage with shape (capacity, num_envs, ...)
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
            A boolean mask of size self.num_envs, which specifies which env buffers receive new data. 
            If None, all envs receive data, by default None
        """
        # Insert data for the specified envs
        if env_mask is None:
            env_mask = np.ones(self.num_envs, dtype=bool)
        
        # Get active environment indices
        active_envs = np.where(env_mask)[0]
        
        # Use advanced indexing - vectorized operation
        def masked_set(x, y):
            # Use advanced indexing with arrays for both dimensions
            x[self.current_inds[active_envs], active_envs] = y[active_envs]
        
        jax.tree.map(masked_set, self.data, data)
        
        # Update buffer state
        self.current_inds[env_mask] = (
            self.current_inds[env_mask] + 1
        ) % self.capacity
        self.sizes[env_mask] = np.minimum(self.sizes[env_mask] + 1, self.capacity)

    def sample(self, batch_size: int, sequence_length: int) -> PyTree:
        """
        Sample a batch of sequences from the buffer.
        Sequences are drawn uniformly from each environment buffer, and they may cross episode boundaries.
        
        Parameters
        ----------
        batch_size : int
        sequence_length : int
        
        Returns
        -------
        PyTree
            A batch of sequences with shape (sequence_length, batch_size, *)
        """
        # Sample envs
        env_inds = self.np_random.integers(
            low=0, high=self.num_envs,
            size=batch_size
        )
        
        # Calculate valid range for each sampled environment
        valid_sizes = self.sizes[env_inds]
        
        # Make sure we have enough data for the sequence length
        min_required = sequence_length
        if np.any(valid_sizes < min_required):
            raise ValueError(f"Some environments don't have enough data. "
                           f"Required: {min_required}, Available: {valid_sizes.min()}")
        
        # For full buffers (circular), we need to adjust sampling
        is_full = self.sizes[env_inds] >= self.capacity
        
        # Sample start indices
        if np.any(is_full):
            # Mix of full and not-full buffers
            start_inds = np.zeros(batch_size, dtype=int)
            
            # For not-full buffers: sample from [0, size - sequence_length]
            not_full_mask = ~is_full
            if np.any(not_full_mask):
                max_starts = valid_sizes[not_full_mask] - sequence_length
                start_inds[not_full_mask] = self.np_random.integers(
                    low=0, 
                    high=max_starts + 1,
                    size=not_full_mask.sum()
                )
            
            # For full buffers: sample from entire circular buffer
            if np.any(is_full):
                start_inds[is_full] = self.np_random.integers(
                    low=0,
                    high=self.capacity,
                    size=is_full.sum()
                )
                # Adjust to be relative to current pointer
                start_inds[is_full] = (
                    self.current_inds[env_inds[is_full]] + start_inds[is_full]
                ) % self.capacity
        else:
            # All buffers are not full - simple case
            max_starts = valid_sizes - sequence_length
            start_inds = self.np_random.integers(
                low=0,
                high=max_starts + 1,
                size=batch_size
            )
        
        # Create sequence indices using broadcasting - fully vectorized
        # Shape: (batch_size, sequence_length)
        sequence_inds = (start_inds[:, None] + np.arange(sequence_length)[None, :]) % self.capacity
        
        # Vectorized gathering using advanced indexing
        # Create index arrays for gathering
        batch_idx = np.arange(batch_size)[:, None]  # (batch_size, 1)
        batch_idx = np.repeat(batch_idx, sequence_length, axis=1)  # (batch_size, sequence_length)
        env_idx = env_inds[:, None]  # (batch_size, 1)
        env_idx = np.repeat(env_idx, sequence_length, axis=1)  # (batch_size, sequence_length)
        
        # Sample from buffer using advanced indexing
        def gather_batch(x):
            # Use advanced indexing to gather all data at once
            # x[sequence_inds.ravel(), env_idx.ravel()] gives us a flat array
            # We reshape it back to (batch_size, sequence_length, ...)
            gathered = x[sequence_inds.ravel(), env_idx.ravel()]
            
            # Determine the shape of remaining dimensions
            remaining_shape = gathered.shape[1:] if len(gathered.shape) > 1 else ()
            
            # Reshape to (batch_size, sequence_length, ...)
            reshaped = gathered.reshape(batch_size, sequence_length, *remaining_shape)
            
            # Transpose to (sequence_length, batch_size, ...)
            return np.swapaxes(reshaped, 0, 1)
        
        batch = jax.tree.map(gather_batch, self.data)
        
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