import numpy as np
from typing import *
import jax
from jaxtyping import PyTree


class RollingSequentialBuffer():

  def __init__(self,
               capacity: int,
               dummy_input: Dict,
               num_envs: int = 1,
               seed: Optional[int] = None,
               ):
    """
    Rolling sequential replay buffer with support for parallel environments.
    Only keeps the most recent 'capacity' transitions per environment and
    overwrites old data in a FIFO manner.

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
    self.data = jax.tree.map(lambda x: np.zeros(
        (capacity,) + np.asarray(x).shape, np.asarray(x).dtype), dummy_input)

    # Size and index counter for each environment buffer
    self.sizes = np.zeros(num_envs, dtype=int)
    self.current_inds = np.zeros(num_envs, dtype=int)
    self.is_full = np.zeros(num_envs, dtype=bool)

    self.np_random = np.random.default_rng(seed=seed)

  def insert(self,
             data: PyTree,
             env_mask: Optional[np.ndarray] = None
             ) -> None:
    """
    Insert data into the buffer, overwriting the oldest data if the buffer is full

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

    def masked_set(x, y):
      x[self.current_inds, env_mask] = y[env_mask]
    jax.tree.map(masked_set, self.data, data)

    # Update buffer state
    self.current_inds[env_mask] = (
        self.current_inds[env_mask] + 1
    ) % self.capacity
    
    # Update sizes and full flags
    for i in np.where(env_mask)[0]:
      if self.sizes[i] < self.capacity:
        self.sizes[i] += 1
      else:
        self.is_full[i] = True

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

    # Sample envs and start indices
    env_inds = self.np_random.integers(
        low=0, high=self.num_envs,
        size=batch_size
    )
    start_inds = self.np_random.integers(
        low=0, high=self.sizes[env_inds] - sequence_length,
        size=batch_size,
        endpoint=True,
    )
    # Handle wrapping: For wrapped buffers, we define the current pointer index as 0 to avoid stepping into an unrelated trajectory
    start_inds = (
        start_inds - (self.sizes[env_inds] - self.current_inds[env_inds])
    ) % self.capacity

    # Sample from buffer and convert from (batch, time, *) to (time, batch, *)
    sequence_inds = start_inds[:, None] + np.arange(sequence_length)
    batch = jax.tree.map(
        lambda x: np.swapaxes(x[sequence_inds, env_inds[:, None]], 0, 1),
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