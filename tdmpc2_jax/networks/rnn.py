from typing import Tuple, Dict, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from tdmpc2_jax.common.activations import mish, simnorm

class GRUDynamics(nn.Module):
    """GRU-based dynamics model for TD-MPC2"""
    latent_dim: int
    hidden_dim: int
    action_dim: int
    simnorm_dim: int = 8
    activation: callable = mish
    
    def setup(self):
        # Input projection layers
        self.input_dense = nn.Dense(self.hidden_dim)
        self.input_norm = nn.LayerNorm()
        
        # GRU cell
        self.gru_cell = nn.GRUCell(features=self.hidden_dim)
        
        # Output projection layers
        self.output_dense1 = nn.Dense(self.hidden_dim)
        self.output_norm = nn.LayerNorm()
        self.output_dense2 = nn.Dense(self.latent_dim)
    
    def __call__(self, z, a, carry=None):
        batch_size = z.shape[0]
        
        if carry is None:
            carry = jnp.zeros((batch_size, self.hidden_dim))
        
        # Input processing
        inputs = jnp.concatenate([z, a], axis=-1)
        inputs = self.input_dense(inputs)
        inputs = self.activation(inputs)
        
        # GRU forward pass - returns (new_carry, gru_output)
        new_carry, gru_output = self.gru_cell(carry, inputs)
        
        # Option 1: Use gru_output (typically same as new_carry for GRU)
        # Option 2: Use new_carry (hidden state)
        # For GRU, they're usually the same, so either works
        
        # Project to latent space with SimNorm
        output = self.output_dense1(gru_output)  # or new_carry
        output = self.activation(output)
        output = self.output_dense2(output)
        next_z = simnorm(output, simplex_dim=self.simnorm_dim)
        
        return next_z, new_carry