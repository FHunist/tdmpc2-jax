from typing import Tuple, Dict, Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
from functools import partial
from tdmpc2_jax.common.activations import mish, simnorm

class GRUDynamics(nn.Module):
    """GRU-based dynamics model for TD-MPC2"""
    latent_dim: int        # Output latent dimension
    hidden_dim: int        # GRU hidden dimension
    action_dim: int        # Action dimension
    simnorm_dim: int = 8   # SimNorm dimension for compatibility
    activation: callable = mish  # Activation function
    
    def setup(self):
        # Input projection layer (latent + action to GRU input)
        self.input_projector = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.activation,
            nn.LayerNorm()
        ])
        
        # GRU cell
        self.gru_cell = nn.GRUCell(features=self.hidden_dim)
        
        # Output projection (GRU hidden to latent with SimNorm for compatibility)
        self.output_projector = nn.Sequential([
            nn.Dense(self.hidden_dim),
            self.activation,
            nn.LayerNorm(),
            nn.Dense(self.latent_dim),
            partial(simnorm, simplex_dim=self.simnorm_dim)
        ])
    
    def __call__(self, z: jnp.ndarray, a: jnp.ndarray, 
                 carry: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass of GRU dynamics
        
        Args:
            z: Current latent state [batch_size, latent_dim]
            a: Action [batch_size, action_dim]
            carry: Previous GRU hidden state [batch_size, hidden_dim]
            
        Returns:
            next_z: Predicted next latent state [batch_size, latent_dim]
            new_carry: Updated GRU hidden state [batch_size, hidden_dim]
        """
        batch_size = z.shape[0]
        
        # Initialize carry (hidden state) if not provided
        if carry is None:
            carry = jnp.zeros((batch_size, self.hidden_dim))
        
        # Concatenate latent and action
        inputs = jnp.concatenate([z, a], axis=-1)
        
        # Process through input projector
        inputs = self.input_projector(inputs)
        
        # Update GRU state
        new_carry = self.gru_cell(carry, inputs)
        
        # Project to latent space
        next_z = self.output_projector(new_carry)
        
        return next_z, new_carry