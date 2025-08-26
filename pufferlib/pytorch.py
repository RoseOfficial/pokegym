"""
PufferLib PyTorch utilities
Neural network utilities and training helpers for PyTorch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

class PufferPolicy(nn.Module):
    """
    Base policy class for PufferLib agents
    """
    
    def __init__(self, observation_space, action_space, hidden_size=512):
        """
        Initialize the policy
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space  
            hidden_size: Hidden layer size
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        
        # Determine if continuous or discrete
        self.is_continuous = hasattr(action_space, 'shape') and len(action_space.shape) > 0
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Get observation dimension
        if hasattr(observation_space, 'shape'):
            self.obs_dim = np.prod(observation_space.shape)
        else:
            # Handle Dict observation spaces
            self.obs_dim = 512  # Default size for complex observations
        
        # Build the network
        self.features = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, self.action_dim)
        
        # Value head  
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, obs, state=None):
        """
        Forward pass
        
        Args:
            obs: Observations
            state: Hidden state (optional)
            
        Returns:
            Policy outputs and value
        """
        # Flatten observations if needed
        if isinstance(obs, dict):
            # Handle dict observations by concatenating
            obs_flat = []
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    obs_flat.append(value.flatten(start_dim=1))
                elif isinstance(value, np.ndarray):
                    obs_flat.append(torch.from_numpy(value.flatten()))
            obs = torch.cat(obs_flat, dim=-1) if obs_flat else torch.zeros((1, self.obs_dim))
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        
        # Ensure tensor is float
        obs = obs.float()
        
        # Flatten if needed
        if len(obs.shape) > 2:
            obs = obs.flatten(start_dim=1)
        elif len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        # Resize if needed
        if obs.shape[-1] != self.obs_dim:
            # Pad or truncate to expected size
            current_size = obs.shape[-1]
            if current_size < self.obs_dim:
                padding = torch.zeros(obs.shape[0], self.obs_dim - current_size)
                obs = torch.cat([obs, padding], dim=-1)
            else:
                obs = obs[:, :self.obs_dim]
        
        # Forward pass
        features = self.features(obs)
        
        # Policy and value outputs
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return {
            'logits': logits,
            'value': value,
            'features': features
        }
    
    def get_action(self, obs, deterministic=False):
        """
        Get action from observation
        
        Args:
            obs: Observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action and log probability
        """
        with torch.no_grad():
            output = self.forward(obs)
            logits = output['logits']
            
            if self.is_continuous:
                # Continuous actions
                action = logits
                log_prob = torch.zeros(action.shape[0])
            else:
                # Discrete actions
                if deterministic:
                    action = torch.argmax(logits, dim=-1)
                    log_prob = torch.log_softmax(logits, dim=-1)[torch.arange(logits.shape[0]), action]
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
            
            return action, log_prob

class RecurrentPolicy(PufferPolicy):
    """
    Recurrent policy with LSTM
    """
    
    def __init__(self, observation_space, action_space, hidden_size=512, num_layers=1):
        super().__init__(observation_space, action_space, hidden_size)
        
        # Replace features with LSTM
        self.lstm = nn.LSTM(self.obs_dim, hidden_size, num_layers, batch_first=True)
        self.num_layers = num_layers
        
        # Keep the heads
        self.policy_head = nn.Linear(hidden_size, self.action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, obs, state=None):
        """Forward pass with LSTM"""
        # Process observation same as base class
        if isinstance(obs, dict):
            obs_flat = []
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    obs_flat.append(value.flatten(start_dim=1))
                elif isinstance(value, np.ndarray):
                    obs_flat.append(torch.from_numpy(value.flatten()))
            obs = torch.cat(obs_flat, dim=-1) if obs_flat else torch.zeros((1, self.obs_dim))
        elif isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        
        obs = obs.float()
        
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dims
        elif len(obs.shape) == 2:
            obs = obs.unsqueeze(1)  # Add sequence dim
        
        # Resize if needed
        if obs.shape[-1] != self.obs_dim:
            current_size = obs.shape[-1]
            if current_size < self.obs_dim:
                padding = torch.zeros(obs.shape[0], obs.shape[1], self.obs_dim - current_size)
                obs = torch.cat([obs, padding], dim=-1)
            else:
                obs = obs[:, :, :self.obs_dim]
        
        # LSTM forward
        if state is None:
            features, state = self.lstm(obs)
        else:
            features, state = self.lstm(obs, state)
        
        # Take the last timestep
        features = features[:, -1, :]
        
        # Policy and value outputs
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return {
            'logits': logits,
            'value': value,
            'features': features,
            'state': state
        }

# Utility functions
def make_policy(observation_space, action_space, recurrent=False, **kwargs):
    """
    Create a policy for the given spaces
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        recurrent: Whether to use recurrent policy
        **kwargs: Additional arguments
        
    Returns:
        Policy instance
    """
    if recurrent:
        return RecurrentPolicy(observation_space, action_space, **kwargs)
    else:
        return PufferPolicy(observation_space, action_space, **kwargs)

def flatten_observation(obs):
    """Flatten a complex observation to a tensor"""
    if isinstance(obs, dict):
        tensors = []
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensors.append(torch.from_numpy(value.flatten()))
            elif isinstance(value, torch.Tensor):
                tensors.append(value.flatten())
        return torch.cat(tensors) if tensors else torch.tensor([])
    elif isinstance(obs, np.ndarray):
        return torch.from_numpy(obs.flatten())
    elif isinstance(obs, torch.Tensor):
        return obs.flatten()
    else:
        return torch.tensor([obs], dtype=torch.float32)