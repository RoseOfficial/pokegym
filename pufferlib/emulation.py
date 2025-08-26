"""
PufferLib emulation module
Provides environment wrappers and utilities for emulated environments
"""

import gymnasium
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

class GymnasiumPufferEnv(gymnasium.Wrapper):
    """
    Wrapper to make any gymnasium environment compatible with PufferLib
    Handles observation/action space conversion and additional metadata
    """
    
    def __init__(self, env, buf=None):
        """
        Initialize the PufferLib gymnasium wrapper
        
        Args:
            env: The base gymnasium environment
            buf: Optional buffer for shared memory (unused in basic implementation)
        """
        super().__init__(env)
        self.env = env
        self.buf = buf
        self._episode_length = 0
        self._episode_reward = 0.0
        
        # Store original spaces
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        
    def reset(self, **kwargs):
        """Reset the environment and return initial observation"""
        self._episode_length = 0
        self._episode_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
    def step(self, action):
        """
        Step the environment
        
        Args:
            action: Action to take
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self._episode_length += 1
        self._episode_reward += reward
        
        # Add episode statistics to info
        if terminated or truncated:
            info['episode'] = {
                'length': self._episode_length,
                'reward': self._episode_reward
            }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    @property
    def unwrapped(self):
        """Get the base environment"""
        return self.env.unwrapped

class PufferEnv:
    """
    Basic PufferLib environment wrapper
    Provides a common interface for all PufferLib environments
    """
    
    def __init__(self, env, num_agents=1):
        """
        Initialize PufferLib environment
        
        Args:
            env: Base environment
            num_agents: Number of agents (for multi-agent support)
        """
        self.env = env
        self.num_agents = num_agents
        
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)
    
    def step(self, actions):
        """Step environment with actions"""
        return self.env.step(actions)
    
    def render(self, mode='human'):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        return self.env.close()

def make_env(name: str, **kwargs):
    """
    Create an environment by name
    
    Args:
        name: Environment name
        **kwargs: Additional arguments
        
    Returns:
        Environment instance
    """
    # Basic environment factory
    if name in ['pokemon_red', 'pokegym']:
        from envs.pokegym.pokegym import Pokegym
        env = Pokegym(kwargs.get('env_config', {}), 
                     headless=kwargs.get('headless', True))
        return env
    else:
        raise ValueError(f"Unknown environment: {name}")

# Additional utility functions for emulation
def observation_space_shape(space):
    """Get the shape of an observation space"""
    if hasattr(space, 'shape'):
        return space.shape
    elif hasattr(space, 'spaces'):
        # Dict space
        shapes = {}
        for key, subspace in space.spaces.items():
            shapes[key] = observation_space_shape(subspace)
        return shapes
    else:
        return None

def action_space_size(space):
    """Get the size of an action space"""
    if hasattr(space, 'n'):
        return space.n
    elif hasattr(space, 'shape'):
        return space.shape[0] if len(space.shape) > 0 else 1
    else:
        return 1