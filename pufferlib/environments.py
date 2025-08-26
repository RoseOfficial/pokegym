"""
PufferLib environments module
Basic environment utilities and base classes
"""

import os
import sys
import importlib
from typing import Dict, Any, Optional

def make_env(env_name: str, **kwargs):
    """
    Create an environment by name
    
    Args:
        env_name: Name of the environment to create
        **kwargs: Additional arguments to pass to the environment
        
    Returns:
        The created environment instance
    """
    # Basic environment factory - can be extended as needed
    if env_name in ['pokemon_red', 'pokegym']:
        try:
            # Try to import the pokegym environment
            from envs.pokegym import environment
            return environment.make(**kwargs)
        except ImportError:
            raise ImportError(f"Could not import environment for {env_name}")
    else:
        raise ValueError(f"Unknown environment: {env_name}")

def list_environments():
    """
    List all available environments
    
    Returns:
        List of environment names
    """
    return ['pokemon_red', 'pokegym']

def environment_info(env_name: str) -> Dict[str, Any]:
    """
    Get information about an environment
    
    Args:
        env_name: Name of the environment
        
    Returns:
        Dictionary with environment information
    """
    info = {
        'name': env_name,
        'available': env_name in list_environments(),
        'description': 'Unknown environment'
    }
    
    if env_name in ['pokemon_red', 'pokegym']:
        info['description'] = 'Pokemon Red gymnasium environment for reinforcement learning'
    
    return info