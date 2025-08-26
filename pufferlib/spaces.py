"""
PufferLib spaces module - wrapper around gymnasium spaces
Provides utility functions for space manipulation and conversion
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Space

def flatten_space(space):
    """Flatten a nested space into a single Box space"""
    if isinstance(space, spaces.Dict):
        # Handle Dict spaces by flattening all subspaces
        total_size = 0
        low_values = []
        high_values = []
        
        for key, subspace in space.spaces.items():
            if isinstance(subspace, spaces.Box):
                flat_size = np.prod(subspace.shape)
                total_size += flat_size
                low_values.extend(subspace.low.flatten())
                high_values.extend(subspace.high.flatten())
            elif isinstance(subspace, spaces.Discrete):
                total_size += subspace.n
                low_values.extend([0] * subspace.n)
                high_values.extend([1] * subspace.n)
            else:
                # For other space types, approximate with Box
                flat_size = 1
                total_size += flat_size
                low_values.extend([0] * flat_size)
                high_values.extend([1] * flat_size)
        
        return spaces.Box(
            low=np.array(low_values, dtype=np.float32),
            high=np.array(high_values, dtype=np.float32),
            shape=(total_size,),
            dtype=np.float32
        )
    
    elif isinstance(space, spaces.Box):
        # Already flat if Box
        return spaces.Box(
            low=space.low.flatten(),
            high=space.high.flatten(), 
            shape=(np.prod(space.shape),),
            dtype=space.dtype
        )
    
    elif isinstance(space, spaces.Discrete):
        # Convert discrete to one-hot Box
        return spaces.Box(
            low=0,
            high=1,
            shape=(space.n,),
            dtype=np.float32
        )
    
    else:
        # Default handling for other space types
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

def unflatten_space(flat_space, original_space):
    """Unflatten a flat space back to its original structure"""
    # This is a complex operation that would require storing the original structure
    # For now, just return the flat space
    return flat_space

def space_shape(space):
    """Get the total shape of a space"""
    if isinstance(space, spaces.Dict):
        total = 0
        for subspace in space.spaces.values():
            total += np.prod(space_shape(subspace))
        return (total,)
    elif isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        return (space.n,)
    else:
        return (1,)

def space_dtype(space):
    """Get the dtype of a space"""
    if isinstance(space, spaces.Dict):
        # Return float32 as default for mixed types
        return np.float32
    elif isinstance(space, spaces.Box):
        return space.dtype
    elif isinstance(space, spaces.Discrete):
        return np.int64
    else:
        return np.float32

# Export commonly used spaces from gymnasium
Box = spaces.Box
Discrete = spaces.Discrete
Dict = spaces.Dict
Tuple = spaces.Tuple
MultiDiscrete = spaces.MultiDiscrete
MultiBinary = spaces.MultiBinary