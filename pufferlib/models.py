"""
PufferLib models module
Pre-built model architectures for common use cases
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union

class CNNPolicy(nn.Module):
    """
    Convolutional Neural Network policy for image-based environments
    """
    
    def __init__(self, observation_space, action_space, channels=3, features_dim=512):
        """
        Initialize CNN policy
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            channels: Number of input channels
            features_dim: Feature dimension after CNN
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        
        # Determine action dimension
        self.is_continuous = hasattr(action_space, 'shape') and len(action_space.shape) > 0
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # CNN layers for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))  # Ensure fixed output size
        )
        
        # Calculate CNN output size
        cnn_output_size = 64 * 6 * 6  # 2304
        
        # Fully connected layers
        self.features = nn.Sequential(
            nn.Linear(cnn_output_size, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )
        
        # Policy and value heads
        self.policy_head = nn.Linear(features_dim, self.action_dim)
        self.value_head = nn.Linear(features_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        """
        Forward pass
        
        Args:
            obs: Image observations
            
        Returns:
            Policy outputs
        """
        if isinstance(obs, dict):
            # Extract image data from dict
            if 'screen' in obs:
                x = obs['screen']
            elif 'image' in obs:
                x = obs['image']
            else:
                # Use first array-like value
                for key, value in obs.items():
                    if isinstance(value, (torch.Tensor, np.ndarray)) and len(value.shape) >= 3:
                        x = value
                        break
                else:
                    raise ValueError("No suitable image data found in observation dict")
        else:
            x = obs
        
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        x = x.float()
        
        # Ensure correct dimensions [batch, channel, height, width]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        if len(x.shape) == 4 and x.shape[1] > x.shape[-1]:
            # Likely [batch, height, width, channel] -> [batch, channel, height, width]
            x = x.permute(0, 3, 1, 2)
        
        # Normalize if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # CNN forward pass
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.flatten(start_dim=1)
        
        # Fully connected layers
        features = self.features(cnn_features)
        
        # Policy and value outputs
        logits = self.policy_head(features)
        value = self.value_head(features)
        
        return {
            'logits': logits,
            'value': value,
            'features': features
        }

class MultiModalPolicy(nn.Module):
    """
    Multi-modal policy that handles both image and vector observations
    """
    
    def __init__(self, observation_space, action_space):
        """
        Initialize multi-modal policy
        
        Args:
            observation_space: Environment observation space (should be Dict)
            action_space: Environment action space
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Determine action dimension
        self.is_continuous = hasattr(action_space, 'shape') and len(action_space.shape) > 0
        self.action_dim = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Vector encoder
        self.vector_encoder = nn.Sequential(
            nn.Linear(128, 256),  # Assume max 128 vector features
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Combined features
        combined_size = 64 * 4 * 4 + 256  # Image + vector features
        self.combined_encoder = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Output heads
        self.policy_head = nn.Linear(512, self.action_dim)
        self.value_head = nn.Linear(512, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        """
        Forward pass with multi-modal observations
        
        Args:
            obs: Dictionary of observations
            
        Returns:
            Policy outputs
        """
        if not isinstance(obs, dict):
            raise ValueError("MultiModalPolicy expects dict observations")
        
        features = []
        
        # Process image observations
        image_keys = ['screen', 'image', 'rgb', 'observation']
        for key in image_keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    img = torch.from_numpy(img)
                img = img.float()
                
                # Fix dimensions
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                if img.shape[-1] <= 4:  # Likely channels last
                    img = img.permute(0, 3, 1, 2)
                
                # Normalize
                if img.max() > 1.0:
                    img = img / 255.0
                
                # Encode
                img_features = self.image_encoder(img)
                img_features = img_features.flatten(start_dim=1)
                features.append(img_features)
                break
        
        # Process vector observations
        vector_data = []
        vector_keys = ['vector', 'features', 'state', 'info']
        for key, value in obs.items():
            if key not in image_keys:
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    vector_data.append(value.flatten())
        
        if vector_data:
            vector_tensor = torch.cat(vector_data, dim=-1)
            # Pad or truncate to expected size
            if vector_tensor.shape[-1] > 128:
                vector_tensor = vector_tensor[..., :128]
            elif vector_tensor.shape[-1] < 128:
                padding = torch.zeros(*vector_tensor.shape[:-1], 128 - vector_tensor.shape[-1])
                vector_tensor = torch.cat([vector_tensor, padding], dim=-1)
            
            vector_features = self.vector_encoder(vector_tensor)
            features.append(vector_features)
        
        # Combine features
        if not features:
            # Fallback for empty observations
            combined_features = torch.zeros((1, 512))
        else:
            combined_input = torch.cat(features, dim=-1)
            combined_features = self.combined_encoder(combined_input)
        
        # Output heads
        logits = self.policy_head(combined_features)
        value = self.value_head(combined_features)
        
        return {
            'logits': logits,
            'value': value,
            'features': combined_features
        }

# Factory function
def make_model(observation_space, action_space, model_type='auto', **kwargs):
    """
    Create a model based on observation space
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        model_type: Type of model ('auto', 'cnn', 'multimodal', 'mlp')
        **kwargs: Additional arguments
        
    Returns:
        Model instance
    """
    if model_type == 'auto':
        if hasattr(observation_space, 'spaces'):
            # Dict space - use multimodal
            return MultiModalPolicy(observation_space, action_space, **kwargs)
        elif hasattr(observation_space, 'shape') and len(observation_space.shape) >= 3:
            # Image space - use CNN
            return CNNPolicy(observation_space, action_space, **kwargs)
        else:
            # Vector space - use MLP
            from .pytorch import PufferPolicy
            return PufferPolicy(observation_space, action_space, **kwargs)
    elif model_type == 'cnn':
        return CNNPolicy(observation_space, action_space, **kwargs)
    elif model_type == 'multimodal':
        return MultiModalPolicy(observation_space, action_space, **kwargs)
    elif model_type == 'mlp':
        from .pytorch import PufferPolicy
        return PufferPolicy(observation_space, action_space, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")