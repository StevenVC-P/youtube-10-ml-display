"""
Tetris environment wrappers for reinforcement learning.

This module provides wrappers for Tetris environments to make them compatible
with our training pipeline.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional
from gymnasium.spaces import Box


class TetrisFrameStackWrapper(gym.Wrapper):
    """
    Stack frames for Tetris environment.
    """
    
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        
        # Get original observation space
        orig_obs_space = env.observation_space
        
        # Create new observation space with stacked frames
        if isinstance(orig_obs_space, Box):
            low = np.repeat(orig_obs_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(orig_obs_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = Box(low=low, high=high, dtype=orig_obs_space.dtype)
        else:
            raise ValueError(f"Unsupported observation space: {orig_obs_space}")
        
        self.frames = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = np.stack([obs] * self.num_stack, axis=0)
        return self.frames, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Shift frames and add new observation
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        
        return self.frames, reward, terminated, truncated, info


class TetrisResizeWrapper(gym.ObservationWrapper):
    """
    Resize Tetris observations to specified dimensions.
    """
    
    def __init__(self, env: gym.Env, size: tuple = (84, 84)):
        super().__init__(env)
        self.size = size
        
        # Update observation space
        if isinstance(env.observation_space, Box):
            if len(env.observation_space.shape) == 3:  # (H, W, C)
                new_shape = (*size, env.observation_space.shape[2])
            elif len(env.observation_space.shape) == 2:  # (H, W)
                new_shape = size
            else:
                raise ValueError(f"Unsupported observation shape: {env.observation_space.shape}")
            
            self.observation_space = Box(
                low=0, high=255, shape=new_shape, dtype=np.uint8
            )
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    def observation(self, obs):
        import cv2
        if len(obs.shape) == 3:  # (H, W, C)
            return cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        elif len(obs.shape) == 2:  # (H, W)
            return cv2.resize(obs, self.size, interpolation=cv2.INTER_AREA)
        else:
            raise ValueError(f"Unsupported observation shape: {obs.shape}")


class TetrisDictToImageWrapper(gym.ObservationWrapper):
    """Convert Tetris dictionary observation to a single image."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        # Check if observation space is Dict
        if not isinstance(env.observation_space, gym.spaces.Dict):
            # If not a dict, pass through unchanged
            return

        # For Tetris, we'll use the 'board' as the main observation
        # The board is typically (24, 18) representing the game field
        board_space = env.observation_space['board']

        # Create a single Box observation space from the board
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(board_space.shape[0], board_space.shape[1], 1),  # Add channel dimension
            dtype=np.uint8
        )

    def observation(self, observation):
        """Convert dictionary observation to single image."""
        if isinstance(observation, dict):
            # Extract the board and normalize to 0-255 range
            board = observation['board']

            # Normalize from 0-9 range to 0-255 range
            normalized_board = (board * 255 / 9).astype(np.uint8)

            # Add channel dimension to make it (height, width, 1)
            return np.expand_dims(normalized_board, axis=2)
        else:
            # Not a dict, pass through unchanged
            return observation


class TetrisGrayscaleWrapper(gym.ObservationWrapper):
    """
    Convert Tetris observations to grayscale.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # Update observation space
        if isinstance(env.observation_space, Box):
            if len(env.observation_space.shape) == 3:  # (H, W, C)
                new_shape = env.observation_space.shape[:2]  # (H, W)
                self.observation_space = Box(
                    low=0, high=255, shape=new_shape, dtype=np.uint8
                )
            else:
                # Already grayscale or different format
                self.observation_space = env.observation_space
        else:
            raise ValueError(f"Unsupported observation space: {env.observation_space}")
    
    def observation(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:  # RGB
            import cv2
            return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            # Already grayscale or different format
            return obs


def make_tetris_env(env_id: str, config: Dict[str, Any], seed: int = None, **kwargs) -> gym.Env:
    """
    Create and wrap a Tetris environment with standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "tetris_gymnasium/Tetris")
        config: Configuration dictionary
        seed: Random seed
        **kwargs: Additional arguments for gym.make()
        
    Returns:
        Wrapped Tetris environment
    """
    # Import tetris_gymnasium to register the environment
    from tetris_gymnasium.envs.tetris import Tetris

    # Create base environment
    env = gym.make(env_id, **kwargs)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    # Get game configuration
    game_config = config.get('game', {})

    # Convert dictionary observation to image if needed
    env = TetrisDictToImageWrapper(env)

    # Apply grayscale conversion if requested
    if game_config.get('grayscale', True):
        env = TetrisGrayscaleWrapper(env)
    
    # Apply resizing if requested
    resize = game_config.get('resize', [84, 84])
    if resize:
        env = TetrisResizeWrapper(env, tuple(resize))
    
    # Apply frame stacking if requested
    frame_stack = game_config.get('frame_stack', 4)
    if frame_stack > 1:
        env = TetrisFrameStackWrapper(env, frame_stack)
    
    return env
