"""
Atari environment wrappers for YouTube 10 ML Display project.
Implements standard Atari preprocessing pipeline: MaxAndSkip, GrayScale, Resize, FrameStack.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import (
    MaxAndSkipObservation,
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation
)
from typing import Dict, Any, Tuple
import cv2

# Register ALE environments with multiple approaches
def register_ale_environments():
    """Register ALE environments using multiple fallback methods"""
    try:
        # Method 1: Standard registration
        import ale_py
        gym.register_envs(ale_py)
        return True
    except ImportError:
        print("Warning: ale_py not available")
    except Exception as e:
        print(f"Warning: Standard ALE registration failed: {e}")

    try:
        # Method 2: Direct import and registration
        import ale_py.env
        from ale_py import ALEInterface
        # Force registration
        import gymnasium.envs.atari
        return True
    except Exception as e:
        print(f"Warning: Direct ALE registration failed: {e}")

    try:
        # Method 3: Manual environment registration
        from gymnasium.envs.registration import register

        # Register BreakoutNoFrameskip-v4 manually if not available
        try:
            gym.make('BreakoutNoFrameskip-v4')
        except:
            register(
                id='BreakoutNoFrameskip-v4',
                entry_point='ale_py.env:AtariEnv',
                kwargs={'game': 'breakout', 'frameskip': 1},
                max_episode_steps=108000,
                nondeterministic=True,
            )
            print("Manually registered BreakoutNoFrameskip-v4")
        return True
    except Exception as e:
        print(f"Warning: Manual registration failed: {e}")

    return False

# Attempt registration
_ale_registered = register_ale_environments()


class MaxAndSkipWrapper(gym.Wrapper):
    """
    Max-and-skip wrapper that takes the maximum over the last 2 frames
    and skips intermediate frames.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4):
        """
        Args:
            env: Environment to wrap
            skip: Number of frames to skip (repeat action)
        """
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
    
    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.fill(0)
        self._obs_buffer[0] = obs
        return obs, info


class GrayScaleWrapper(gym.ObservationWrapper):
    """
    Convert RGB observation to grayscale.
    """
    
    def __init__(self, env: gym.Env, keep_dim: bool = False):
        """
        Args:
            env: Environment to wrap
            keep_dim: Whether to keep the channel dimension (shape becomes H,W,1)
        """
        super().__init__(env)
        self.keep_dim = keep_dim
        
        assert isinstance(env.observation_space, spaces.Box), "Expected Box observation space"
        old_shape = env.observation_space.shape
        
        if keep_dim:
            new_shape = old_shape[:2] + (1,)
        else:
            new_shape = old_shape[:2]
            
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, observation):
        """Convert RGB to grayscale."""
        # Use OpenCV for consistent grayscale conversion
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)
        return gray


class ResizeWrapper(gym.ObservationWrapper):
    """
    Resize observation to specified dimensions.
    """
    
    def __init__(self, env: gym.Env, size: Tuple[int, int] = (84, 84)):
        """
        Args:
            env: Environment to wrap
            size: Target size as (height, width)
        """
        super().__init__(env)
        self.size = size
        
        assert isinstance(env.observation_space, spaces.Box), "Expected Box observation space"
        old_shape = env.observation_space.shape
        
        if len(old_shape) == 3:
            new_shape = self.size + (old_shape[2],)
        else:
            new_shape = self.size
            
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, observation):
        """Resize observation using OpenCV."""
        # OpenCV expects (width, height) but we store (height, width)
        resized = cv2.resize(observation, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        
        # If original was 3D but resize made it 2D, add back the channel dimension
        if len(self.observation_space.shape) == 3 and len(resized.shape) == 2:
            resized = np.expand_dims(resized, axis=-1)
            
        return resized


class FrameStackWrapper(gym.Wrapper):
    """
    Stack the last N frames along a new axis.
    """
    
    def __init__(self, env: gym.Env, num_stack: int = 4):
        """
        Args:
            env: Environment to wrap
            num_stack: Number of frames to stack
        """
        super().__init__(env)
        self.num_stack = num_stack
        
        assert isinstance(env.observation_space, spaces.Box), "Expected Box observation space"
        old_shape = env.observation_space.shape
        
        # Stack along the last axis
        new_shape = old_shape + (num_stack,)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
        
        # Initialize frame buffer
        self.frames = np.zeros(new_shape, dtype=env.observation_space.dtype)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill all frames with the initial observation
        for i in range(self.num_stack):
            self.frames[..., i] = obs
        return self.frames.copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Shift frames and add new observation
        self.frames[..., :-1] = self.frames[..., 1:]
        self.frames[..., -1] = obs
        return self.frames.copy(), reward, terminated, truncated, info


def apply_atari_wrappers(env: gym.Env, config: Dict[str, Any]) -> gym.Env:
    """
    Apply the standard Atari preprocessing pipeline based on config.
    
    Args:
        env: Base Atari environment
        config: Configuration dictionary with game settings
        
    Returns:
        Wrapped environment
    """
    game_config = config['game']
    
    # 1. Max-and-skip wrapper
    if game_config.get('max_skip', 4) > 1:
        env = MaxAndSkipWrapper(env, skip=game_config['max_skip'])
    
    # 2. Grayscale conversion
    if game_config.get('grayscale', True):
        env = GrayScaleWrapper(env, keep_dim=True)
    
    # 3. Resize frames
    if 'resize' in game_config:
        size = tuple(game_config['resize'])  # [height, width]
        env = ResizeWrapper(env, size=size)
    
    # 4. Frame stacking
    if game_config.get('frame_stack', 4) > 1:
        env = FrameStackWrapper(env, num_stack=game_config['frame_stack'])
    
    return env


def make_atari_env(env_id: str, config: Dict[str, Any], seed: int = None, **kwargs) -> gym.Env:
    """
    Create and wrap an Atari environment with standard preprocessing.
    
    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/Breakout-v5")
        config: Configuration dictionary
        seed: Random seed
        **kwargs: Additional arguments for gym.make()
        
    Returns:
        Wrapped Atari environment
    """
    # Create base environment
    env = gym.make(env_id, **kwargs)
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply Atari wrappers
    env = apply_atari_wrappers(env, config)
    
    return env
