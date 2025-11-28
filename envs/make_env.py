"""
Environment factory for YouTube 10 ML Display project.
Creates training and evaluation environments with vectorization support.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Callable, Optional
import os
from pathlib import Path

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
        # Method 2: Try importing gymnasium atari environments
        import gymnasium.envs.atari
        return True
    except Exception as e:
        print(f"Warning: Gymnasium atari import failed: {e}")

    return False

# Attempt registration
_ale_registered = register_ale_environments()

from .atari_wrappers import make_atari_env
from .tetris_wrappers import make_tetris_env
from .gameboy_wrappers import make_gameboy_env
from .pyboy_wrappers import make_pyboy_env, get_rom_path


def make_single_env(
    env_id: str,
    config: Dict[str, Any],
    seed: int,
    rank: int = 0,
    monitor_dir: Optional[str] = None,
    record_video: bool = False,
    video_dir: Optional[str] = None,
    video_length: int = 0
) -> Callable[[], gym.Env]:
    """
    Create a single environment factory function.
    
    Args:
        env_id: Gymnasium environment ID
        config: Configuration dictionary
        seed: Random seed
        rank: Environment rank (for multi-processing)
        monitor_dir: Directory to save Monitor logs
        record_video: Whether to record videos
        video_dir: Directory to save videos
        video_length: Length of video episodes to record (0 = all episodes)
        
    Returns:
        Function that creates the environment
    """
    def _init() -> gym.Env:
        # Create environment with appropriate wrappers based on environment type
        if "authentic" in env_id.lower() or "pyboy" in env_id.lower():
            # PyBoy authentic Gameboy emulation
            rom_path = get_rom_path(env_id)
            env = make_pyboy_env(
                rom_path=rom_path,
                config=config,
                seed=seed + rank,
                render_mode="rgb_array" if record_video else None
            )
        elif "tetris" in env_id.lower() and "gymnasium" in env_id.lower():
            # Tetris-Gymnasium (coded remake)
            env = make_tetris_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode="rgb_array" if record_video else None
            )
        elif "gameboy" in env_id.lower() or env_id.endswith("-GameBoy"):
            # stable-retro Gameboy emulation
            env = make_gameboy_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode="rgb_array" if record_video else None
            )
        else:
            # Assume Atari environment
            env = make_atari_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode="rgb_array" if record_video else None
            )
        
        # Add episode statistics tracking
        env = RecordEpisodeStatistics(env)
        
        # Add Monitor wrapper for logging
        if monitor_dir:
            os.makedirs(monitor_dir, exist_ok=True)
            monitor_path = os.path.join(monitor_dir, f"monitor_{rank}.log")
            env = Monitor(env, monitor_path)
        
        # Add video recording if requested
        if record_video and video_dir:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda x: video_length == 0 or x % video_length == 0,
                name_prefix=f"env_{rank}"
            )
        
        return env
    
    return _init


def make_train_env(
    config: Dict[str, Any],
    seed: int = 42,
    rank: int = 0,
    monitor_dir: Optional[str] = None
) -> Callable[[], gym.Env]:
    """
    Create a training environment factory.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        rank: Environment rank for vectorization
        monitor_dir: Directory to save training logs
        
    Returns:
        Function that creates a training environment
    """
    env_id = config['game']['env_id']
    
    return make_single_env(
        env_id=env_id,
        config=config,
        seed=seed,
        rank=rank,
        monitor_dir=monitor_dir,
        record_video=False  # No video recording during training
    )


def make_eval_env(
    config: Dict[str, Any],
    seed: int = 42,
    record_video: bool = False,
    video_dir: Optional[str] = None
) -> gym.Env:
    """
    Create a single evaluation environment.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        record_video: Whether to record evaluation videos
        video_dir: Directory to save videos
        
    Returns:
        Evaluation environment
    """
    env_id = config['game']['env_id']
    
    env_factory = make_single_env(
        env_id=env_id,
        config=config,
        seed=seed,
        rank=0,
        monitor_dir=None,
        record_video=record_video,
        video_dir=video_dir,
        video_length=0  # Record full episodes
    )
    
    return env_factory()


def make_vec_env(
    config: Dict[str, Any],
    seed: int = 42,
    n_envs: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    vec_env_cls: Optional[type] = None
) -> VecEnv:
    """
    Create a vectorized training environment.
    
    Args:
        config: Configuration dictionary
        seed: Base random seed
        n_envs: Number of parallel environments (defaults to config value)
        monitor_dir: Directory to save training logs
        vec_env_cls: Vectorization class (DummyVecEnv or SubprocVecEnv)
        
    Returns:
        Vectorized environment
    """
    if n_envs is None:
        n_envs = config['train']['vec_envs']
    
    if vec_env_cls is None:
        # Use DummyVecEnv on Windows due to Python 3.13 multiprocessing issues
        # SubprocVecEnv causes BrokenPipeError with multiple environments
        import sys
        if sys.platform == 'win32':
            vec_env_cls = DummyVecEnv
        else:
            # Use SubprocVecEnv for better performance with multiple envs on Unix
            vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    
    # Create environment factories
    env_fns = []
    for i in range(n_envs):
        env_fn = make_train_env(
            config=config,
            seed=seed,
            rank=i,
            monitor_dir=monitor_dir
        )
        env_fns.append(env_fn)
    
    # Create vectorized environment
    vec_env = vec_env_cls(env_fns)
    
    return vec_env


def get_env_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about the environment without creating it.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with environment information
    """
    env_id = config['game']['env_id']
    
    # Create a temporary environment to get info
    temp_env = make_eval_env(config, seed=0)
    
    info = {
        'env_id': env_id,
        'observation_space': temp_env.observation_space,
        'action_space': temp_env.action_space,
        'observation_shape': temp_env.observation_space.shape,
        'n_actions': temp_env.action_space.n if hasattr(temp_env.action_space, 'n') else None,
        'action_meanings': getattr(temp_env.unwrapped, 'get_action_meanings', lambda: None)()
    }
    
    temp_env.close()
    return info


def validate_env_creation(config: Dict[str, Any]) -> bool:
    """
    Test that environments can be created successfully.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if environments can be created successfully
    """
    try:
        # Test single evaluation environment
        eval_env = make_eval_env(config, seed=42)
        obs, _ = eval_env.reset()
        action = eval_env.action_space.sample()
        obs, reward, terminated, truncated, info = eval_env.step(action)
        eval_env.close()
        
        # Test vectorized training environment
        vec_env = make_vec_env(config, seed=42, n_envs=2, vec_env_cls=DummyVecEnv)
        obs = vec_env.reset()
        actions = [vec_env.action_space.sample() for _ in range(2)]
        obs, rewards, dones, infos = vec_env.step(actions)
        vec_env.close()
        
        return True
        
    except Exception as e:
        print(f"Environment creation test failed: {e}")
        return False


if __name__ == "__main__":
    # Test environment creation
    from conf.config import load_config
    
    config = load_config()
    
    print("Testing environment creation...")
    if test_env_creation(config):
        print("✅ Environment creation test passed!")
        
        # Print environment info
        env_info = get_env_info(config)
        print(f"\nEnvironment Info:")
        print(f"  ID: {env_info['env_id']}")
        print(f"  Observation shape: {env_info['observation_shape']}")
        print(f"  Action space: {env_info['action_space']}")
        print(f"  Number of actions: {env_info['n_actions']}")
        if env_info['action_meanings']:
            print(f"  Action meanings: {env_info['action_meanings']}")
    else:
        print("❌ Environment creation test failed!")
        exit(1)
