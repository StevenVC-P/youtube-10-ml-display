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
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
    video_length: int = 0,
    render_mode: Optional[str] = None
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
        render_mode: Render mode for the environment (None, "rgb_array", "human")

    Returns:
        Function that creates the environment
    """
    def _init() -> gym.Env:
        # Determine render mode: use explicit parameter if provided, otherwise infer from record_video
        effective_render_mode = render_mode if render_mode is not None else ("rgb_array" if record_video else None)

        # Create environment with appropriate wrappers based on environment type
        if "authentic" in env_id.lower() or "pyboy" in env_id.lower():
            # PyBoy authentic Gameboy emulation
            rom_path = get_rom_path(env_id)
            env = make_pyboy_env(
                rom_path=rom_path,
                config=config,
                seed=seed + rank,
                render_mode=effective_render_mode
            )
        elif "tetris" in env_id.lower() and "gymnasium" in env_id.lower():
            # Tetris-Gymnasium (coded remake)
            env = make_tetris_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode=effective_render_mode
            )
        elif "gameboy" in env_id.lower() or env_id.endswith("-GameBoy"):
            # stable-retro Gameboy emulation
            env = make_gameboy_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode=effective_render_mode
            )
        else:
            # Assume Atari environment
            env = make_atari_env(
                env_id=env_id,
                config=config,
                seed=seed + rank,
                render_mode=effective_render_mode
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
            logger.info(f"[VIDEO DEBUG] Applying RecordVideo wrapper to env rank={rank}")
            logger.info(f"[VIDEO DEBUG] video_folder={video_dir}")
            logger.info(f"[VIDEO DEBUG] video_length={video_length}")
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda x: video_length == 0 or x % video_length == 0,
                name_prefix=f"env_{rank}"
            )
            logger.info(f"[VIDEO DEBUG] RecordVideo wrapper applied successfully")
        
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
    vec_env_cls: Optional[type] = None,
    run_id: Optional[str] = None
) -> VecEnv:
    """
    Create a vectorized training environment.

    Args:
        config: Configuration dictionary
        seed: Base random seed
        n_envs: Number of parallel environments (defaults to config value)
        monitor_dir: Directory to save training logs
        vec_env_cls: Vectorization class (DummyVecEnv or SubprocVecEnv)
        run_id: Training run ID (for video recording directory)

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

    # Check if training video recording is enabled
    training_video_enabled = config.get('training_video', {}).get('enabled', False)
    record_env_index = config.get('training_video', {}).get('record_env_index', 0)

    # Determine if we should enable video recording for this training run
    enable_video_recording = training_video_enabled and run_id is not None

    # DEBUG LOGGING
    logger.info(f"[VIDEO DEBUG] make_vec_env called with run_id={run_id}")
    logger.info(f"[VIDEO DEBUG] training_video_enabled={training_video_enabled}")
    logger.info(f"[VIDEO DEBUG] enable_video_recording={enable_video_recording}")
    logger.info(f"[VIDEO DEBUG] record_env_index={record_env_index}")

    # Create environment factories
    env_fns = []
    for i in range(n_envs):
        # Enable video recording ONLY for the specified environment index
        should_record = enable_video_recording and (i == record_env_index)

        if should_record:
            # Create video directory for this training run
            video_base_dir = config.get('paths', {}).get('videos_training', 'video/training')
            video_dir = os.path.join(video_base_dir, run_id)

            # DEBUG LOGGING
            logger.info(f"[VIDEO DEBUG] Environment {i} will record video")
            logger.info(f"[VIDEO DEBUG] video_base_dir={video_base_dir}")
            logger.info(f"[VIDEO DEBUG] video_dir={video_dir}")

            env_fn = make_single_env(
                env_id=config['game']['env_id'],
                config=config,
                seed=seed,
                rank=i,
                monitor_dir=monitor_dir,
                record_video=True,
                video_dir=video_dir,
                video_length=0,  # Record all episodes
                render_mode="rgb_array"  # Required for video recording
            )
        else:
            # For all other environments, use make_train_env
            # BUT if video recording is enabled, we need to set render_mode="rgb_array"
            # for ALL environments to avoid render_mode mismatch in vectorized env
            if enable_video_recording:
                # Create environment with render_mode="rgb_array" but no video recording
                env_fn = make_single_env(
                    env_id=config['game']['env_id'],
                    config=config,
                    seed=seed,
                    rank=i,
                    monitor_dir=monitor_dir,
                    record_video=False,  # Don't record video for this env
                    video_dir=None,
                    video_length=0,
                    render_mode="rgb_array"  # Match render_mode of recording env
                )
            else:
                # Normal training environment (no render_mode)
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
    if validate_env_creation(config):
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
