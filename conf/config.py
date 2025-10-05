"""
Config loader for YouTube 10 ML Display project.
Validates and echoes resolved config and library versions on startup.
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Dictionary containing the loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        KeyError: If required config keys are missing
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required top-level keys
    required_keys = ['project_name', 'seed', 'paths', 'game', 'train', 'recording', 'stream', 'render']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Required config key missing: {key}")
    
    # Validate required nested keys
    required_paths = ['videos_milestones', 'videos_eval', 'videos_parts', 'logs_tb', 'models', 'ffmpeg_path']
    for path_key in required_paths:
        if path_key not in config['paths']:
            raise KeyError(f"Required path key missing: paths.{path_key}")
    
    required_game = ['env_id', 'frame_stack', 'resize', 'grayscale', 'max_skip']
    for game_key in required_game:
        if game_key not in config['game']:
            raise KeyError(f"Required game key missing: game.{game_key}")
    
    required_train = ['algo', 'policy', 'total_timesteps', 'vec_envs', 'checkpoint_every_sec']
    for train_key in required_train:
        if train_key not in config['train']:
            raise KeyError(f"Required train key missing: train.{train_key}")
    
    return config


def echo_config_and_versions(config: Dict[str, Any]) -> None:
    """
    Echo the resolved configuration and library versions to stdout.
    
    Args:
        config: The loaded configuration dictionary
    """
    print("=" * 60)
    print("YouTube 10 ML Display - Configuration & Environment")
    print("=" * 60)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Library versions
    try:
        import gymnasium
        print(f"Gymnasium: {gymnasium.__version__}")
    except ImportError:
        print("Gymnasium: Not installed")
    
    try:
        import stable_baselines3
        print(f"Stable-Baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print("Stable-Baselines3: Not installed")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import numpy
        print(f"NumPy: {numpy.__version__}")
    except ImportError:
        print("NumPy: Not installed")
    
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)
    
    # Project info
    print(f"Project: {config['project_name']}")
    print(f"Seed: {config['seed']}")
    
    # Paths (resolve to absolute paths)
    print(f"\nPaths:")
    for key, path in config['paths'].items():
        abs_path = Path(path).resolve()
        print(f"  {key}: {abs_path}")
    
    # Game settings
    print(f"\nGame Settings:")
    print(f"  Environment: {config['game']['env_id']}")
    print(f"  Frame Stack: {config['game']['frame_stack']}")
    print(f"  Resize: {config['game']['resize']}")
    print(f"  Grayscale: {config['game']['grayscale']}")
    print(f"  Max Skip: {config['game']['max_skip']}")
    
    # Training settings
    print(f"\nTraining Settings:")
    print(f"  Algorithm: {config['train']['algo']}")
    print(f"  Policy: {config['train']['policy']}")
    print(f"  Total Timesteps: {config['train']['total_timesteps']:,}")
    print(f"  Vector Envs: {config['train']['vec_envs']}")
    print(f"  Checkpoint Every: {config['train']['checkpoint_every_sec']}s")
    
    # Stream settings
    print(f"\nStream Settings:")
    print(f"  Enabled: {config['stream']['enabled']}")
    print(f"  Grid: {config['stream']['grid']}")
    print(f"  FPS: {config['stream']['fps']}")
    print(f"  Save Mode: {config['stream']['save_mode']}")
    
    print("=" * 60)


def get_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    Load config and echo information. Main entry point for config loading.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        Dictionary containing the loaded configuration
    """
    config = load_config(config_path)
    echo_config_and_versions(config)
    return config


if __name__ == "__main__":
    # Test the config loader
    try:
        config = get_config()
        print("\n✅ Config loaded successfully!")
    except Exception as e:
        print(f"\n❌ Config loading failed: {e}")
        sys.exit(1)
