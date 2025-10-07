#!/usr/bin/env python3
"""
Algorithm factory for creating RL algorithms (PPO, DQN, etc.)
"""

import os
from typing import Dict, Any, Union
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
import torch


def print_system_info():
    """Print system and library information at startup"""
    print("=" * 60)
    print(">> YouTube 10 ML Display - Algorithm Factory")
    print("=" * 60)
    
    # Library versions
    import stable_baselines3
    import gymnasium
    import numpy as np
    import cv2
    
    print(f">> Library Versions:")
    print(f"  - Stable-Baselines3: {stable_baselines3.__version__}")
    print(f"  - Gymnasium: {gymnasium.__version__}")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - NumPy: {np.__version__}")
    print(f"  - OpenCV: {cv2.__version__}")

    # CUDA information
    if torch.cuda.is_available():
        print(f">> CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(">> CUDA Not Available - Using CPU")
    
    print("=" * 60)


def create_algo(config: Dict[str, Any], vec_env: VecEnv, tensorboard_log: str = None) -> Union[PPO, DQN]:
    """
    Create and configure RL algorithm based on config.
    
    Args:
        config: Configuration dictionary
        vec_env: Vectorized environment
        tensorboard_log: Path for TensorBoard logging
        
    Returns:
        Configured RL algorithm (PPO or DQN)
    """
    train_config = config['train']
    algo_name = train_config['algo'].lower()
    
    # Common parameters
    common_params = {
        'env': vec_env,
        'tensorboard_log': tensorboard_log,
        'seed': config.get('seed', 42),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    if algo_name == 'ppo':
        return create_ppo(train_config, **common_params)
    elif algo_name == 'dqn':
        return create_dqn(train_config, **common_params)
    else:
        raise ValueError(f"Unsupported algorithm: {algo_name}")


def create_ppo(train_config: Dict[str, Any], **kwargs) -> PPO:
    """
    Create and configure PPO algorithm.
    
    Args:
        train_config: Training configuration
        **kwargs: Common parameters (env, tensorboard_log, seed, device)
        
    Returns:
        Configured PPO algorithm
    """
    # PPO-specific parameters from config
    ppo_params = {
        'policy': train_config.get('policy', 'CnnPolicy'),
        'learning_rate': train_config.get('learning_rate', 2.5e-4),
        'n_steps': train_config.get('n_steps', 128),
        'batch_size': train_config.get('batch_size', 256),
        'n_epochs': train_config.get('n_epochs', 10),
        'gamma': train_config.get('gamma', 0.99),
        'gae_lambda': train_config.get('gae_lambda', 0.95),
        'clip_range': train_config.get('clip_range', 0.1),
        'clip_range_vf': train_config.get('clip_range_vf', None),
        'ent_coef': train_config.get('ent_coef', 0.01),
        'vf_coef': train_config.get('vf_coef', 0.5),
        'max_grad_norm': train_config.get('max_grad_norm', 0.5),
        'target_kl': train_config.get('target_kl', None),
        'stats_window_size': train_config.get('stats_window_size', 100),
        'verbose': 1
    }
    
    # Merge with common parameters
    ppo_params.update(kwargs)
    
    print(f"🤖 Creating PPO Algorithm:")
    print(f"  • Policy: {ppo_params['policy']}")
    print(f"  • Learning Rate: {ppo_params['learning_rate']}")
    print(f"  • N Steps: {ppo_params['n_steps']}")
    print(f"  • Batch Size: {ppo_params['batch_size']}")
    print(f"  • Gamma: {ppo_params['gamma']}")
    print(f"  • Device: {ppo_params['device']}")
    
    return PPO(**ppo_params)


def create_dqn(train_config: Dict[str, Any], **kwargs) -> DQN:
    """
    Create and configure DQN algorithm.
    
    Args:
        train_config: Training configuration
        **kwargs: Common parameters (env, tensorboard_log, seed, device)
        
    Returns:
        Configured DQN algorithm
    """
    # DQN-specific parameters from config
    dqn_params = {
        'policy': train_config.get('policy', 'CnnPolicy'),
        'learning_rate': train_config.get('learning_rate', 1e-4),
        'buffer_size': train_config.get('buffer_size', 100000),
        'learning_starts': train_config.get('learning_starts', 50000),
        'batch_size': train_config.get('batch_size', 32),
        'tau': train_config.get('tau', 1.0),
        'gamma': train_config.get('gamma', 0.99),
        'train_freq': train_config.get('train_freq', 4),
        'gradient_steps': train_config.get('gradient_steps', 1),
        'target_update_interval': train_config.get('target_update_interval', 10000),
        'exploration_fraction': train_config.get('exploration_fraction', 0.1),
        'exploration_initial_eps': train_config.get('exploration_initial_eps', 1.0),
        'exploration_final_eps': train_config.get('exploration_final_eps', 0.05),
        'max_grad_norm': train_config.get('max_grad_norm', 10),
        'stats_window_size': train_config.get('stats_window_size', 100),
        'verbose': 1
    }
    
    # Merge with common parameters
    dqn_params.update(kwargs)
    
    print(f"🤖 Creating DQN Algorithm:")
    print(f"  • Policy: {dqn_params['policy']}")
    print(f"  • Learning Rate: {dqn_params['learning_rate']}")
    print(f"  • Buffer Size: {dqn_params['buffer_size']}")
    print(f"  • Batch Size: {dqn_params['batch_size']}")
    print(f"  • Device: {dqn_params['device']}")
    
    return DQN(**dqn_params)


def get_algorithm_info(algo) -> Dict[str, Any]:
    """
    Extract information about the algorithm for logging.
    
    Args:
        algo: RL algorithm instance
        
    Returns:
        Dictionary with algorithm information
    """
    info = {
        'algorithm': algo.__class__.__name__,
        'policy': algo.policy.__class__.__name__,
        'device': str(algo.device),
        'learning_rate': getattr(algo, 'learning_rate', 'N/A'),
    }
    
    # Algorithm-specific info
    if isinstance(algo, PPO):
        info.update({
            'n_steps': algo.n_steps,
            'batch_size': algo.batch_size,
            'n_epochs': algo.n_epochs,
            'gamma': algo.gamma,
            'gae_lambda': algo.gae_lambda,
            'clip_range': algo.clip_range,
            'ent_coef': algo.ent_coef,
            'vf_coef': algo.vf_coef
        })
    elif isinstance(algo, DQN):
        info.update({
            'buffer_size': algo.buffer_size,
            'batch_size': algo.batch_size,
            'gamma': algo.gamma,
            'tau': algo.tau,
            'target_update_interval': algo.target_update_interval
        })
    
    return info


if __name__ == "__main__":
    # Test the algorithm factory
    print_system_info()

    # Example usage
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        from conf.config import load_config
        from envs.make_env import make_vec_env

        config = load_config('conf/config.yaml')
        vec_env = make_vec_env(config, n_envs=2, seed=42)

        algo = create_algo(config, vec_env, tensorboard_log='logs/tb/test')
        info = get_algorithm_info(algo)

        print(f"\n>> Algorithm Info:")
        for key, value in info.items():
            print(f"  - {key}: {value}")

        vec_env.close()
        print("\n>> Algorithm factory test completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
