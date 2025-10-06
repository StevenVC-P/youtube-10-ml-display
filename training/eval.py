#!/usr/bin/env python3
"""
Standalone evaluation script for trained RL models.
Loads a checkpoint and records K evaluation episodes with video recording.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordVideo

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from conf.config import load_config
from envs.make_env import make_eval_env
from agents.algo_factory import print_system_info


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL model and record episodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.zip file)"
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to evaluate"
    )
    
    parser.add_argument(
        "--seconds",
        type=int,
        default=120,
        help="Maximum evaluation time in seconds (0 = no limit)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for videos (defaults to config paths.videos_eval)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for evaluation"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic policy (no exploration)"
    )
    
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording"
    )
    
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=silent, 1=info, 2=debug)"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config: Dict[str, Any], verbose: int = 1) -> PPO:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        verbose: Verbosity level
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if verbose >= 1:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Determine algorithm type from config
    algo_type = config['train']['algo'].lower()
    
    try:
        if algo_type == "ppo":
            model = PPO.load(checkpoint_path)
        elif algo_type == "dqn":
            model = DQN.load(checkpoint_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algo_type}")
        
        if verbose >= 1:
            print(f"✅ Model loaded successfully ({algo_type.upper()})")
            
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def create_eval_environment(
    config: Dict[str, Any], 
    output_dir: str, 
    record_video: bool = True,
    verbose: int = 1
) -> tuple:
    """
    Create evaluation environment with optional video recording.
    
    Args:
        config: Configuration dictionary
        output_dir: Directory for video output
        record_video: Whether to record videos
        verbose: Verbosity level
        
    Returns:
        Tuple of (environment, video_dir)
    """
    if record_video:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        if verbose >= 1:
            print(f"📹 Video recording enabled: {output_dir}")
        
        # Create environment with video recording
        env = make_eval_env(
            config=config,
            seed=42,
            record_video=True,
            video_dir=output_dir
        )
        
        return env, output_dir
    else:
        if verbose >= 1:
            print("🚫 Video recording disabled")
        
        # Create environment without video recording
        env = make_eval_env(
            config=config,
            seed=42,
            record_video=False
        )
        
        return env, None


def evaluate_model(
    model,
    env,
    episodes: int,
    max_seconds: int,
    deterministic: bool = True,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Evaluate the model for specified episodes or time limit.
    
    Args:
        model: Trained model
        env: Evaluation environment
        episodes: Number of episodes to run
        max_seconds: Maximum evaluation time (0 = no limit)
        deterministic: Use deterministic policy
        verbose: Verbosity level
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose >= 1:
        print(f"🎮 Starting evaluation:")
        print(f"  • Episodes: {episodes}")
        print(f"  • Max time: {max_seconds}s" if max_seconds > 0 else "  • Max time: unlimited")
        print(f"  • Deterministic: {deterministic}")
    
    start_time = time.time()
    episode_rewards = []
    episode_lengths = []
    episodes_completed = 0
    
    try:
        for episode in range(episodes):
            # Check time limit
            if max_seconds > 0 and (time.time() - start_time) > max_seconds:
                if verbose >= 1:
                    print(f"⏰ Time limit reached ({max_seconds}s)")
                break
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            if verbose >= 2:
                print(f"🎯 Episode {episode + 1}/{episodes} started")
            
            # Run episode
            while not done:
                # Check time limit during episode
                if max_seconds > 0 and (time.time() - start_time) > max_seconds:
                    if verbose >= 1:
                        print(f"⏰ Time limit reached during episode {episode + 1}")
                    break
                
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            # Record episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episodes_completed += 1
            
            if verbose >= 1:
                print(f"📊 Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    except KeyboardInterrupt:
        if verbose >= 1:
            print("\n⚠️ Evaluation interrupted by user")
    
    finally:
        env.close()
    
    # Calculate statistics
    total_time = time.time() - start_time
    
    results = {
        "episodes_completed": episodes_completed,
        "total_time": total_time,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
        "std_reward": np.std(episode_rewards) if episode_rewards else 0.0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0.0,
        "std_length": np.std(episode_lengths) if episode_lengths else 0.0,
        "total_reward": sum(episode_rewards),
        "total_steps": sum(episode_lengths)
    }
    
    return results


def print_results(results: Dict[str, Any], video_dir: Optional[str] = None, verbose: int = 1):
    """Print evaluation results summary."""
    if verbose == 0:
        return
    
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Episodes completed: {results['episodes_completed']}")
    print(f"Total time: {results['total_time']:.1f}s")
    print(f"Total steps: {results['total_steps']}")
    print(f"Total reward: {results['total_reward']:.2f}")
    
    if results['episodes_completed'] > 0:
        print(f"\nPer-episode statistics:")
        print(f"  • Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  • Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
        
        if results['episodes_completed'] > 1:
            print(f"  • Min reward: {min(results['episode_rewards']):.2f}")
            print(f"  • Max reward: {max(results['episode_rewards']):.2f}")
    
    if video_dir and os.path.exists(video_dir):
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if video_files:
            print(f"\n🎬 Videos saved:")
            for video_file in sorted(video_files):
                video_path = os.path.join(video_dir, video_file)
                print(f"  • {os.path.abspath(video_path)}")
        else:
            print(f"\n⚠️ No video files found in {video_dir}")
    
    print("=" * 60)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Print system info
    if args.verbose >= 1:
        print_system_info()

    try:
        # Load configuration
        config = load_config(args.config)
        if args.verbose >= 2:
            print(f"📋 Configuration loaded from: {args.config}")

        # Determine output directory
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = config['paths']['videos_eval']

        # Load model
        model = load_model(args.checkpoint, config, args.verbose)

        # Create evaluation environment
        env, video_dir = create_eval_environment(
            config=config,
            output_dir=output_dir,
            record_video=not args.no_video,
            verbose=args.verbose
        )

        # Set random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Run evaluation
        results = evaluate_model(
            model=model,
            env=env,
            episodes=args.episodes,
            max_seconds=args.seconds,
            deterministic=args.deterministic,
            verbose=args.verbose
        )

        # Print results
        print_results(results, video_dir, args.verbose)

        # Return success
        return 0

    except Exception as e:
        if args.verbose >= 1:
            print(f"\n❌ Evaluation failed: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
