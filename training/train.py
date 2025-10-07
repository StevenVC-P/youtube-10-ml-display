#!/usr/bin/env python3
"""
Main training script for PPO/DQN on Atari environments.
Supports vectorized environments, TensorBoard logging, and timed checkpoints.
"""

import os
import sys
import time
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from conf.config import load_config
from envs.make_env import make_vec_env, get_env_info
from agents.algo_factory import create_algo, print_system_info, get_algorithm_info
from training.callbacks import MilestoneVideoCallback, TrainingProgressCallback


class TimedCheckpointCallback(BaseCallback):
    """
    Callback that saves checkpoints based on wall-clock time intervals.
    Implements atomic file operations for Windows compatibility.
    """
    
    def __init__(
        self,
        save_path: str,
        checkpoint_every_sec: int = 60,
        keep_last: int = 5,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.checkpoint_every_sec = checkpoint_every_sec
        self.keep_last = keep_last
        self.last_save_time = time.time()
        self.checkpoint_count = 0
        
        # Ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose > 0:
            print(f"⏰ TimedCheckpointCallback initialized:")
            print(f"  • Save path: {self.save_path.absolute()}")
            print(f"  • Checkpoint every: {checkpoint_every_sec} seconds")
            print(f"  • Keep last: {keep_last} checkpoints")
    
    def _on_step(self) -> bool:
        current_time = time.time()
        
        # Check if it's time to save
        if current_time - self.last_save_time >= self.checkpoint_every_sec:
            self._save_checkpoint()
            self.last_save_time = current_time
            
        return True
    
    def _save_checkpoint(self):
        """Save checkpoint with atomic file operations"""
        try:
            # Create temporary file for atomic save
            with tempfile.NamedTemporaryFile(
                suffix='.zip',
                dir=self.save_path,
                delete=False
            ) as tmp_file:
                temp_path = Path(tmp_file.name)
            
            # Save to temporary file
            self.model.save(str(temp_path))
            
            # Atomic move to final location
            latest_path = self.save_path / "latest.zip"
            if latest_path.exists():
                latest_path.unlink()  # Remove existing file on Windows
            shutil.move(str(temp_path), str(latest_path))
            
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.save_path / f"checkpoint_{timestamp}_{self.checkpoint_count:06d}.zip"
            shutil.copy2(str(latest_path), str(backup_path))
            
            self.checkpoint_count += 1
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            if self.verbose > 0:
                mtime = datetime.fromtimestamp(latest_path.stat().st_mtime)
                print(f">> Checkpoint saved: {latest_path.absolute()} (mtime: {mtime})")
                print(f">> Backup created: {backup_path.name}")
                
        except Exception as e:
            print(f"ERROR: Error saving checkpoint: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint backups, keeping only the most recent ones"""
        if self.keep_last <= 0:
            return
            
        # Get all checkpoint files (excluding latest.zip)
        checkpoint_files = list(self.save_path.glob("checkpoint_*.zip"))
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove old files
        for old_file in checkpoint_files[self.keep_last:]:
            try:
                old_file.unlink()
                if self.verbose > 1:
                    print(f"🗑️  Removed old checkpoint: {old_file.name}")
            except Exception as e:
                print(f"⚠️  Warning: Could not remove {old_file.name}: {e}")


class TrainingStatsCallback(BaseCallback):
    """Callback for logging training statistics"""
    
    def __init__(self, log_every_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps
        self.start_time = None
        
    def _on_training_start(self) -> None:
        self.start_time = time.time()
        if self.verbose > 0:
            print(f">> Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_every_steps == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = self.n_calls / elapsed if elapsed > 0 else 0
            
            if self.verbose > 0:
                print(f"📊 Step {self.n_calls:,} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Speed: {steps_per_sec:.1f} steps/s")
        
        return True


def setup_tensorboard_logging(config: Dict[str, Any]) -> str:
    """Setup TensorBoard logging directory"""
    tb_log_path = Path(config['paths']['logs_tb'])
    tb_log_path.mkdir(parents=True, exist_ok=True)
    
    # Create run-specific directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_path = tb_log_path / run_name
    
    print(f">> TensorBoard logs: {run_path.absolute()}")
    return str(run_path)


def train_agent(
    config: Dict[str, Any],
    dryrun_seconds: Optional[int] = None,
    verbose: int = 1
) -> str:
    """
    Train the RL agent with the given configuration.
    
    Args:
        config: Configuration dictionary
        dryrun_seconds: If set, stop training after this many seconds
        verbose: Verbosity level
        
    Returns:
        Path to the saved checkpoint
    """
    # Print system information
    if verbose > 0:
        print_system_info()
    
    # Setup paths
    models_path = Path(config['paths']['models'])
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard logging
    tb_log_path = setup_tensorboard_logging(config)
    
    # Create vectorized environment
    train_config = config['train']
    n_envs = train_config.get('vec_envs', 8)
    
    if verbose > 0:
        print(f">> Creating {n_envs} vectorized environments...")

    vec_env = make_vec_env(
        config=config,
        n_envs=n_envs,
        seed=config.get('seed', 42)
    )

    # Get environment info
    env_info = get_env_info(config)
    if verbose > 0:
        print(f">> Environment Info:")
        print(f"  - Environment: {env_info['env_id']}")
        print(f"  - Observation shape: {env_info['observation_shape']}")
        print(f"  - Action space: {env_info['action_space']}")
        print(f"  - Number of actions: {env_info['n_actions']}")
    
    # Create algorithm
    if verbose > 0:
        print(f"🤖 Creating {train_config['algo'].upper()} algorithm...")
    
    algo = create_algo(config, vec_env, tensorboard_log=tb_log_path)
    
    # Get algorithm info
    algo_info = get_algorithm_info(algo)
    if verbose > 0:
        print(f"📊 Algorithm Configuration:")
        for key, value in algo_info.items():
            print(f"  • {key}: {value}")

    # Determine training duration
    total_timesteps = train_config.get('total_timesteps', 10000000)

    if dryrun_seconds:
        # Estimate timesteps for dryrun
        estimated_steps_per_sec = 1000  # Conservative estimate
        total_timesteps = min(total_timesteps, dryrun_seconds * estimated_steps_per_sec)
        if verbose > 0:
            print(f"🧪 Dry run mode: Training for ~{dryrun_seconds} seconds ({total_timesteps:,} steps)")

    # Setup callbacks
    callbacks = []

    # Timed checkpoint callback
    checkpoint_callback = TimedCheckpointCallback(
        save_path=str(models_path),
        checkpoint_every_sec=train_config.get('checkpoint_every_sec', 60),
        keep_last=train_config.get('keep_last', 5),
        verbose=verbose
    )
    callbacks.append(checkpoint_callback)

    # Milestone video callback
    recording_config = config.get('recording', {})
    if recording_config.get('milestones_pct'):
        milestone_callback = MilestoneVideoCallback(
            config=config,
            milestones_pct=recording_config['milestones_pct'],
            clip_seconds=recording_config.get('milestone_clip_seconds', 90),
            fps=recording_config.get('fps', 30),
            verbose=verbose
        )
        callbacks.append(milestone_callback)

        # Training progress callback (works well with milestone callback)
        progress_callback = TrainingProgressCallback(
            total_timesteps=total_timesteps,
            milestones_pct=recording_config['milestones_pct'],
            log_every_steps=10000,
            verbose=verbose
        )
        callbacks.append(progress_callback)

    # Training stats callback
    stats_callback = TrainingStatsCallback(
        log_every_steps=1000,
        verbose=verbose
    )
    callbacks.append(stats_callback)
    
    callback_list = CallbackList(callbacks)

    if verbose > 0:
        print(f">> Starting training for {total_timesteps:,} timesteps...")
        print(f">> Checkpoints will be saved to: {models_path.absolute()}")
        print("=" * 60)
    
    # Start training
    start_time = time.time()
    
    try:
        algo.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name="training",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Final checkpoint save
        final_path = models_path / "latest.zip"
        algo.save(str(final_path))
        
        elapsed = time.time() - start_time
        if verbose > 0:
            print("=" * 60)
            print(f">> Training completed!")
            print(f">> Total time: {elapsed:.1f} seconds")
            print(f">> Final checkpoint: {final_path.absolute()}")
            print(f">> TensorBoard logs: {tb_log_path}")
        
        return str(final_path)
        
    except KeyboardInterrupt:
        if verbose > 0:
            print("\n⏹️  Training interrupted by user")
        
        # Save final checkpoint
        final_path = models_path / "latest.zip"
        algo.save(str(final_path))
        
        if verbose > 0:
            print(f">> Final checkpoint saved: {final_path.absolute()}")
        
        return str(final_path)
        
    finally:
        vec_env.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train RL agent on Atari games")
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dryrun-seconds",
        type=int,
        default=None,
        help="Run training for specified seconds (for testing)"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=quiet, 1=normal, 2=verbose)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Error loading config: {e}")
        sys.exit(1)

    # Start training
    try:
        checkpoint_path = train_agent(
            config=config,
            dryrun_seconds=args.dryrun_seconds,
            verbose=args.verbose
        )
        print(f"\n>> Training complete! Checkpoint: {checkpoint_path}")

    except Exception as e:
        print(f"ERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
