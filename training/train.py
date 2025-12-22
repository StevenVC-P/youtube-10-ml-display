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
import logging
import subprocess

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

# Set up logging
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Fix Windows console encoding to support Unicode output
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from conf.config import load_config

# Check for epic-specific configuration
def get_config_path():
    """Get the appropriate config path based on environment variables."""
    epic_dir = os.environ.get('EPIC_DIR')
    if epic_dir:
        epic_config = Path(epic_dir) / "config" / "config.yaml"
        if epic_config.exists():
            return str(epic_config)
    return None  # Use default
from envs.make_env import make_vec_env, get_env_info
from agents.algo_factory import create_algo, print_system_info, get_algorithm_info
from training.callbacks import MilestoneVideoCallback, TrainingProgressCallback
from training.ml_analytics_callback import MLAnalyticsVideoCallback
from training.hour_video_callback import HourVideoCallback
from training.milestone_checkpoint_callback import MilestoneCheckpointCallback


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
            print(f"[Timer] TimedCheckpointCallback initialized:")
            print(f"  - Save path: {self.save_path.absolute()}")
            print(f"  - Checkpoint every: {checkpoint_every_sec} seconds")
            print(f"  - Keep last: {keep_last} checkpoints")
    
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
                    print(f"[Cleanup]  Removed old checkpoint: {old_file.name}")
            except Exception as e:
                print(f"Warning  Warning: Could not remove {old_file.name}: {e}")


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
                print(f"[Stats] Step {self.n_calls:,} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Speed: {steps_per_sec:.1f} steps/s")
        
        return True


class InlineProgressPrinter(BaseCallback):
    """SB3 callback that prints compact progress lines to stdout for desktop collector."""

    def __init__(
        self,
        total_target_steps: int,
        start_timestep: int = 0,
        run_id: str = None,
        log_every_steps: int = 5000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.total_target_steps = total_target_steps or 0
        self.start_timestep = start_timestep or 0
        self.run_id = run_id
        self.log_every_steps = max(1, log_every_steps or 1)
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_every_steps != 0:
            return True

        current_steps = self.num_timesteps + self.start_timestep
        total_steps = self.total_target_steps or max(current_steps, 1)
        progress_pct = (current_steps / total_steps) * 100 if total_steps else 0.0

        elapsed = (time.time() - self._start_time) if self._start_time else 0.0
        speed = (self.n_calls / elapsed) if elapsed > 0 else 0.0
        remaining_steps = max(total_steps - current_steps, 0)
        eta_hours = (remaining_steps / speed) / 3600 if speed > 0 else float("inf")
        eta_str = f"{eta_hours:.2f}" if eta_hours != float("inf") else "inf"
        run_tag = f" run_id={self.run_id}" if self.run_id else ""

        print(
            f"[Stats] Progress: {current_steps:,}/{total_steps:,} ({progress_pct:.1f}%) | "
            f"Speed: {speed:.1f} steps/s | ETA: {eta_str}h{run_tag}",
            flush=True
        )
        return True


# Simple guard to ensure the class is defined
assert InlineProgressPrinter is not None

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
    verbose: int = 1,
    resume_from_checkpoint: Optional[str] = None
) -> str:
    """
    Train the RL agent with the given configuration.

    Args:
        config: Configuration dictionary
        dryrun_seconds: If set, stop training after this many seconds
        verbose: Verbosity level
        resume_from_checkpoint: Path to checkpoint to resume training from

    Returns:
        Path to the saved checkpoint
    """
    # Print system information
    if verbose > 0:
        print_system_info()

    # Setup paths
    models_path = Path(config['paths']['models'])
    models_path.mkdir(parents=True, exist_ok=True)

    # Extract run_id from models path for training video recording
    # Path format: "models/checkpoints/{run_id}" or "models/checkpoints/{run_id}/milestones"
    run_id = None
    try:
        # Get the last part of the path that looks like a run_id (e.g., "run-abc123")
        path_parts = models_path.parts
        logger.info(f"[VIDEO DEBUG] models_path={models_path}")
        logger.info(f"[VIDEO DEBUG] path_parts={path_parts}")
        for part in reversed(path_parts):
            if part.startswith('run-') or part.startswith('test-'):
                run_id = part
                logger.info(f"[VIDEO DEBUG] Found run_id={run_id}")
                break
    except Exception as e:
        logger.warning(f"[VIDEO DEBUG] Failed to extract run_id: {e}")
        pass  # If extraction fails, run_id will be None (no video recording)

    logger.info(f"[VIDEO DEBUG] Final run_id={run_id}")

    # Setup TensorBoard logging
    tb_log_path = setup_tensorboard_logging(config)

    # Create vectorized environment
    train_config = config['train']
    n_envs = train_config.get('vec_envs', 8)
    force_single_env = bool(train_config.get('force_single_env'))
    if force_single_env:
        n_envs = 1
        print("[SENTINEL] force_single_env enabled: vec_envs=1", flush=True)

    if verbose > 0:
        print(f">> Creating {n_envs} vectorized environments...")

        training_video_enabled_cfg = bool(config.get("training_video", {}).get("enabled", False))
        training_video_base_dir = config.get("paths", {}).get("videos_training", "video/training")
        training_video_out_dir = os.path.join(training_video_base_dir, run_id) if run_id else None

        training_video_reasons: list[str] = []
        if not training_video_enabled_cfg:
            training_video_reasons.append("config.training_video.enabled=false")
        if run_id is None:
            training_video_reasons.append("run_id=None")

        training_video_enabled = training_video_enabled_cfg and run_id is not None
        training_video_reason = "enabled" if training_video_enabled else "; ".join(training_video_reasons) or "disabled"

        print(
            f"TRAINING_VIDEO enabled={training_video_enabled} reason={training_video_reason} out_dir={training_video_out_dir} n_envs={n_envs}",
            flush=True,
        )
    print(f"[SENTINEL] before_env_creation n_envs={n_envs}", flush=True)

    vec_env = make_vec_env(
        config=config,
        n_envs=n_envs,
        seed=config.get('seed', 42),
        run_id=run_id  # Pass run_id for training video recording
    )
    print("[SENTINEL] after_env_creation", flush=True)

    # Get environment info
    env_info = get_env_info(config)
    if verbose > 0:
        print(f">> Environment Info:")
        print(f"  - Environment: {env_info['env_id']}")
        print(f"  - Observation shape: {env_info['observation_shape']}")
        print(f"  - Action space: {env_info['action_space']}")
        print(f"  - Number of actions: {env_info['n_actions']}")

    # Create or load algorithm
    print(f"[SENTINEL] before_model_creation resume_from_checkpoint={bool(resume_from_checkpoint and Path(resume_from_checkpoint).exists())}", flush=True)
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        # Resume from checkpoint
        if verbose > 0:
            print(f"[Agent] Loading checkpoint from {resume_from_checkpoint}...")

        # Import the appropriate algorithm class
        algo_name = train_config['algo'].upper()
        if algo_name == 'PPO':
            from stable_baselines3 import PPO
            algo = PPO.load(resume_from_checkpoint, env=vec_env, tensorboard_log=tb_log_path)
        elif algo_name == 'DQN':
            from stable_baselines3 import DQN
            algo = DQN.load(resume_from_checkpoint, env=vec_env, tensorboard_log=tb_log_path)
        else:
            raise ValueError(f"Unsupported algorithm for resume: {algo_name}")

        if verbose > 0:
            print(f"[Agent] Successfully loaded checkpoint, continuing training...")
    else:
        # Create new algorithm
        if verbose > 0:
            print(f"[Agent] Creating {train_config['algo'].upper()} algorithm...")

        algo = create_algo(config, vec_env, tensorboard_log=tb_log_path)
    print("[SENTINEL] after_model_creation", flush=True)

    # Get algorithm info
    algo_info = get_algorithm_info(algo)
    if verbose > 0:
        print(f"[Stats] Algorithm Configuration:")
        for key, value in algo_info.items():
            print(f"  - {key}: {value}")

    # Determine training duration
    total_timesteps = train_config.get('total_timesteps', 10000000)

    if dryrun_seconds:
        # Estimate timesteps for dryrun
        estimated_steps_per_sec = 1000  # Conservative estimate
        total_timesteps = min(total_timesteps, dryrun_seconds * estimated_steps_per_sec)
        if verbose > 0:
            print(f"[Test] Dry run mode: Training for ~{dryrun_seconds} seconds ({total_timesteps:,} steps)")

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

    # Epic 10-Hour Neural Network Learning Journey
    # Check if we should skip video recording (for desktop app)
    epic_clip_seconds = config.get('recording', {}).get('milestone_clip_seconds', 3600)

    if epic_clip_seconds == 0:
        # Desktop app mode: Skip video recording, save checkpoints instead
        milestone_checkpoint_callback = MilestoneCheckpointCallback(
            save_path=str(Path(models_path) / "milestones"),
            total_timesteps=total_timesteps,
            milestones_pct=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            verbose=verbose
        )
        callbacks.append(milestone_checkpoint_callback)

        if verbose >= 1:
            print("[Training] Using checkpoint-only mode (no video recording during training)")
    else:
        # Production mode: Record videos during training
        # Get custom name from config if available
        custom_name = config.get('train', {}).get('custom_name')

        epic_journey_callback = MLAnalyticsVideoCallback(
            config=config,
            milestones_pct=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # 10 milestones = 10 hours
            clip_seconds=epic_clip_seconds,  # Use config setting (default 3600 for epic, 10 for desktop)
            fps=30,
            verbose=verbose,
            custom_name=custom_name
        )
        callbacks.append(epic_journey_callback)

        if verbose >= 1:
            print(f"[Training] Using video recording mode ({epic_clip_seconds}s clips)")

    # Training progress callback
    progress_callback = TrainingProgressCallback(
        total_timesteps=total_timesteps,
        milestones_pct=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Progress markers
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
    
    # Inline stdout progress printer to keep desktop collector fed with metrics
    absolute_target = train_config.get('target_total_timesteps', total_timesteps)
    start_timestep = train_config.get('start_timestep', 0)
    progress_printer = InlineProgressPrinter(
        total_target_steps=absolute_target,
        start_timestep=start_timestep,
        run_id=run_id,
        log_every_steps=train_config.get('log_every_steps', 5000),
        verbose=verbose
    )
    callbacks.append(progress_printer)
    
    callback_list = CallbackList(callbacks)

    if verbose > 0:
        print(f">> Starting training for {total_timesteps:,} timesteps...")
        print(f">> Checkpoints will be saved to: {models_path.absolute()}")
        print("=" * 60)
    
    # Start training
    start_time = time.time()
    print(f"[SENTINEL] before_model_learn total_timesteps={total_timesteps} reset_num_timesteps={not bool(resume_from_checkpoint and Path(resume_from_checkpoint).exists())}", flush=True)
    
    try:
        # When resuming, don't reset timesteps to preserve progress
        reset_timesteps = not bool(resume_from_checkpoint and Path(resume_from_checkpoint).exists())

        print("[SENTINEL] entering_learn", flush=True)
        algo.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name="training",
            reset_num_timesteps=reset_timesteps,
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
            print("\n[Stop]  Training interrupted by user")
        
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

    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (no live video recording)"
    )

    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Skip training, run post-training render only"
    )

    args = parser.parse_args()
    
    # Load configuration (epic-specific if available)
    try:
        epic_config_path = get_config_path()
        # Use epic config if available and user didn't explicitly specify a config
        if epic_config_path and args.config == "conf/config.yaml":
            print(f"[Epic] Using epic-specific config: {epic_config_path}")
            config = load_config(epic_config_path)
            print(f"[Epic] Loaded environment ID: {config['game']['env_id']}")
        else:
            config = load_config(args.config)
            print(f"[Default] Using config: {args.config}, environment ID: {config['game']['env_id']}")
        print(f"[SENTINEL] after_config_load env_id={config['game']['env_id']} vec_envs={config.get('train', {}).get('vec_envs')} seed={config.get('seed')}", flush=True)
    except Exception as e:
        print(f"ERROR: Error loading config: {e}")
        sys.exit(1)

    # Handle Fast Mode and Render Only overrides
    if args.fast:
        print("🚀 Fast Mode: Disabling live video recording to maximize training speed")
        if 'recording' not in config: config['recording'] = {}
        config['recording']['milestone_clip_seconds'] = 0
        if 'training_video' not in config: config['training_video'] = {}
        config['training_video']['enabled'] = False

    # Start training
    checkpoint_path = None
    if not args.render_only:
        try:
            checkpoint_path = train_agent(
                config=config,
                dryrun_seconds=args.dryrun_seconds,
                verbose=args.verbose,
                resume_from_checkpoint=args.resume_from
            )
            print(f"\n>> Training complete! Checkpoint: {checkpoint_path}")

        except Exception as e:
            # Enhanced error handling with CUDA diagnostics
            import traceback
            traceback_str = traceback.format_exc()

            # Check if this is a CUDA-related error
            error_message = str(e).lower()
            is_cuda_error = any(keyword in error_message for keyword in [
                'cuda', 'gpu', 'memory allocation', 'device', 'kernel'
            ])

            if is_cuda_error:
                try:
                    # Import CUDA diagnostics (if available)
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from tools.retro_ml_desktop.cuda_diagnostics import create_user_friendly_error_message

                    # Create user-friendly error message
                    friendly_message = create_user_friendly_error_message(e, traceback_str)
                    print(friendly_message)

                except ImportError:
                    # Fallback to basic CUDA error handling
                    print(f"ERROR: CUDA/GPU Training failed: {e}")
                    print("\n[SOLUTIONS] QUICK SOLUTIONS TO TRY:")
                    print("  1. Reduce batch size (try 128 or 64 instead of 256)")
                    print("  2. Reduce number of environments (try 4 or 2 instead of 8)")
                    print("  3. Close other GPU applications")
                    print("  4. Restart the application")
                    print("  5. Use CPU training instead")
                    print("\n[TIP] Check GPU memory usage and update NVIDIA drivers.")
            else:
                print(f"ERROR: Training failed: {e}")

            print(f"\n[ERROR DETAILS] Full error details:")
            traceback.print_exc()
            sys.exit(1)
    
    # Auto-Render Logic (Fast Mode or Render Only)
    if (args.fast or args.render_only):
        print("\n🚀 Fast Mode/Render Only: Triggering automatic post-training render...")
        try:
            # Determine correct paths
            run_id = config.get('train', {}).get('run_id')
            if not run_id:
                # Try to infer from config paths
                models_path = config.get('paths', {}).get('models', '')
                run_id = Path(models_path).name if models_path else "unknown_run"
            
            # Use specific target hours from render config, or default
            target_hours = config.get('render', {}).get('target_hours', 10)
            total_seconds = int(target_hours * 3600)
            
            # Resolve script path
            script_dir = Path(__file__).parent
            render_script = script_dir / "post_training_video_generator.py"
            
            if not render_script.exists():
                print(f"⚠️ Could not find render script at: {render_script}")
            else:
                # Build command
                model_dir = config.get('paths', {}).get('models')
                if not model_dir:
                    # Fallback
                    model_dir = f"models/checkpoints/{run_id}"

                # For output dir, we want the parent of 'milestones' usually, or just use config output path
                # Ideally read from config['paths']['videos_milestones'] and go up one level
                milestones_path = config.get('paths', {}).get('videos_milestones')
                if milestones_path:
                    output_dir = str(Path(milestones_path).parent)
                else:
                    output_dir = f"outputs/{run_id}"

                cmd = [
                    sys.executable, str(render_script),
                    "--model-dir", str(model_dir),
                    "--output-dir", str(output_dir),
                    "--config", args.config, # Use the config file we loaded
                    "--total-seconds", str(total_seconds)
                ]
                
                print(f"   Executing: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                print("✅ Post-training render complete!")
                
        except Exception as e:
            print(f"❌ Failed to run post-training render: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
class InlineProgressPrinter(BaseCallback):
    """
    Lightweight progress printer to ensure stdout always contains parsable stats
    for both fresh and resumed runs. Prints at a fixed step interval.
    """

    def __init__(self, total_target_steps: int, start_timestep: int, run_id: Optional[str], log_every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.total_target_steps = total_target_steps
        self.start_timestep = start_timestep
        self.log_every_steps = max(1, log_every_steps)
        self.run_id = run_id
        self._last_step = 0
        self._last_time = time.time()

    def _on_step(self) -> bool:
        # Only log every N new steps
        if (self.num_timesteps - self._last_step) < self.log_every_steps:
            return True

        now = time.time()
        elapsed = max(1e-6, now - self._last_time)
        step_delta = self.num_timesteps - self._last_step
        steps_per_sec = step_delta / elapsed if step_delta > 0 else 0.0

        current_abs = self.start_timestep + self.num_timesteps
        progress_pct = 100.0 * current_abs / self.total_target_steps if self.total_target_steps else 0.0
        eta_hours = ((self.total_target_steps - current_abs) / steps_per_sec / 3600.0) if steps_per_sec > 0 else 0.0

        # Emit SB3-style stats line (matched by LogParser sb3_progress regex)
        print(
            f"[Stats] Progress: {current_abs:,}/{self.total_target_steps:,} ({progress_pct:.1f}%) | Speed: {steps_per_sec:.1f} steps/s | ETA: {eta_hours:.2f}h"
            + (f" | run_id:{self.run_id}" if self.run_id else ""),
            flush=True
        )

        self._last_step = self.num_timesteps
        self._last_time = now
        return True
