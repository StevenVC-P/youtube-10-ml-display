"""
Training callbacks for milestone video recording and evaluation.

This module provides callbacks for Stable-Baselines3 training that:
- Record milestone videos at specific training progress percentages
- Capture agent performance at key training intervals
- Save videos with meaningful filenames including step and percentage
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from gymnasium.wrappers import RecordVideo

from envs.make_env import make_eval_env


class MilestoneVideoCallback(BaseCallback):
    """
    Callback that records milestone videos at specific training progress percentages.
    
    Records short video clips when the training crosses predefined milestone thresholds
    (e.g., 1%, 5%, 10%, etc. of total training steps). Videos are saved with descriptive
    filenames including the global step and percentage milestone.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        milestones_pct: List[float],
        clip_seconds: int = 90,
        fps: int = 30,
        verbose: int = 1
    ):
        """
        Initialize the milestone video callback.
        
        Args:
            config: Full configuration dictionary
            milestones_pct: List of percentage milestones (e.g., [1, 5, 10, 20])
            clip_seconds: Duration of each milestone clip in seconds
            fps: Frames per second for video recording
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose)
        
        self.config = config
        self.milestones_pct = sorted(milestones_pct)  # Ensure sorted order
        self.clip_seconds = clip_seconds
        self.fps = fps
        
        # Calculate total timesteps and milestone thresholds
        self.total_timesteps = config['train']['total_timesteps']
        self.milestone_steps = [
            int(pct / 100.0 * self.total_timesteps) 
            for pct in self.milestones_pct
        ]
        
        # Track which milestones have been recorded
        self.recorded_milestones = set()
        
        # Setup output directory
        self.output_dir = Path(config['paths']['videos_milestones'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose >= 1:
            print(f"🎬 MilestoneVideoCallback initialized:")
            print(f"  • Output directory: {self.output_dir.absolute()}")
            print(f"  • Total timesteps: {self.total_timesteps:,}")
            print(f"  • Milestone percentages: {self.milestones_pct}")
            print(f"  • Milestone steps: {[f'{s:,}' for s in self.milestone_steps]}")
            print(f"  • Clip duration: {self.clip_seconds}s @ {self.fps} FPS")
    
    def _on_step(self) -> bool:
        """
        Called after each training step. Check if we've crossed a milestone threshold.
        
        Returns:
            bool: True to continue training, False to stop
        """
        current_step = self.num_timesteps
        
        # Check if we've crossed any new milestone thresholds
        for i, (pct, milestone_step) in enumerate(zip(self.milestones_pct, self.milestone_steps)):
            if current_step >= milestone_step and pct not in self.recorded_milestones:
                self._record_milestone_video(current_step, pct)
                self.recorded_milestones.add(pct)
                
                if self.verbose >= 1:
                    print(f"📹 Milestone {pct}% recorded at step {current_step:,}")
        
        return True
    
    def _record_milestone_video(self, current_step: int, milestone_pct: float) -> None:
        """
        Record a milestone video clip.
        
        Args:
            current_step: Current training step
            milestone_pct: Milestone percentage (e.g., 1.0 for 1%)
        """
        try:
            # Create filename with step and percentage
            filename = f"step_{current_step:08d}_pct_{milestone_pct:g}"
            video_path = self.output_dir / filename
            
            if self.verbose >= 2:
                print(f"🎥 Recording milestone video: {filename}")
            
            # Create evaluation environment with video recording
            eval_env = self._create_recording_env(video_path)
            
            # Calculate number of frames needed
            target_frames = self.clip_seconds * self.fps
            frames_recorded = 0
            episodes_completed = 0
            
            start_time = time.time()
            
            # Record until we have enough frames or reasonable time limit
            while frames_recorded < target_frames and episodes_completed < 10:
                obs, _ = eval_env.reset()
                episode_frames = 0
                done = False
                
                while not done and frames_recorded < target_frames:
                    # Use current model to predict action
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    
                    episode_frames += 1
                    frames_recorded += 1
                
                episodes_completed += 1
            
            eval_env.close()
            
            # Log recording results
            duration = time.time() - start_time
            actual_seconds = frames_recorded / self.fps
            
            if self.verbose >= 1:
                print(f"✅ Milestone {milestone_pct}% video saved:")
                print(f"  • File: {video_path}.mp4")
                print(f"  • Duration: {actual_seconds:.1f}s ({frames_recorded} frames)")
                print(f"  • Episodes: {episodes_completed}")
                print(f"  • Recording time: {duration:.1f}s")
        
        except Exception as e:
            if self.verbose >= 1:
                print(f"❌ Failed to record milestone {milestone_pct}% video: {e}")
    
    def _create_recording_env(self, video_path: Path) -> gym.Env:
        """
        Create an evaluation environment with video recording wrapper.

        Args:
            video_path: Path for video output (without extension)

        Returns:
            gym.Env: Environment with RecordVideo wrapper
        """
        # Create base evaluation environment with render mode for video recording
        import gymnasium as gym
        from envs.atari_wrappers import apply_atari_wrappers

        # Create base environment with rgb_array render mode
        env_id = self.config['game']['env_id']
        base_env = gym.make(env_id, render_mode="rgb_array")

        # Apply Atari wrappers
        wrapped_env = apply_atari_wrappers(base_env, self.config)

        # Wrap with RecordVideo
        # Note: RecordVideo expects the directory and will create filename
        video_dir = video_path.parent
        video_name = video_path.name

        recording_env = RecordVideo(
            wrapped_env,
            video_folder=str(video_dir),
            name_prefix=video_name,
            episode_trigger=lambda x: True,  # Record all episodes
            video_length=0,  # Record full episodes
        )

        return recording_env


class TrainingProgressCallback(BaseCallback):
    """
    Callback that logs training progress and milestone information.
    
    Provides detailed logging of training progress, milestone crossings,
    and estimated time to completion.
    """
    
    def __init__(
        self,
        total_timesteps: int,
        milestones_pct: List[float],
        log_every_steps: int = 10000,
        verbose: int = 1
    ):
        """
        Initialize the training progress callback.
        
        Args:
            total_timesteps: Total training timesteps
            milestones_pct: List of milestone percentages
            log_every_steps: Log progress every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.total_timesteps = total_timesteps
        self.milestones_pct = sorted(milestones_pct)
        self.log_every_steps = log_every_steps
        
        self.start_time = None
        self.last_log_step = 0
        self.crossed_milestones = set()
    
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        if self.verbose >= 1:
            print(f"🚀 Training started - Target: {self.total_timesteps:,} timesteps")
    
    def _on_step(self) -> bool:
        """Called after each training step."""
        current_step = self.num_timesteps
        
        # Log progress at regular intervals
        if current_step - self.last_log_step >= self.log_every_steps:
            self._log_progress(current_step)
            self.last_log_step = current_step
        
        # Check for milestone crossings
        current_pct = (current_step / self.total_timesteps) * 100
        for milestone_pct in self.milestones_pct:
            if current_pct >= milestone_pct and milestone_pct not in self.crossed_milestones:
                self._log_milestone_crossing(current_step, milestone_pct)
                self.crossed_milestones.add(milestone_pct)
        
        return True
    
    def _log_progress(self, current_step: int) -> None:
        """Log current training progress."""
        if self.start_time is None:
            return
        
        elapsed_time = time.time() - self.start_time
        progress_pct = (current_step / self.total_timesteps) * 100
        
        if current_step > 0:
            steps_per_sec = current_step / elapsed_time
            eta_seconds = (self.total_timesteps - current_step) / steps_per_sec
            eta_hours = eta_seconds / 3600
            
            if self.verbose >= 1:
                print(f"📊 Progress: {current_step:,}/{self.total_timesteps:,} "
                      f"({progress_pct:.1f}%) | "
                      f"Speed: {steps_per_sec:.0f} steps/s | "
                      f"ETA: {eta_hours:.1f}h")
    
    def _log_milestone_crossing(self, current_step: int, milestone_pct: float) -> None:
        """Log milestone crossing."""
        if self.verbose >= 1:
            print(f"🎯 Milestone reached: {milestone_pct}% at step {current_step:,}")
