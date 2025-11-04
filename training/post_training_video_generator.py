#!/usr/bin/env python3
"""
Post-Training Video Generator

Generates milestone videos AFTER training completes using saved model checkpoints.
This prevents video recording from blocking the training process.

Usage:
    python post_training_video_generator.py --model-dir models/checkpoints --config conf/config.yaml
"""

import os
import sys
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import time

import numpy as np
import cv2
import torch
import gymnasium as gym
from stable_baselines3 import PPO, DQN

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.make_env import make_eval_env
from tools.stream.grid_composer import SingleScreenComposer


class PostTrainingVideoGenerator:
    """
    Generates milestone videos after training completes using saved model checkpoints.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_dir: Path,
        output_dir: Path,
        milestones_pct: List[float] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        clip_seconds: int = 90,
        fps: int = 30,
        verbose: int = 1
    ):
        self.config = config
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        self.milestones_pct = milestones_pct
        self.clip_seconds = clip_seconds
        self.fps = fps
        self.verbose = verbose
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose >= 1:
            print(f"[PostVideo] Post-Training Video Generator initialized:")
            print(f"  - Model directory: {self.model_dir}")
            print(f"  - Output directory: {self.output_dir}")
            print(f"  - Milestones: {self.milestones_pct}")
            print(f"  - Clip duration: {clip_seconds}s @ {fps} FPS")
    
    def find_checkpoint_files(self) -> Dict[float, Path]:
        """Find all available checkpoint files and map them to milestone percentages."""
        checkpoint_files = {}
        
        if not self.model_dir.exists():
            print(f"[PostVideo] Warning: Model directory {self.model_dir} does not exist")
            return checkpoint_files
        
        # Look for checkpoint files (various naming patterns)
        patterns = [
            "checkpoint_*.zip",
            "model_*.zip", 
            "step_*.zip",
            "pct_*.zip"
        ]
        
        for pattern in patterns:
            for file_path in self.model_dir.glob(pattern):
                # Try to extract milestone percentage from filename
                filename = file_path.stem
                
                # Extract percentage from various naming patterns
                pct = self._extract_percentage_from_filename(filename)
                if pct and pct in self.milestones_pct:
                    checkpoint_files[pct] = file_path
                    if self.verbose >= 2:
                        print(f"[PostVideo] Found checkpoint for {pct}%: {file_path.name}")
        
        # If no milestone-specific checkpoints, use the final model
        if not checkpoint_files:
            final_model_patterns = ["final_model.zip", "best_model.zip", "model.zip"]
            for pattern in final_model_patterns:
                final_model = self.model_dir / pattern
                if final_model.exists():
                    # Use final model for all milestones
                    for pct in self.milestones_pct:
                        checkpoint_files[pct] = final_model
                    if self.verbose >= 1:
                        print(f"[PostVideo] Using final model for all milestones: {final_model.name}")
                    break
        
        return checkpoint_files
    
    def _extract_percentage_from_filename(self, filename: str) -> Optional[float]:
        """Extract milestone percentage from checkpoint filename."""
        import re
        
        # Try various patterns
        patterns = [
            r'pct_(\d+)',           # pct_50
            r'milestone_(\d+)',     # milestone_50  
            r'step_\d+_pct_(\d+)',  # step_12345_pct_50
            r'(\d+)pct',            # 50pct
            r'(\d+)_percent'        # 50_percent
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def generate_all_videos(self) -> List[Path]:
        """Generate videos for all available checkpoints."""
        checkpoint_files = self.find_checkpoint_files()
        
        if not checkpoint_files:
            print(f"[PostVideo] No checkpoint files found in {self.model_dir}")
            return []
        
        generated_videos = []
        
        print(f"[PostVideo] Generating videos for {len(checkpoint_files)} checkpoints...")
        
        for milestone_pct in sorted(checkpoint_files.keys()):
            checkpoint_path = checkpoint_files[milestone_pct]
            
            try:
                video_path = self._generate_milestone_video(milestone_pct, checkpoint_path)
                if video_path:
                    generated_videos.append(video_path)
                    if self.verbose >= 1:
                        print(f"[PostVideo] [OK] Generated video for {milestone_pct}%: {video_path.name}")
                else:
                    print(f"[PostVideo] [ERROR] Failed to generate video for {milestone_pct}%")
                    
            except Exception as e:
                print(f"[PostVideo] [ERROR] Error generating video for {milestone_pct}%: {e}")
        
        print(f"[PostVideo] Video generation complete! Generated {len(generated_videos)} videos.")
        return generated_videos
    
    def _generate_milestone_video(self, milestone_pct: float, checkpoint_path: Path) -> Optional[Path]:
        """Generate a single milestone video from a checkpoint."""
        try:
            # Create video filename
            video_filename = f"step_post_training_pct_{milestone_pct:.0f}_analytics.mp4"
            video_path = self.output_dir / video_filename
            
            if self.verbose >= 1:
                print(f"[PostVideo] Generating video: {video_filename}")
            
            # Load model from checkpoint
            model = self._load_model(checkpoint_path)
            if model is None:
                return None
            
            # Create evaluation environment with proper wrappers and rgb_array render mode
            env = make_eval_env(
                config=self.config,
                seed=42,  # Fixed seed for reproducible videos
                record_video=True  # This enables rgb_array render mode
            )
            
            # Record gameplay frames
            frames = self._record_gameplay_frames(model, env)
            
            if not frames:
                print(f"[PostVideo] No frames recorded for {milestone_pct}%")
                return None
            
            # Create video from frames
            success = self._create_video_from_frames(frames, video_path)
            
            # Cleanup
            env.close()
            
            return video_path if success else None
            
        except Exception as e:
            print(f"[PostVideo] Error in _generate_milestone_video: {e}")
            return None
    
    def _load_model(self, model_path: Path):
        """Load model from checkpoint file."""
        try:
            algorithm = self.config['train']['algo'].upper()
            
            if algorithm == 'PPO':
                return PPO.load(str(model_path))
            elif algorithm == 'DQN':
                return DQN.load(str(model_path))
            else:
                print(f"[PostVideo] Unsupported algorithm: {algorithm}")
                return None
                
        except Exception as e:
            print(f"[PostVideo] Failed to load model from {model_path}: {e}")
            return None
    
    def _record_gameplay_frames(self, model, env) -> List[np.ndarray]:
        """Record gameplay frames using the trained model."""
        frames = []
        target_frames = self.clip_seconds * self.fps
        
        obs, _ = env.reset()
        
        for frame_idx in range(target_frames):
            # Get frame from environment
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset if episode ends
            if terminated or truncated:
                obs, _ = env.reset()
        
        return frames
    
    def _create_video_from_frames(self, frames: List[np.ndarray], output_path: Path) -> bool:
        """Create MP4 video from frames using OpenCV."""
        try:
            if not frames:
                return False
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))
            
            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            # Release video writer
            out.release()
            
            return output_path.exists()
            
        except Exception as e:
            print(f"[PostVideo] Error creating video: {e}")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate milestone videos after training')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='video/post_training',
                       help='Output directory for videos')
    parser.add_argument('--clip-seconds', type=int, default=10,
                       help='Length of each video clip in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for videos')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (0-2)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create video generator
    generator = PostTrainingVideoGenerator(
        config=config,
        model_dir=Path(args.model_dir),
        output_dir=Path(args.output_dir),
        clip_seconds=args.clip_seconds,
        fps=args.fps,
        verbose=args.verbose
    )
    
    # Generate videos
    generated_videos = generator.generate_all_videos()
    
    print(f"\n[VIDEO] Video generation complete!")
    print(f"Generated {len(generated_videos)} videos in {args.output_dir}")
    
    for video_path in generated_videos:
        print(f"  [VIDEO] {video_path.name}")


if __name__ == "__main__":
    main()
