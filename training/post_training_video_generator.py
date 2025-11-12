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

    def generate_continuous_video(self, total_seconds: int) -> Optional[Path]:
        """
        Generate a single continuous video showing progression through all checkpoints.

        Args:
            total_seconds: Total length of the video in seconds

        Returns:
            Path to the generated video, or None if generation failed
        """
        checkpoint_files = self.find_checkpoint_files()

        if not checkpoint_files:
            print(f"[PostVideo] No checkpoint files found in {self.model_dir}")
            return None

        sorted_milestones = sorted(checkpoint_files.keys())
        num_checkpoints = len(sorted_milestones)

        if num_checkpoints == 0:
            print(f"[PostVideo] No checkpoints available")
            return None

        # Calculate seconds per checkpoint to match total_seconds
        seconds_per_checkpoint = total_seconds / num_checkpoints

        print(f"[PostVideo] Generating continuous video:")
        print(f"  - Total duration: {total_seconds}s ({total_seconds/60:.1f} minutes)")
        print(f"  - Checkpoints: {num_checkpoints}")
        print(f"  - Seconds per checkpoint: {seconds_per_checkpoint:.1f}s")

        # Generate video for each checkpoint
        temp_videos = []
        for milestone_pct in sorted_milestones:
            checkpoint_path = checkpoint_files[milestone_pct]

            try:
                # Temporarily set clip_seconds for this checkpoint
                original_clip_seconds = self.clip_seconds
                self.clip_seconds = int(seconds_per_checkpoint)

                video_path = self._generate_milestone_video(milestone_pct, checkpoint_path)

                # Restore original clip_seconds
                self.clip_seconds = original_clip_seconds

                if video_path:
                    temp_videos.append(video_path)
                    if self.verbose >= 1:
                        print(f"[PostVideo] [OK] Generated segment for {milestone_pct}%: {video_path.name}")
                else:
                    print(f"[PostVideo] [ERROR] Failed to generate segment for {milestone_pct}%")

            except Exception as e:
                print(f"[PostVideo] [ERROR] Error generating segment for {milestone_pct}%: {e}")

        if not temp_videos:
            print(f"[PostVideo] No video segments were generated")
            return None

        # Concatenate all videos into one
        # Use a more descriptive filename with game name and duration
        game_name = self.config.get('game', {}).get('env_id', 'game').split('/')[-1].split('-')[0].lower()
        minutes = total_seconds / 60
        if minutes >= 60:
            hours = minutes / 60
            duration_str = f"{hours:.1f}h"
        else:
            duration_str = f"{int(minutes)}min"

        output_path = self.output_dir / f"{game_name}_{duration_str}_training.mp4"

        print(f"[PostVideo] Concatenating {len(temp_videos)} segments into continuous video...")

        success = self._concatenate_videos(temp_videos, output_path)

        if success:
            print(f"[PostVideo] ✅ Continuous video generated: {output_path.name}")

            # IMPORTANT: Clean up intermediate milestone videos
            # The user only wants the final continuous video, not the segments
            if self.verbose >= 1:
                print(f"[PostVideo] Cleaning up {len(temp_videos)} intermediate video segments...")

            for temp_video in temp_videos:
                try:
                    if temp_video.exists():
                        temp_video.unlink()
                        if self.verbose >= 2:
                            print(f"[PostVideo]   Deleted: {temp_video.name}")
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[PostVideo]   Warning: Could not delete {temp_video.name}: {e}")

            if self.verbose >= 1:
                print(f"[PostVideo] ✅ Cleanup complete. Only final video remains: {output_path.name}")

            return output_path
        else:
            print(f"[PostVideo] ❌ Failed to concatenate videos")
            # Don't clean up temp videos if concatenation failed (for debugging)
            return None
    
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

            # Record gameplay frames DIRECTLY to video (streaming mode)
            success = self._record_gameplay_to_video(model, env, video_path)

            # Cleanup
            env.close()

            return video_path if success else None

        except Exception as e:
            print(f"[PostVideo] Error in _generate_milestone_video: {e}")
            import traceback
            traceback.print_exc()
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
    
    def _record_gameplay_to_video(self, model, env, output_path: Path) -> bool:
        """
        Record gameplay frames directly to video file (streaming mode).
        This avoids storing all frames in memory, which is critical for long videos.
        """
        try:
            target_frames = self.clip_seconds * self.fps

            if self.verbose >= 1:
                print(f"[PostVideo] Recording {target_frames} frames ({self.clip_seconds}s) directly to video...")

            # Initialize video writer (will be created on first frame)
            video_writer = None

            obs, _ = env.reset()
            total_reward = 0.0
            frames_since_reset = 0
            episode_count = 0
            frames_written = 0

            # Add small exploration to prevent stuck patterns
            exploration_rate = 0.05  # 5% random actions

            for frame_idx in range(target_frames):
                # Get frame from environment
                game_frame = env.render()

                if game_frame is not None:
                    # Extract ML analytics for this frame
                    analytics = self._extract_ml_analytics(model, obs, frame_idx, total_reward)
                    analytics['episode_count'] = episode_count
                    analytics['frames_since_reset'] = frames_since_reset

                    # Create enhanced frame with neural network visualization
                    enhanced_frame = self._create_enhanced_frame(game_frame, analytics)

                    # Initialize video writer on first frame
                    if video_writer is None:
                        height, width = enhanced_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (width, height))

                        if not video_writer.isOpened():
                            print(f"[PostVideo] Failed to open video writer for {output_path}")
                            return False

                    # Write frame directly to video file
                    video_writer.write(enhanced_frame)
                    frames_written += 1

                    # Progress indicator every 10 seconds
                    if self.verbose >= 1 and frame_idx % (self.fps * 10) == 0 and frame_idx > 0:
                        elapsed_seconds = frame_idx // self.fps
                        print(f"[PostVideo]   Progress: {elapsed_seconds}s / {self.clip_seconds}s ({frames_written} frames)")

                # Get action from model with epsilon-greedy exploration
                if np.random.random() < exploration_rate:
                    # Random action for exploration
                    action = env.action_space.sample()
                else:
                    # Use model's action (deterministic for consistency)
                    action, _ = model.predict(obs, deterministic=True)

                # For Breakout: ensure FIRE is pressed shortly after reset to launch ball
                # This prevents the agent from getting stuck at the start screen
                if frames_since_reset < 10 and frames_since_reset % 3 == 0:
                    # Action 1 is typically FIRE in Breakout
                    if hasattr(env.action_space, 'n') and env.action_space.n >= 2:
                        action = 1  # FIRE action

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                frames_since_reset += 1

                # Reset if episode ends
                if terminated or truncated:
                    obs, _ = env.reset()
                    total_reward = 0.0
                    frames_since_reset = 0
                    episode_count += 1

            # Release video writer
            if video_writer is not None:
                video_writer.release()

            if self.verbose >= 1:
                print(f"[PostVideo]   Completed: {frames_written} frames written to {output_path.name}")

            return output_path.exists() and frames_written > 0

        except Exception as e:
            print(f"[PostVideo] Error in _record_gameplay_to_video: {e}")
            import traceback
            traceback.print_exc()
            if video_writer is not None:
                video_writer.release()
            return False

    def _record_gameplay_frames(self, model, env) -> List[np.ndarray]:
        """
        DEPRECATED: Record gameplay frames with neural network visualization using the trained model.
        This method stores all frames in memory and should only be used for short videos.
        For long videos, use _record_gameplay_to_video instead.
        """
        frames = []
        target_frames = self.clip_seconds * self.fps

        obs, _ = env.reset()
        total_reward = 0.0
        frames_since_reset = 0
        episode_count = 0

        # Add small exploration to prevent stuck patterns
        exploration_rate = 0.05  # 5% random actions

        for frame_idx in range(target_frames):
            # Get frame from environment
            game_frame = env.render()

            if game_frame is not None:
                # Extract ML analytics for this frame
                analytics = self._extract_ml_analytics(model, obs, frame_idx, total_reward)
                analytics['episode_count'] = episode_count
                analytics['frames_since_reset'] = frames_since_reset

                # Create enhanced frame with neural network visualization
                enhanced_frame = self._create_enhanced_frame(game_frame, analytics)
                frames.append(enhanced_frame)

            # Get action from model with epsilon-greedy exploration
            if np.random.random() < exploration_rate:
                # Random action for exploration
                action = env.action_space.sample()
            else:
                # Use model's action (deterministic for consistency)
                action, _ = model.predict(obs, deterministic=True)

            # For Breakout: ensure FIRE is pressed shortly after reset to launch ball
            # This prevents the agent from getting stuck at the start screen
            if frames_since_reset < 10 and frames_since_reset % 3 == 0:
                # Action 1 is typically FIRE in Breakout
                if hasattr(env.action_space, 'n') and env.action_space.n >= 2:
                    action = 1  # FIRE action

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames_since_reset += 1

            # Reset if episode ends
            if terminated or truncated:
                obs, _ = env.reset()
                total_reward = 0.0
                frames_since_reset = 0
                episode_count += 1

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
                # Frames are already in BGR format from _create_enhanced_frame
                out.write(frame)

            # Release video writer
            out.release()

            return output_path.exists()

        except Exception as e:
            print(f"[PostVideo] Error creating video: {e}")
            return False

    def _extract_ml_analytics(self, model, obs: np.ndarray, frame_idx: int, total_reward: float) -> Dict[str, Any]:
        """Extract ML analytics from the current model state, including real layer activations."""
        analytics = {
            'step': frame_idx,
            'frame_in_video': frame_idx,
            'progress_pct': (frame_idx / (self.clip_seconds * self.fps)) * 100,
            'action_probs': None,
            'value_estimate': 0.0,
            'episode_reward': total_reward,
            'layer_activations': {}  # Store real layer activations
        }

        try:
            if hasattr(model, 'policy'):
                # Prepare observation for policy
                # Gymnasium returns observations in shape (H, W, C) but PyTorch expects (C, H, W)
                import torch

                if len(obs.shape) == 3:
                    # Shape is (H, W, C), need to transpose to (C, H, W) and add batch dimension
                    obs_tensor = np.transpose(obs, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                    obs_tensor = obs_tensor[np.newaxis, ...]   # (C, H, W) -> (1, C, H, W)
                elif len(obs.shape) == 4:
                    # Already has batch dimension, just transpose
                    obs_tensor = np.transpose(obs, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)
                else:
                    obs_tensor = obs[np.newaxis, ...]

                # Convert to torch tensor
                obs_tensor = torch.as_tensor(obs_tensor).float()

                # Move tensor to the same device as the model
                # Check if model is on CUDA
                if hasattr(model.policy, 'device'):
                    obs_tensor = obs_tensor.to(model.policy.device)
                elif next(model.policy.parameters(), None) is not None:
                    # Get device from model parameters
                    device = next(model.policy.parameters()).device
                    obs_tensor = obs_tensor.to(device)

                # Extract layer activations using forward hooks
                activations = {}
                hooks = []

                def get_activation(name):
                    def hook(module, input, output):
                        # Store mean activation across spatial dimensions
                        if isinstance(output, torch.Tensor):
                            # For conv layers: average across spatial dimensions
                            if len(output.shape) == 4:  # (batch, channels, height, width)
                                activations[name] = output[0].mean(dim=(1, 2)).cpu().numpy()
                            # For linear layers: just use the values
                            elif len(output.shape) == 2:  # (batch, features)
                                activations[name] = output[0].cpu().numpy()
                    return hook

                # Register hooks for feature extractor layers
                if hasattr(model.policy, 'features_extractor'):
                    features = model.policy.features_extractor
                    if hasattr(features, 'cnn'):
                        # For NatureCNN architecture
                        cnn = features.cnn
                        if len(cnn) > 0:
                            hooks.append(cnn[0].register_forward_hook(get_activation('conv1')))
                        if len(cnn) > 2:
                            hooks.append(cnn[2].register_forward_hook(get_activation('conv2')))
                        if len(cnn) > 4:
                            hooks.append(cnn[4].register_forward_hook(get_activation('conv3')))

                    # Hook for the linear layer after CNN
                    if hasattr(features, 'linear'):
                        hooks.append(features.linear.register_forward_hook(get_activation('dense')))

                # Get action probabilities and value estimate
                with torch.no_grad():
                    # Get policy distribution
                    if hasattr(model.policy, 'get_distribution'):
                        distribution = model.policy.get_distribution(obs_tensor)
                        if hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'probs'):
                            analytics['action_probs'] = distribution.distribution.probs[0].cpu().numpy()

                    # Get value estimate
                    if hasattr(model.policy, 'predict_values'):
                        values = model.policy.predict_values(obs_tensor)
                        if values is not None:
                            analytics['value_estimate'] = float(values[0].cpu().numpy().item())

                # Remove hooks
                for hook in hooks:
                    hook.remove()

                # Store activations in analytics
                analytics['layer_activations'] = activations

        except Exception as e:
            if self.verbose >= 2:
                print(f"[PostVideo] Analytics extraction error: {e}")

        return analytics

    def _create_enhanced_frame(self, game_frame: np.ndarray, analytics: Dict[str, Any]) -> np.ndarray:
        """Create enhanced frame with neural network visualization."""
        # Resize game frame to fit right side (480x540)
        game_resized = cv2.resize(game_frame, (480, 540))

        # Create analytics panel (left side, 480x540)
        analytics_panel = np.zeros((540, 480, 3), dtype=np.uint8)

        # Draw neural network and analytics
        self._draw_analytics_panel(analytics_panel, analytics)
        self._draw_neural_network(analytics_panel, analytics)

        # Combine panels horizontally (total: 960x540)
        enhanced_frame = np.hstack([analytics_panel, game_resized])

        # Convert RGB to BGR for OpenCV (game_frame is RGB from render())
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

        return enhanced_frame

    def _draw_analytics_panel(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw ML analytics information on the panel."""
        # Colors
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        cyan = (255, 255, 0)

        # Font
        font = cv2.FONT_HERSHEY_SIMPLEX
        mono_font = cv2.FONT_HERSHEY_SIMPLEX

        # Get game name from config
        game_name = self.config.get('game', {}).get('env_id', 'ATARI').split('/')[-1].split('-')[0].upper()
        algo_name = self.config.get('train', {}).get('algo', 'PPO').upper()

        # Title
        cv2.putText(panel, f"{game_name} - {algo_name} Neural Activity Viewer", (10, 30), font, 0.5, white, 1)

        # Subtitle with training parameters
        progress = analytics.get('progress_pct', 0)
        lr = self.config.get('train', {}).get('learning_rate', 0.00025)
        cv2.putText(panel, f"Post-Training Evaluation | Progress: {progress:.1f}% | lr={lr:.1e}", (10, 50), font, 0.35, (180, 180, 180), 1)

        # Training metrics
        y_pos = 80
        line_height = 20

        frame_idx = analytics.get('frame_in_video', 0)
        episode_reward = analytics.get('episode_reward', 0.0)
        episode_count = analytics.get('episode_count', 0)
        frames_since_reset = analytics.get('frames_since_reset', 0)

        cv2.putText(panel, f"Frame:      {frame_idx:,}", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height
        cv2.putText(panel, f"Progress:   {progress:.2f}%", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height
        cv2.putText(panel, f"Episode:    #{episode_count}", (10, int(y_pos)), mono_font, 0.4, cyan, 1)
        y_pos += line_height
        cv2.putText(panel, f"Ep Reward:  {episode_reward:.1f}", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height

        # Show "NEW EPISODE" indicator for first few frames after reset
        if frames_since_reset < 30:
            cv2.putText(panel, ">>> NEW EPISODE <<<", (10, int(y_pos)), mono_font, 0.4, yellow, 1)
        y_pos += line_height * 1.5

        # Action probabilities
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs')

        if action_probs is not None and len(action_probs) >= 4:
            cv2.putText(panel, "Policy Distribution:", (10, int(y_pos)), mono_font, 0.4, (200, 200, 200), 1)
            y_pos += line_height * 0.8

            for i, (name, prob) in enumerate(zip(action_names, action_probs[:4])):
                color = yellow if prob == max(action_probs[:4]) else (150, 150, 150)
                cv2.putText(panel, f"  {name:5s}: {prob:.3f}", (10, int(y_pos)), mono_font, 0.35, color, 1)
                y_pos += line_height * 0.8

        y_pos += line_height * 0.5

        # Value estimate
        value = analytics.get('value_estimate', 0.0)
        cv2.putText(panel, f"Value Est:  {value:.3f}", (10, int(y_pos)), mono_font, 0.4, green, 1)

    def _draw_neural_network(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw neural network visualization with REAL layer activations from the model."""
        # Network area
        net_x, net_y = 20, 280
        net_width, net_height = 440, 240

        # Layer configuration (realistic CNN for Atari)
        layers = [
            {'name': 'Input', 'shape': '84×84×4', 'nodes': 8, 'color': (0, 255, 0), 'key': None},      # Green
            {'name': 'Conv1', 'shape': '32@20×20', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv1'},   # Blue
            {'name': 'Conv2', 'shape': '64@9×9', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv2'},     # Blue
            {'name': 'Dense', 'shape': '512', 'nodes': 8, 'color': (100, 100, 255), 'key': 'dense'},      # Light blue
            {'name': 'Output', 'shape': '4', 'nodes': 4, 'color': (0, 165, 255), 'key': 'output'}          # Orange
        ]

        # Get real layer activations
        layer_activations = analytics.get('layer_activations', {})

        # Calculate positions
        layer_spacing = net_width // (len(layers) - 1)
        layer_positions = []

        for i, layer in enumerate(layers):
            x = net_x + i * layer_spacing
            nodes = layer['nodes']
            node_spacing = net_height // (nodes + 1)

            positions = []
            for j in range(nodes):
                y = net_y + (j + 1) * node_spacing
                positions.append((x, y))

            layer_positions.append(positions)

        # Get node activations for each layer
        step = analytics.get('step', 0)
        layer_node_activations = []

        for layer_idx, layer in enumerate(layers):
            layer_key = layer.get('key')
            num_nodes = layer['nodes']

            if layer_idx == 0:  # Input layer - use constant moderate activation
                activations = [0.5] * num_nodes
            elif layer_idx == len(layers) - 1:  # Output layer - use action probabilities
                action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])
                activations = [action_probs[i] if i < len(action_probs) else 0.25 for i in range(num_nodes)]
            elif layer_key and layer_key in layer_activations:
                # Use REAL activations from the network
                real_activations = layer_activations[layer_key]
                # Normalize to 0-1 range and sample nodes
                if len(real_activations) > 0:
                    # Take absolute value and normalize
                    abs_activations = np.abs(real_activations)
                    max_val = abs_activations.max() if abs_activations.max() > 0 else 1.0
                    normalized = abs_activations / max_val

                    # Sample evenly across the activation array to get values for our display nodes
                    indices = np.linspace(0, len(normalized) - 1, num_nodes, dtype=int)
                    activations = [float(normalized[i]) for i in indices]
                else:
                    activations = [0.5] * num_nodes
            else:
                # Fallback to moderate activation if no real data available
                activations = [0.5] * num_nodes

            layer_node_activations.append(activations)

        # Draw connections based on real activations
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]
            current_activations = layer_node_activations[i]
            next_activations = layer_node_activations[i + 1]

            for j, start_pos in enumerate(current_layer):
                for k, end_pos in enumerate(next_layer):
                    # Connection strength based on REAL activations of connected nodes
                    current_act = current_activations[j] if j < len(current_activations) else 0.5
                    next_act = next_activations[k] if k < len(next_activations) else 0.5
                    strength = (current_act + next_act) / 2.0

                    if strength > 0.3:  # Only draw connections with sufficient activation
                        alpha = int(strength * 200)
                        color = (alpha // 3, alpha // 3, alpha)
                        thickness = 1 if strength > 0.6 else 1
                        cv2.line(panel, start_pos, end_pos, color, thickness)

        # Draw nodes with REAL activations
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])

        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            activations = layer_node_activations[layer_idx]

            for node_idx, pos in enumerate(positions):
                # Get REAL activation for this node
                activation = activations[node_idx] if node_idx < len(activations) else 0.5

                # Node color based on REAL activation
                base_color = layer['color']
                brightness = int(activation * 255)
                node_color = tuple(int(c * brightness / 255) for c in base_color)

                # Draw node
                radius = 8 if layer_idx == len(layers) - 1 else 6
                cv2.circle(panel, pos, radius, node_color, -1)
                cv2.circle(panel, pos, radius, (255, 255, 255), 1)

                # Highlight chosen action
                if layer_idx == len(layers) - 1 and action_probs is not None:
                    if node_idx < len(action_probs) and action_probs[node_idx] == max(action_probs):
                        # Pulsing effect
                        pulse = int(128 + 127 * np.sin(step * 0.1))
                        cv2.circle(panel, pos, radius + 2, (pulse, pulse, pulse), 2)

        # Layer labels
        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            if positions:
                label_x = positions[0][0]
                label_y = net_y + net_height + 20

                # Layer name
                cv2.putText(panel, layer['name'], (label_x - 20, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                # Layer shape
                cv2.putText(panel, layer['shape'], (label_x - 20, label_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Output action labels
        if len(layer_positions) > 0:
            output_positions = layer_positions[-1]
            for i, (pos, name) in enumerate(zip(output_positions, action_names)):
                if i < len(output_positions):
                    cv2.putText(panel, name, (pos[0] + 15, pos[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    def _concatenate_videos(self, video_paths: List[Path], output_path: Path) -> bool:
        """
        Concatenate multiple videos into a single continuous video using FFmpeg.

        Args:
            video_paths: List of video file paths to concatenate
            output_path: Path for the output concatenated video

        Returns:
            True if concatenation succeeded, False otherwise
        """
        try:
            import tempfile

            # Create a temporary file list for FFmpeg concat demuxer
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for video_path in video_paths:
                    # FFmpeg concat demuxer requires absolute paths
                    abs_path = video_path.resolve()
                    # Escape single quotes and write in FFmpeg concat format
                    escaped_path = str(abs_path).replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Use FFmpeg to concatenate videos
            ffmpeg_cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # Copy streams without re-encoding for speed
                '-y',  # Overwrite output file
                str(output_path)
            ]

            if self.verbose >= 2:
                print(f"[PostVideo] Running FFmpeg: {' '.join(ffmpeg_cmd)}")

            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                check=False
            )

            # Clean up temp file
            try:
                os.unlink(concat_file)
            except:
                pass

            if result.returncode != 0:
                print(f"[PostVideo] FFmpeg concatenation failed:")
                print(f"  stderr: {result.stderr}")
                return False

            return output_path.exists()

        except Exception as e:
            print(f"[PostVideo] Error concatenating videos: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Generate videos after training')
    parser.add_argument('--model-dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='video/post_training',
                       help='Output directory for videos')
    parser.add_argument('--clip-seconds', type=int, default=10,
                       help='Length of each video clip in seconds (for milestone mode)')
    parser.add_argument('--total-seconds', type=int, default=None,
                       help='Total video length in seconds (generates one continuous video)')
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
    if args.total_seconds:
        # Generate single continuous video
        video_path = generator.generate_continuous_video(args.total_seconds)
        if video_path:
            print(f"\n[VIDEO] Continuous video generated!")
            print(f"  [VIDEO] {video_path.name}")
        else:
            print(f"\n[VIDEO] Failed to generate continuous video")
    else:
        # Generate separate milestone videos
        generated_videos = generator.generate_all_videos()

        print(f"\n[VIDEO] Video generation complete!")
        print(f"Generated {len(generated_videos)} videos in {args.output_dir}")

        for video_path in generated_videos:
            print(f"  [VIDEO] {video_path.name}")


if __name__ == "__main__":
    main()
