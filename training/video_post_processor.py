#!/usr/bin/env python3
"""
Video post-processing tools for training videos.
Provides functions for creating time-lapses, compilations, and comparisons.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import time

import numpy as np
import cv2
import torch
from stable_baselines3 import PPO, DQN

try:
    from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, clips_array
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from envs.make_env import make_single_env

logger = logging.getLogger(__name__)


class VideoPostProcessor:
    """Post-processing tools for training videos."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize video post-processor.
        
        Args:
            config: Configuration dictionary
        """
        if not MOVIEPY_AVAILABLE:
            raise ImportError(
                "MoviePy is required for video post-processing. "
                "Install it with: pip install moviepy"
            )
        
        self.config = config
        self.video_training_dir = Path(config.get('paths', {}).get('videos_training', 'video/training'))
        self.video_output_dir = Path(config.get('paths', {}).get('videos_output', 'video/output'))
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
    
    def concatenate_training_videos(
        self,
        run_id: str,
        output_filename: Optional[str] = None,
        speed_multiplier: float = 1.0,
        max_duration: Optional[float] = None
    ) -> Optional[str]:
        """
        Concatenate all episode videos from a training run into a single video.
        
        Args:
            run_id: Training run ID
            output_filename: Output filename (default: {run_id}_training.mp4)
            speed_multiplier: Speed up factor (e.g., 2.0 = 2x speed)
            max_duration: Maximum duration in seconds (None = no limit)
            
        Returns:
            Path to output video file, or None if failed
        """
        logger.info(f"Concatenating training videos for run: {run_id}")
        
        # Find all episode videos for this run
        run_dir = self.video_training_dir / run_id
        if not run_dir.exists():
            logger.error(f"Training video directory not found: {run_dir}")
            return None
        
        # Get all episode videos sorted by episode number
        video_files = sorted(
            run_dir.glob("env_0-episode-*.mp4"),
            key=lambda p: int(p.stem.split('-')[-1])
        )
        
        if not video_files:
            logger.error(f"No episode videos found in {run_dir}")
            return None
        
        logger.info(f"Found {len(video_files)} episode videos")
        
        try:
            # Load all video clips
            clips = []
            total_duration = 0
            
            for video_file in video_files:
                clip = VideoFileClip(str(video_file))

                # Apply speed multiplier if specified
                if speed_multiplier != 1.0:
                    clip = clip.with_speed_scaled(speed_multiplier)

                # Check if we've reached max duration
                if max_duration and (total_duration + clip.duration) > max_duration:
                    # Trim the last clip to fit max_duration
                    remaining = max_duration - total_duration
                    if remaining > 0:
                        clip = clip.subclipped(0, remaining)
                        clips.append(clip)
                    break

                clips.append(clip)
                total_duration += clip.duration
            
            if not clips:
                logger.error("No clips to concatenate")
                return None
            
            # Concatenate all clips
            logger.info(f"Concatenating {len(clips)} clips (total duration: {total_duration:.1f}s)")
            final_clip = concatenate_videoclips(clips, method="compose")
            
            # Generate output filename
            if output_filename is None:
                speed_suffix = f"_{speed_multiplier}x" if speed_multiplier != 1.0 else ""
                output_filename = f"{run_id}_training{speed_suffix}.mp4"
            
            output_path = self.video_output_dir / output_filename
            
            # Write output video
            logger.info(f"Writing output video: {output_path}")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                preset='medium',
                logger=None  # Suppress moviepy progress bar
            )
            
            # Clean up
            for clip in clips:
                clip.close()
            final_clip.close()
            
            logger.info(f"Successfully created training video: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to concatenate videos: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_milestone_progression_video(
        self,
        run_id: str,
        milestone_percentages: List[int] = [10, 50, 100],
        output_filename: Optional[str] = None,
        layout: str = "horizontal",
        clip_duration: float = 30.0
    ) -> Optional[str]:
        """
        Create a side-by-side comparison video showing AI progress at different milestones.

        Args:
            run_id: Training run ID
            milestone_percentages: List of milestone percentages to compare (e.g., [10, 50, 100])
            output_filename: Output filename (default: {run_id}_progression.mp4)
            layout: Layout style - "horizontal", "vertical", or "grid"
            clip_duration: Duration of each clip in seconds

        Returns:
            Path to output video file, or None if failed
        """
        logger.info(f"Creating milestone progression video for run: {run_id}")

        # Find training videos for each milestone
        run_dir = self.video_training_dir / run_id
        if not run_dir.exists():
            logger.error(f"Training video directory not found: {run_dir}")
            return None

        try:
            clips = []
            labels = []

            for pct in milestone_percentages:
                # Find videos around this milestone
                # For now, we'll use the first few episodes as a proxy
                # In a real implementation, we'd track which episodes correspond to which milestones
                episode_num = int((pct / 100.0) * 20)  # Rough estimate
                video_file = run_dir / f"env_0-episode-{episode_num}.mp4"

                if not video_file.exists():
                    # Try to find the closest episode
                    all_episodes = sorted(run_dir.glob("env_0-episode-*.mp4"))
                    if not all_episodes:
                        logger.warning(f"No episodes found for {pct}% milestone")
                        continue

                    # Use the episode closest to our target
                    target_idx = min(episode_num, len(all_episodes) - 1)
                    video_file = all_episodes[target_idx]

                logger.info(f"Loading {pct}% milestone video: {video_file}")
                clip = VideoFileClip(str(video_file))

                # Trim to desired duration
                if clip.duration > clip_duration:
                    clip = clip.subclipped(0, clip_duration)

                clips.append(clip)
                labels.append(f"{pct}% Training")

            if not clips:
                logger.error("No milestone clips found")
                return None

            # Create side-by-side layout
            if layout == "horizontal":
                # Resize clips to fit horizontally
                target_width = 640 // len(clips)
                target_height = 480
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]
                final_clip = clips_array([resized_clips])

            elif layout == "vertical":
                # Resize clips to fit vertically
                target_width = 640
                target_height = 480 // len(clips)
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]
                final_clip = clips_array([[clip] for clip in resized_clips])

            elif layout == "grid":
                # Create 2x2 grid (or adjust based on number of clips)
                import math
                grid_size = math.ceil(math.sqrt(len(clips)))
                target_width = 640 // grid_size
                target_height = 480 // grid_size
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]

                # Pad clips to fill grid
                while len(resized_clips) < grid_size * grid_size:
                    # Create a black clip as filler
                    black_clip = ImageClip(
                        [[0, 0, 0]] * target_height * target_width,
                        duration=clip_duration
                    ).set_duration(clip_duration)
                    resized_clips.append(black_clip)

                # Arrange in grid
                rows = []
                for i in range(grid_size):
                    row = resized_clips[i * grid_size:(i + 1) * grid_size]
                    rows.append(row)
                final_clip = clips_array(rows)
            else:
                logger.error(f"Unknown layout: {layout}")
                return None

            # Generate output filename
            if output_filename is None:
                output_filename = f"{run_id}_progression_{layout}.mp4"

            output_path = self.video_output_dir / output_filename

            # Write output video
            logger.info(f"Writing progression video: {output_path}")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                preset='medium',
                logger=None
            )

            # Clean up
            for clip in clips:
                clip.close()
            final_clip.close()

            logger.info(f"Successfully created progression video: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create progression video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_timelapse(
        self,
        run_id: str,
        speed_multiplier: float = 10.0,
        target_duration: Optional[float] = None,
        output_filename: Optional[str] = None,
        add_overlays: bool = True
    ) -> Optional[str]:
        """
        Create a time-lapse video of the entire training run.

        Args:
            run_id: Training run ID
            speed_multiplier: Speed up factor (e.g., 10.0 = 10x speed)
            target_duration: Target duration in seconds (overrides speed_multiplier if set)
            output_filename: Output filename (default: {run_id}_timelapse.mp4)
            add_overlays: If True, generate new footage with neural network overlays (slower but more informative)

        Returns:
            Path to output video file, or None if failed
        """
        logger.info(f"Creating time-lapse for run: {run_id} (speed: {speed_multiplier}x, overlays: {add_overlays})")

        if add_overlays:
            # Generate new footage with overlays using the model
            return self._create_timelapse_with_overlays(
                run_id=run_id,
                speed_multiplier=speed_multiplier,
                target_duration=target_duration,
                output_filename=output_filename
            )
        else:
            # Use existing training videos (raw footage, no overlays)
            # If target_duration is specified, calculate required speed multiplier
            if target_duration:
                run_dir = self.video_training_dir / run_id
                if not run_dir.exists():
                    logger.error(f"Training video directory not found: {run_dir}")
                    return None

                # Calculate total duration of all videos
                video_files = list(run_dir.glob("env_0-episode-*.mp4"))
                total_duration = 0
                for video_file in video_files:
                    try:
                        clip = VideoFileClip(str(video_file))
                        total_duration += clip.duration
                        clip.close()
                    except Exception as e:
                        logger.warning(f"Failed to read {video_file}: {e}")

                if total_duration > 0:
                    speed_multiplier = total_duration / target_duration
                    logger.info(f"Calculated speed multiplier: {speed_multiplier:.2f}x for target duration {target_duration}s")

            # Use concatenate_training_videos with speed multiplier
            if output_filename is None:
                output_filename = f"{run_id}_timelapse_{speed_multiplier:.0f}x.mp4"

            return self.concatenate_training_videos(
                run_id=run_id,
                output_filename=output_filename,
                speed_multiplier=speed_multiplier
            )

    def compare_multiple_runs(
        self,
        run_ids: List[str],
        output_filename: Optional[str] = None,
        layout: str = "grid",
        clip_duration: float = 60.0,
        labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create a side-by-side comparison of multiple training runs.

        Args:
            run_ids: List of training run IDs to compare
            output_filename: Output filename (default: comparison_{timestamp}.mp4)
            layout: Layout style - "horizontal", "vertical", or "grid"
            clip_duration: Duration of each clip in seconds
            labels: Optional labels for each run (default: run IDs)

        Returns:
            Path to output video file, or None if failed
        """
        logger.info(f"Creating comparison video for {len(run_ids)} runs")

        if labels is None:
            labels = run_ids

        try:
            clips = []

            for run_id in run_ids:
                run_dir = self.video_training_dir / run_id
                if not run_dir.exists():
                    logger.warning(f"Training video directory not found: {run_dir}")
                    continue

                # Get first episode as representative sample
                video_files = sorted(run_dir.glob("env_0-episode-*.mp4"))
                if not video_files:
                    logger.warning(f"No videos found for run: {run_id}")
                    continue

                # Use first episode
                video_file = video_files[0]
                logger.info(f"Loading video for {run_id}: {video_file}")
                clip = VideoFileClip(str(video_file))

                # Trim to desired duration
                if clip.duration > clip_duration:
                    clip = clip.subclipped(0, clip_duration)

                clips.append(clip)

            if not clips:
                logger.error("No clips found for comparison")
                return None

            # Create layout (similar to milestone progression)
            if layout == "horizontal":
                target_width = 640 // len(clips)
                target_height = 480
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]
                final_clip = clips_array([resized_clips])

            elif layout == "vertical":
                target_width = 640
                target_height = 480 // len(clips)
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]
                final_clip = clips_array([[clip] for clip in resized_clips])

            elif layout == "grid":
                import math
                grid_size = math.ceil(math.sqrt(len(clips)))
                target_width = 640 // grid_size
                target_height = 480 // grid_size
                resized_clips = [clip.resized((target_width, target_height)) for clip in clips]

                # Pad clips to fill grid
                while len(resized_clips) < grid_size * grid_size:
                    black_clip = ImageClip(
                        [[0, 0, 0]] * target_height * target_width,
                        duration=clip_duration
                    ).set_duration(clip_duration)
                    resized_clips.append(black_clip)

                # Arrange in grid
                rows = []
                for i in range(grid_size):
                    row = resized_clips[i * grid_size:(i + 1) * grid_size]
                    rows.append(row)
                final_clip = clips_array(rows)
            else:
                logger.error(f"Unknown layout: {layout}")
                return None

            # Generate output filename
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"comparison_{timestamp}.mp4"

            output_path = self.video_output_dir / output_filename

            # Write output video
            logger.info(f"Writing comparison video: {output_path}")
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                preset='medium',
                logger=None
            )

            # Clean up
            for clip in clips:
                clip.close()
            final_clip.close()

            logger.info(f"Successfully created comparison video: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create comparison video: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ========== OVERLAY GENERATION METHODS ==========

    def _load_model(self, model_path: Path):
        """Load model from checkpoint file."""
        try:
            algorithm = self.config.get('train', {}).get('algo', 'ppo').upper()

            logger.info(f"Loading {algorithm} model from {model_path}")

            if algorithm == 'PPO':
                return PPO.load(str(model_path))
            elif algorithm == 'DQN':
                return DQN.load(str(model_path))
            else:
                logger.error(f"Unsupported algorithm: {algorithm}")
                return None

        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return None

    def _extract_ml_analytics(self, model, obs: np.ndarray, frame_idx: int, total_reward: float,
                              clip_seconds: int, fps: int, training_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract ML analytics from the current model state, including real layer activations."""
        analytics = {
            'step': frame_idx,
            'frame_in_video': frame_idx,
            'progress_pct': (frame_idx / (clip_seconds * fps)) * 100,
            'action_probs': None,
            'value_estimate': 0.0,
            'episode_reward': total_reward,
            'layer_activations': {},
            'training_context': training_context or {}
        }

        try:
            if hasattr(model, 'policy'):
                # Prepare observation for policy
                if len(obs.shape) == 3:
                    obs_tensor = np.transpose(obs, (2, 0, 1))
                    obs_tensor = obs_tensor[np.newaxis, ...]
                elif len(obs.shape) == 4:
                    obs_tensor = np.transpose(obs, (0, 3, 1, 2))
                else:
                    obs_tensor = obs[np.newaxis, ...]

                obs_tensor = torch.as_tensor(obs_tensor).float()

                # Move to model device
                if hasattr(model.policy, 'device'):
                    obs_tensor = obs_tensor.to(model.policy.device)
                elif next(model.policy.parameters(), None) is not None:
                    device = next(model.policy.parameters()).device
                    obs_tensor = obs_tensor.to(device)

                # Extract layer activations
                activations = {}
                hooks = []

                def get_activation(name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            if len(output.shape) == 4:
                                activations[name] = output[0].mean(dim=(1, 2)).cpu().numpy()
                            elif len(output.shape) == 2:
                                activations[name] = output[0].cpu().numpy()
                    return hook

                # Register hooks
                if hasattr(model.policy, 'features_extractor'):
                    features = model.policy.features_extractor
                    if hasattr(features, 'cnn'):
                        cnn = features.cnn
                        if len(cnn) > 0:
                            hooks.append(cnn[0].register_forward_hook(get_activation('conv1')))
                        if len(cnn) > 2:
                            hooks.append(cnn[2].register_forward_hook(get_activation('conv2')))
                        if len(cnn) > 4:
                            hooks.append(cnn[4].register_forward_hook(get_activation('conv3')))

                    if hasattr(features, 'linear'):
                        hooks.append(features.linear.register_forward_hook(get_activation('dense')))

                # Get action probabilities and value estimate
                with torch.no_grad():
                    if hasattr(model.policy, 'get_distribution'):
                        distribution = model.policy.get_distribution(obs_tensor)
                        if hasattr(distribution, 'distribution') and hasattr(distribution.distribution, 'probs'):
                            analytics['action_probs'] = distribution.distribution.probs[0].cpu().numpy()

                    if hasattr(model.policy, 'predict_values'):
                        values = model.policy.predict_values(obs_tensor)
                        if values is not None:
                            analytics['value_estimate'] = float(values[0].cpu().numpy().item())

                # Remove hooks
                for hook in hooks:
                    hook.remove()

                analytics['layer_activations'] = activations

        except Exception as e:
            logger.debug(f"Analytics extraction error: {e}")

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

        # Convert RGB to BGR for OpenCV
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

        return enhanced_frame

    def _draw_analytics_panel(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw ML analytics information on the panel."""
        # Colors
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        cyan = (255, 255, 0)
        orange = (0, 165, 255)
        gray = (180, 180, 180)

        font = cv2.FONT_HERSHEY_SIMPLEX
        mono_font = cv2.FONT_HERSHEY_SIMPLEX

        # Get game name from config
        game_name = self.config.get('game', {}).get('env_id', 'ATARI').split('/')[-1].split('-')[0].upper()
        algo_name = self.config.get('train', {}).get('algo', 'PPO').upper()

        # Title
        cv2.putText(panel, f"{game_name} - {algo_name} Neural Activity Viewer", (10, 30), font, 0.5, white, 1)

        # Get training context
        training_context = analytics.get('training_context', {})

        # Subtitle
        if training_context and training_context.get('total_timesteps', 0) > 0:
            checkpoint_timestep = training_context.get('checkpoint_timestep', 0)
            total_timesteps = training_context.get('total_timesteps', 0)
            training_progress = training_context.get('training_progress_pct', 0)
            cv2.putText(panel, f"Training: {checkpoint_timestep:,} / {total_timesteps:,} steps ({training_progress:.0f}%)",
                       (10, 50), font, 0.35, orange, 1)
        else:
            progress = analytics.get('progress_pct', 0)
            lr = self.config.get('train', {}).get('learning_rate', 0.00025)
            cv2.putText(panel, f"Post-Training Evaluation | Progress: {progress:.1f}% | lr={lr:.1e}",
                       (10, 50), font, 0.35, gray, 1)

        # Metrics section
        y_pos = 75
        line_height = 18

        # Training progress (if available)
        if training_context and training_context.get('total_timesteps', 0) > 0:
            cv2.putText(panel, "=== TRAINING PROGRESS ===", (10, int(y_pos)), mono_font, 0.35, cyan, 1)
            y_pos += line_height

            training_hours = training_context.get('training_duration_hours', 0)
            training_episodes = training_context.get('training_episodes_completed', 0)
            best_reward = training_context.get('best_reward', 0)
            avg_reward = training_context.get('avg_reward', 0)

            cv2.putText(panel, f"Duration:   {training_hours:.1f} hours", (10, int(y_pos)), mono_font, 0.35, white, 1)
            y_pos += line_height

            if training_episodes > 0:
                cv2.putText(panel, f"Episodes:   {training_episodes:,} completed", (10, int(y_pos)), mono_font, 0.35, cyan, 1)
                y_pos += line_height

            cv2.putText(panel, f"Best Reward: {best_reward:.1f}", (10, int(y_pos)), mono_font, 0.35, green, 1)
            y_pos += line_height
            cv2.putText(panel, f"Avg Reward:  {avg_reward:.1f}", (10, int(y_pos)), mono_font, 0.35, white, 1)
            y_pos += line_height * 1.3

        # Current playback section
        cv2.putText(panel, "=== PLAYBACK ===", (10, int(y_pos)), mono_font, 0.35, cyan, 1)
        y_pos += line_height

        frame_idx = analytics.get('frame_in_video', 0)
        episode_reward = analytics.get('episode_reward', 0.0)
        episode_count = analytics.get('episode_count', 1)
        frames_since_reset = analytics.get('frames_since_reset', 0)

        # Highlight episode number if new episode
        episode_color = yellow if frames_since_reset < 60 else white
        cv2.putText(panel, f"Episode:    #{episode_count}", (10, int(y_pos)), mono_font, 0.35, episode_color,
                   2 if frames_since_reset < 60 else 1)
        y_pos += line_height

        cv2.putText(panel, f"Score:      {episode_reward:.1f}", (10, int(y_pos)), mono_font, 0.35, white, 1)
        y_pos += line_height
        cv2.putText(panel, f"Frame:      {frame_idx:,}", (10, int(y_pos)), mono_font, 0.35, gray, 1)
        y_pos += line_height

        # New episode indicator
        if frames_since_reset < 60:
            pulse_color = yellow if (frames_since_reset // 10) % 2 == 0 else orange
            cv2.putText(panel, ">>> NEW EPISODE <<<", (10, int(y_pos)), mono_font, 0.4, pulse_color, 2)
        y_pos += line_height * 1.3

        # Action probabilities
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs')

        if action_probs is not None and len(action_probs) >= 4:
            cv2.putText(panel, "=== POLICY ===", (10, int(y_pos)), mono_font, 0.35, cyan, 1)
            y_pos += line_height

            for i, (name, prob) in enumerate(zip(action_names, action_probs[:4])):
                color = yellow if prob == max(action_probs[:4]) else (150, 150, 150)
                cv2.putText(panel, f"  {name:5s}: {prob:.3f}", (10, int(y_pos)), mono_font, 0.35, color, 1)
                y_pos += line_height * 0.9

        y_pos += line_height * 0.3

        # Value estimate
        value = analytics.get('value_estimate', 0.0)
        cv2.putText(panel, f"Value Est:  {value:.3f}", (10, int(y_pos)), mono_font, 0.35, green, 1)

    def _draw_neural_network(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw neural network visualization with real layer activations."""
        # Network area
        net_x, net_y = 20, 280
        net_width, net_height = 440, 240

        # Layer configuration
        layers = [
            {'name': 'Input', 'shape': '84×84×4', 'nodes': 8, 'color': (0, 255, 0), 'key': None},
            {'name': 'Conv1', 'shape': '32@20×20', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv1'},
            {'name': 'Conv2', 'shape': '64@9×9', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv2'},
            {'name': 'Dense', 'shape': '512', 'nodes': 8, 'color': (100, 100, 255), 'key': 'dense'},
            {'name': 'Output', 'shape': '4', 'nodes': 4, 'color': (0, 165, 255), 'key': 'output'}
        ]

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

        # Get node activations
        step = analytics.get('step', 0)
        layer_node_activations = []

        for layer_idx, layer in enumerate(layers):
            layer_key = layer.get('key')
            num_nodes = layer['nodes']

            if layer_idx == 0:
                activations = [0.5] * num_nodes
            elif layer_idx == len(layers) - 1:
                action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])
                activations = [action_probs[i] if i < len(action_probs) else 0.25 for i in range(num_nodes)]
            elif layer_key and layer_key in layer_activations:
                real_activations = layer_activations[layer_key]
                if len(real_activations) > 0:
                    abs_activations = np.abs(real_activations)
                    max_val = abs_activations.max() if abs_activations.max() > 0 else 1.0
                    normalized = abs_activations / max_val
                    indices = np.linspace(0, len(normalized) - 1, num_nodes, dtype=int)
                    activations = [float(normalized[i]) for i in indices]
                else:
                    activations = [0.5] * num_nodes
            else:
                activations = [0.5] * num_nodes

            layer_node_activations.append(activations)

        # Draw connections
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]
            current_activations = layer_node_activations[i]
            next_activations = layer_node_activations[i + 1]

            for j, start_pos in enumerate(current_layer):
                for k, end_pos in enumerate(next_layer):
                    current_act = current_activations[j] if j < len(current_activations) else 0.5
                    next_act = next_activations[k] if k < len(next_activations) else 0.5
                    strength = (current_act + next_act) / 2.0

                    if strength > 0.3:
                        alpha = int(strength * 200)
                        color = (alpha // 3, alpha // 3, alpha)
                        cv2.line(panel, start_pos, end_pos, color, 1)

        # Draw nodes
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])

        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            activations = layer_node_activations[layer_idx]

            for node_idx, pos in enumerate(positions):
                activation = activations[node_idx] if node_idx < len(activations) else 0.5

                base_color = layer['color']
                brightness = int(activation * 255)
                node_color = tuple(int(c * brightness / 255) for c in base_color)

                radius = 8 if layer_idx == len(layers) - 1 else 6
                cv2.circle(panel, pos, radius, node_color, -1)
                cv2.circle(panel, pos, radius, (255, 255, 255), 1)

                # Highlight chosen action
                if layer_idx == len(layers) - 1 and action_probs is not None:
                    if node_idx < len(action_probs) and action_probs[node_idx] == max(action_probs):
                        pulse = int(128 + 127 * np.sin(step * 0.1))
                        cv2.circle(panel, pos, radius + 2, (pulse, pulse, pulse), 2)

        # Layer labels
        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            if positions:
                label_x = positions[0][0]
                label_y = net_y + net_height + 20

                cv2.putText(panel, layer['name'], (label_x - 20, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
                cv2.putText(panel, layer['shape'], (label_x - 20, label_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Output action labels
        if len(layer_positions) > 0:
            output_positions = layer_positions[-1]
            for i, (pos, name) in enumerate(zip(output_positions, action_names)):
                if i < len(output_positions):
                    cv2.putText(panel, name, (pos[0] + 15, pos[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)

    def _generate_video_with_overlays(
        self,
        model_path: Path,
        output_path: Path,
        clip_seconds: int,
        fps: int = 30,
        training_context: Dict[str, Any] = None
    ) -> bool:
        """
        Generate a video with overlays by running the model.

        Args:
            model_path: Path to model checkpoint
            output_path: Path for output video
            clip_seconds: Duration in seconds
            fps: Frames per second
            training_context: Optional training context for overlay

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            model = self._load_model(model_path)
            if model is None:
                logger.error("Failed to load model")
                return False

            # Create environment with rgb_array render mode for frame capture
            # IMPORTANT: Use full_action_space=True to match the training environment
            env_id = self.config['game']['env_id']

            # Import make_atari_env to pass full_action_space parameter
            from envs.atari_wrappers import make_atari_env
            from gymnasium.wrappers import RecordEpisodeStatistics

            # Create environment with full action space
            env = make_atari_env(
                env_id=env_id,
                config=self.config,
                seed=42,
                render_mode="rgb_array",
                full_action_space=True  # Match training environment
            )

            # Add episode statistics tracking
            env = RecordEpisodeStatistics(env)

            # Debug: Log action space info
            logger.info(f"Environment action space: {env.action_space}")
            logger.info(f"Action space n: {env.action_space.n if hasattr(env.action_space, 'n') else 'N/A'}")

            # Calculate target frames
            target_frames = clip_seconds * fps

            # Initialize video writer
            video_writer = None
            frames_written = 0

            obs, _ = env.reset()
            total_reward = 0.0
            frames_since_reset = 0
            episode_count = 1

            exploration_rate = 0.05

            logger.info(f"Generating {clip_seconds}s video with overlays ({target_frames} frames)")

            for frame_idx in range(target_frames):
                # Get frame from environment
                game_frame = env.render()

                if game_frame is not None:
                    # Extract analytics
                    analytics = self._extract_ml_analytics(
                        model, obs, frame_idx, total_reward,
                        clip_seconds, fps, training_context
                    )
                    analytics['episode_count'] = episode_count
                    analytics['frames_since_reset'] = frames_since_reset

                    # Create enhanced frame
                    enhanced_frame = self._create_enhanced_frame(game_frame, analytics)

                    # Initialize video writer on first frame
                    if video_writer is None:
                        height, width = enhanced_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                        if not video_writer.isOpened():
                            logger.error(f"Failed to open video writer for {output_path}")
                            env.close()
                            return False

                    # Write frame
                    video_writer.write(enhanced_frame)
                    frames_written += 1

                # Get action
                if np.random.random() < exploration_rate:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)

                # Convert action to int (model.predict returns numpy array)
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)

                # Fire action for Breakout
                if frames_since_reset < 10 and frames_since_reset % 3 == 0:
                    if hasattr(env.action_space, 'n') and env.action_space.n >= 2:
                        action = 1

                # Debug: Log action on first few frames
                if frame_idx < 5:
                    logger.info(f"Frame {frame_idx}: action={action}, type={type(action)}, action_space.n={env.action_space.n if hasattr(env.action_space, 'n') else 'N/A'}")

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                frames_since_reset += 1

                # Handle episode end
                if terminated or truncated:
                    obs, _ = env.reset()
                    total_reward = 0.0
                    frames_since_reset = 0
                    episode_count += 1

            # Cleanup
            if video_writer is not None:
                video_writer.release()
            env.close()

            logger.info(f"Successfully generated video with overlays: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating video with overlays: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _create_timelapse_with_overlays(
        self,
        run_id: str,
        speed_multiplier: float,
        target_duration: Optional[float],
        output_filename: Optional[str]
    ) -> Optional[str]:
        """
        Create a time-lapse video with neural network overlays.

        This generates new footage by running the trained model, so it includes
        all the overlay information (neural network visualization, stats, etc.).
        """
        try:
            # Find the model checkpoint for this run
            model_dir = Path("models/checkpoints") / run_id
            if not model_dir.exists():
                # Try alternative location
                model_dir = Path("models/checkpoints")

            # Look for the latest or final checkpoint
            checkpoint_patterns = [
                f"{run_id}_final.zip",
                f"{run_id}_latest.zip",
                "latest.zip",
                "final.zip"
            ]

            model_path = None
            for pattern in checkpoint_patterns:
                potential_path = model_dir / pattern
                if potential_path.exists():
                    model_path = potential_path
                    break

            if model_path is None:
                # Try to find any checkpoint file
                checkpoints = list(model_dir.glob("*.zip"))
                if checkpoints:
                    # Use the most recent one
                    model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                else:
                    logger.error(f"No model checkpoint found for run {run_id}")
                    return None

            logger.info(f"Using model checkpoint: {model_path}")

            # Calculate clip duration
            if target_duration:
                # Generate footage for target duration, then speed it up
                base_duration = int(target_duration * speed_multiplier)
            else:
                # Default: 60 seconds of footage, sped up
                base_duration = 60

            # Generate temporary video with overlays
            temp_output = self.video_output_dir / f"temp_{run_id}_with_overlays.mp4"

            success = self._generate_video_with_overlays(
                model_path=model_path,
                output_path=temp_output,
                clip_seconds=base_duration,
                fps=30,
                training_context=None  # Could add training context here if available
            )

            if not success:
                logger.error("Failed to generate video with overlays")
                return None

            # Speed up the video to create time-lapse
            if output_filename is None:
                output_filename = f"{run_id}_timelapse_{speed_multiplier:.0f}x_overlays.mp4"

            output_path = self.video_output_dir / output_filename

            logger.info(f"Speeding up video by {speed_multiplier}x")

            clip = VideoFileClip(str(temp_output))
            sped_up_clip = clip.with_speed_scaled(speed_multiplier)

            sped_up_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio=False,
                preset='medium',
                logger=None
            )

            # Clean up
            clip.close()
            sped_up_clip.close()
            temp_output.unlink()  # Delete temporary file

            logger.info(f"Successfully created time-lapse with overlays: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to create time-lapse with overlays: {e}")
            import traceback
            traceback.print_exc()
            return None

