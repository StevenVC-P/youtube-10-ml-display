#!/usr/bin/env python3
"""
Continuous evaluation streamer for trained RL models.
Maintains multiple evaluation environments, composes grid layout with HUD overlay,
and streams to FFmpeg for continuous video recording.
"""

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import cv2
import torch
from stable_baselines3 import PPO, DQN

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from conf.config import load_config
from envs.make_env import make_eval_env
from envs.atari_wrappers import make_atari_env
from agents.algo_factory import print_system_info
from stream.ffmpeg_io import FFmpegWriter, test_ffmpeg_availability


class GridComposer:
    """Composes multiple environment frames into a grid layout with HUD overlay."""
    
    def __init__(
        self,
        grid_size: int,
        pane_size: Tuple[int, int],
        overlay_hud: bool = True,
        background_color: Tuple[int, int, int] = (32, 32, 32)
    ):
        """
        Initialize grid composer.
        
        Args:
            grid_size: Number of panes (1, 4, or 9)
            pane_size: Size of each pane (width, height)
            overlay_hud: Whether to overlay HUD information
            background_color: Background color for empty areas (BGR)
        """
        self.grid_size = grid_size
        self.pane_width, self.pane_height = pane_size
        self.overlay_hud = overlay_hud
        self.background_color = background_color
        
        # Calculate grid layout
        if grid_size == 1:
            self.grid_rows, self.grid_cols = 1, 1
        elif grid_size == 4:
            self.grid_rows, self.grid_cols = 2, 2
        elif grid_size == 9:
            self.grid_rows, self.grid_cols = 3, 3
        else:
            raise ValueError(f"Unsupported grid size: {grid_size}")
        
        # Calculate output dimensions
        self.output_width = self.grid_cols * self.pane_width
        self.output_height = self.grid_rows * self.pane_height
        
        # Add space for HUD if enabled
        if self.overlay_hud:
            self.hud_height = 80
            self.output_height += self.hud_height
        else:
            self.hud_height = 0
    
    def compose_frame(
        self,
        frames: List[np.ndarray],
        hud_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Compose multiple frames into a grid with optional HUD.
        
        Args:
            frames: List of frame arrays (can be fewer than grid_size)
            hud_info: Dictionary with HUD information
            
        Returns:
            Composed frame as numpy array
        """
        # Create output canvas
        canvas = np.full(
            (self.output_height, self.output_width, 3),
            self.background_color,
            dtype=np.uint8
        )
        
        # Place frames in grid
        for i, frame in enumerate(frames[:self.grid_size]):
            if frame is None:
                continue
            
            # Calculate grid position
            row = i // self.grid_cols
            col = i % self.grid_cols
            
            # Calculate canvas position
            y_start = row * self.pane_height
            y_end = y_start + self.pane_height
            x_start = col * self.pane_width
            x_end = x_start + self.pane_width
            
            # Resize frame to pane size if needed
            if frame.shape[:2] != (self.pane_height, self.pane_width):
                frame = cv2.resize(frame, (self.pane_width, self.pane_height))
            
            # Ensure frame is BGR format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assume RGB, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 2:
                # Grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Place frame on canvas
            canvas[y_start:y_end, x_start:x_end] = frame
        
        # Add HUD overlay
        if self.overlay_hud and hud_info:
            self._draw_hud(canvas, hud_info)
        
        return canvas
    
    def _draw_hud(self, canvas: np.ndarray, hud_info: Dict[str, Any]):
        """Draw HUD overlay on the canvas."""
        if not self.overlay_hud:
            return
        
        # HUD area (bottom of canvas)
        hud_y_start = self.output_height - self.hud_height
        
        # Draw HUD background
        cv2.rectangle(
            canvas,
            (0, hud_y_start),
            (self.output_width, self.output_height),
            (16, 16, 16),  # Dark background
            -1
        )
        
        # Draw HUD text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)  # White text
        thickness = 1
        
        # Line 1: Basic info
        y_pos = hud_y_start + 25
        text_items = []
        
        if 'global_step' in hud_info:
            text_items.append(f"Step: {hud_info['global_step']:,}")
        
        if 'checkpoint_time' in hud_info:
            text_items.append(f"Checkpoint: {hud_info['checkpoint_time']}")
        
        if 'fps' in hud_info:
            text_items.append(f"FPS: {hud_info['fps']:.1f}")
        
        line1_text = " | ".join(text_items)
        cv2.putText(canvas, line1_text, (10, y_pos), font, font_scale, color, thickness)
        
        # Line 2: Rewards
        y_pos += 30
        if 'rewards' in hud_info and hud_info['rewards']:
            rewards = hud_info['rewards']
            if isinstance(rewards, list):
                reward_text = f"Rewards: {[f'{r:.1f}' for r in rewards]}"
            else:
                reward_text = f"Reward: {rewards:.1f}"
            cv2.putText(canvas, reward_text, (10, y_pos), font, font_scale, color, thickness)


class ContinuousEvaluator:
    """Main continuous evaluation streamer."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        """Initialize continuous evaluator."""
        self.config = config
        self.args = args
        
        # Stream configuration
        self.grid_size = args.grid
        self.pane_size = tuple(config['stream']['pane_size'])
        self.fps = args.fps
        self.checkpoint_poll_sec = config['stream']['checkpoint_poll_sec']
        self.overlay_hud = config['stream']['overlay_hud']
        
        # Output configuration
        self.save_mode = args.save_mode
        self.segment_seconds = args.segment_seconds if args.save_mode == "segments" else None
        self.output_dir = Path(config['paths']['videos_parts'])
        self.output_basename = config['stream']['output_basename']
        
        # FFmpeg configuration
        self.ffmpeg_writer = None
        self.preset = config['stream']['preset']
        self.crf = config['stream']['crf']
        
        # Evaluation state
        self.model = None
        self.environments = []
        self.current_checkpoint = None
        self.last_checkpoint_check = 0
        self.global_step = 0
        self.episode_rewards = []
        
        # Grid composer
        self.grid_composer = GridComposer(
            grid_size=self.grid_size,
            pane_size=self.pane_size,
            overlay_hud=self.overlay_hud
        )
        
        # Threading
        self.running = False
        self.checkpoint_thread = None
        
        # Statistics
        self.frames_streamed = 0
        self.start_time = None
    
    def setup_environments(self) -> bool:
        """Setup evaluation environments."""
        try:
            self.environments = []
            
            for i in range(self.grid_size):
                # Create environment with render mode for streaming
                env = make_atari_env(
                    env_id=self.config['game']['env_id'],
                    config=self.config,
                    seed=42 + i,
                    render_mode="rgb_array"  # Required for streaming
                )
                self.environments.append({
                    'env': env,
                    'obs': None,
                    'episode_reward': 0.0,
                    'episode_length': 0,
                    'done': True  # Start with reset needed
                })
            
            if self.args.verbose >= 1:
                print(f"✅ Created {len(self.environments)} evaluation environments")
            
            return True
            
        except Exception as e:
            if self.args.verbose >= 1:
                print(f"❌ Failed to setup environments: {e}")
            return False
    
    def load_latest_checkpoint(self) -> bool:
        """Load the latest checkpoint if available."""
        try:
            checkpoint_dir = Path(self.config['paths']['models'])
            latest_checkpoint = checkpoint_dir / "latest.zip"
            
            if not latest_checkpoint.exists():
                if self.args.verbose >= 2:
                    print(f"⚠️ No checkpoint found at {latest_checkpoint}")
                return False
            
            # Check if checkpoint is newer
            checkpoint_mtime = latest_checkpoint.stat().st_mtime
            if self.current_checkpoint and checkpoint_mtime <= self.current_checkpoint:
                return True  # No update needed
            
            # Load model
            algo_type = self.config['train']['algo'].lower()
            
            if algo_type == "ppo":
                self.model = PPO.load(str(latest_checkpoint))
            elif algo_type == "dqn":
                self.model = DQN.load(str(latest_checkpoint))
            else:
                raise ValueError(f"Unsupported algorithm: {algo_type}")
            
            self.current_checkpoint = checkpoint_mtime
            
            if self.args.verbose >= 1:
                checkpoint_time = datetime.fromtimestamp(checkpoint_mtime).strftime("%H:%M:%S")
                print(f"🔄 Loaded checkpoint from {checkpoint_time}")
            
            return True
            
        except Exception as e:
            if self.args.verbose >= 1:
                print(f"❌ Failed to load checkpoint: {e}")
            return False
    
    def step_environments(self) -> List[np.ndarray]:
        """Step all environments and return rendered frames."""
        frames = []
        
        for i, env_data in enumerate(self.environments):
            env = env_data['env']
            
            try:
                # Reset if needed
                if env_data['done']:
                    if env_data['episode_reward'] > 0:  # Completed episode
                        self.episode_rewards.append(env_data['episode_reward'])
                        if len(self.episode_rewards) > 100:  # Keep last 100
                            self.episode_rewards.pop(0)
                    
                    env_data['obs'], _ = env.reset()
                    env_data['episode_reward'] = 0.0
                    env_data['episode_length'] = 0
                    env_data['done'] = False
                
                # Get action from model
                if self.model and env_data['obs'] is not None:
                    action, _ = self.model.predict(env_data['obs'], deterministic=True)
                else:
                    action = env.action_space.sample()  # Random action if no model
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                env_data['obs'] = obs
                env_data['episode_reward'] += reward
                env_data['episode_length'] += 1
                env_data['done'] = terminated or truncated
                
                # Render frame
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                else:
                    # Create placeholder frame
                    placeholder = np.zeros((self.pane_size[1], self.pane_size[0], 3), dtype=np.uint8)
                    frames.append(placeholder)
                
            except Exception as e:
                if self.args.verbose >= 2:
                    print(f"⚠️ Error in environment {i}: {e}")
                
                # Create error placeholder
                placeholder = np.full((self.pane_size[1], self.pane_size[0], 3), (64, 0, 0), dtype=np.uint8)
                frames.append(placeholder)
        
        return frames

    def checkpoint_monitor_thread(self):
        """Background thread to monitor for checkpoint updates."""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_checkpoint_check >= self.checkpoint_poll_sec:
                    self.load_latest_checkpoint()
                    self.last_checkpoint_check = current_time

                time.sleep(1)  # Check every second

            except Exception as e:
                if self.args.verbose >= 2:
                    print(f"⚠️ Checkpoint monitor error: {e}")
                time.sleep(5)  # Wait longer on error

    def run_streaming(self) -> bool:
        """Main streaming loop."""
        if self.args.verbose >= 1:
            print(f"🎬 Starting continuous evaluation stream:")
            print(f"  • Grid: {self.grid_size} panes ({self.grid_composer.grid_rows}x{self.grid_composer.grid_cols})")
            print(f"  • Resolution: {self.grid_composer.output_width}x{self.grid_composer.output_height}")
            print(f"  • FPS: {self.fps}")
            print(f"  • Save mode: {self.save_mode}")
            if self.segment_seconds:
                print(f"  • Segment duration: {self.segment_seconds}s")

        # Setup FFmpeg writer
        if self.save_mode == "segments":
            output_path = self.output_dir
            segment_time = self.segment_seconds
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"{self.output_basename}_{timestamp}.mp4"
            segment_time = None

        try:
            # Check if FFmpeg is available, use mock mode if not
            ffmpeg_available = test_ffmpeg_availability()
            if not ffmpeg_available and self.args.verbose >= 1:
                print("⚠️ FFmpeg not available, using mock mode for testing")

            with FFmpegWriter(
                output_path=output_path,
                width=self.grid_composer.output_width,
                height=self.grid_composer.output_height,
                fps=self.fps,
                crf=self.crf,
                preset=self.preset,
                segment_time=segment_time,
                verbose=self.args.verbose >= 1,
                mock_mode=not ffmpeg_available
            ) as writer:

                self.ffmpeg_writer = writer
                self.running = True
                self.start_time = time.time()

                # Start checkpoint monitoring thread
                self.checkpoint_thread = threading.Thread(
                    target=self.checkpoint_monitor_thread,
                    daemon=True
                )
                self.checkpoint_thread.start()

                # Load initial checkpoint
                self.load_latest_checkpoint()

                # Main streaming loop
                frame_interval = 1.0 / self.fps
                next_frame_time = time.time()

                while self.running:
                    loop_start = time.time()

                    try:
                        # Step environments and get frames
                        env_frames = self.step_environments()

                        # Prepare HUD info
                        hud_info = {
                            'global_step': self.global_step,
                            'checkpoint_time': datetime.fromtimestamp(self.current_checkpoint).strftime("%H:%M:%S") if self.current_checkpoint else "None",
                            'fps': self.frames_streamed / (time.time() - self.start_time) if self.start_time else 0,
                            'rewards': [env['episode_reward'] for env in self.environments]
                        }

                        # Compose grid frame
                        composed_frame = self.grid_composer.compose_frame(env_frames, hud_info)

                        # Write frame to FFmpeg
                        if writer.write_frame(composed_frame):
                            self.frames_streamed += 1
                            self.global_step += 1
                        else:
                            if self.args.verbose >= 1:
                                print("⚠️ Failed to write frame, stopping...")
                            break

                        # Frame rate control
                        next_frame_time += frame_interval
                        sleep_time = next_frame_time - time.time()
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        elif sleep_time < -frame_interval:
                            # We're falling behind, reset timing
                            next_frame_time = time.time()

                        # Periodic status update
                        if self.frames_streamed % (self.fps * 30) == 0:  # Every 30 seconds
                            elapsed = time.time() - self.start_time
                            actual_fps = self.frames_streamed / elapsed
                            if self.args.verbose >= 1:
                                print(f"📊 Streaming: {self.frames_streamed} frames, {elapsed:.0f}s, {actual_fps:.1f} FPS")

                    except KeyboardInterrupt:
                        if self.args.verbose >= 1:
                            print("\n⚠️ Streaming interrupted by user")
                        break
                    except Exception as e:
                        if self.args.verbose >= 1:
                            print(f"❌ Streaming error: {e}")
                        if self.args.verbose >= 2:
                            import traceback
                            traceback.print_exc()
                        break

                self.running = False

                # Final statistics
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    actual_fps = self.frames_streamed / elapsed if elapsed > 0 else 0

                    if self.args.verbose >= 1:
                        print(f"\n📈 Streaming completed:")
                        print(f"  • Total frames: {self.frames_streamed}")
                        print(f"  • Duration: {elapsed:.1f}s")
                        print(f"  • Average FPS: {actual_fps:.1f}")
                        print(f"  • Episodes completed: {len(self.episode_rewards)}")
                        if self.episode_rewards:
                            print(f"  • Mean reward: {np.mean(self.episode_rewards):.2f}")

                return True

        except Exception as e:
            if self.args.verbose >= 1:
                print(f"❌ Streaming failed: {e}")
            return False
        finally:
            self.running = False

            # Cleanup environments
            for env_data in self.environments:
                try:
                    env_data['env'].close()
                except:
                    pass

    def run(self) -> int:
        """Main entry point."""
        try:
            # Setup environments
            if not self.setup_environments():
                return 1

            # Run streaming
            if not self.run_streaming():
                return 1

            return 0

        except Exception as e:
            if self.args.verbose >= 1:
                print(f"❌ Evaluation failed: {e}")
            return 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Continuous evaluation streamer for RL models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--grid",
        type=int,
        choices=[1, 4, 9],
        default=4,
        help="Grid size (1, 4, or 9 panes)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Streaming frame rate"
    )

    parser.add_argument(
        "--save-mode",
        type=str,
        choices=["single", "segments"],
        default="segments",
        help="Save mode: single file or segments"
    )

    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=1800,
        help="Segment duration in seconds (for segments mode)"
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=silent, 1=info, 2=debug)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Print system info
    if args.verbose >= 1:
        print_system_info()

    # Test FFmpeg availability (warn but don't fail)
    if not test_ffmpeg_availability():
        if args.verbose >= 1:
            print("⚠️ FFmpeg not found. Running in mock mode for testing.")
            print("   For production use, please install FFmpeg and ensure it's in PATH.")

    try:
        # Load configuration
        config = load_config(args.config)
        if args.verbose >= 2:
            print(f"📋 Configuration loaded from: {args.config}")

        # Create evaluator
        evaluator = ContinuousEvaluator(config, args)

        # Run evaluation
        return evaluator.run()

    except Exception as e:
        if args.verbose >= 1:
            print(f"❌ Failed to start evaluator: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
