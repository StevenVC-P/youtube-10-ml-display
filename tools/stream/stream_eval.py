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
from stream.grid_composer import SingleScreenComposer


# Removed duplicate GridComposer class - now using SingleScreenComposer from grid_composer.py


class ContinuousEvaluator:
    """Main continuous evaluation streamer."""
    
    def __init__(self, config: Dict[str, Any], args: argparse.Namespace):
        """Initialize continuous evaluator."""
        self.config = config
        self.args = args
        
        # Stream configuration (simplified for single screen)
        self.frame_size = tuple(config['stream']['pane_size'])
        self.fps = args.fps
        self.checkpoint_poll_sec = config['stream']['checkpoint_poll_sec']
        self.overlay_hud = config['stream']['overlay_hud']

        # Output configuration
        self.save_mode = args.save_mode
        self.segment_seconds = args.segment_seconds if args.save_mode == "segments" else None
        self.output_dir = Path(config['paths']['videos_parts'])
        self.output_basename = config['stream']['output_basename']

        # FFmpeg configuration
        self.ffmpeg_path = config['paths'].get('ffmpeg_path', 'ffmpeg')
        self.ffmpeg_writer = None
        self.preset = config['stream']['preset']
        self.crf = config['stream']['crf']

        # Evaluation state (single environment)
        self.model = None
        self.environment = None
        self.current_checkpoint = None
        self.last_checkpoint_check = 0
        self.global_step = 0
        self.episode_rewards = []

        # Single screen composer
        self.screen_composer = SingleScreenComposer(
            frame_size=self.frame_size,
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
                print(f">> Created {len(self.environments)} evaluation environments")

            return True

        except Exception as e:
            if self.args.verbose >= 1:
                print(f"ERROR: Failed to setup environments: {e}")
            return False
    
    def load_latest_checkpoint(self) -> bool:
        """Load the latest checkpoint if available."""
        try:
            checkpoint_dir = Path(self.config['paths']['models'])
            latest_checkpoint = checkpoint_dir / "latest.zip"
            
            if not latest_checkpoint.exists():
                if self.args.verbose >= 2:
                    print(f"WARNING: No checkpoint found at {latest_checkpoint}")
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
                print(f">> Loaded checkpoint from {checkpoint_time}")

            return True

        except Exception as e:
            if self.args.verbose >= 1:
                print(f"ERROR: Failed to load checkpoint: {e}")
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
                    print(f"WARNING: Error in environment {i}: {e}")
                
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
                    print(f"WARNING: Checkpoint monitor error: {e}")
                time.sleep(5)  # Wait longer on error

    def run_streaming(self) -> bool:
        """Main streaming loop."""
        if self.args.verbose >= 1:
            print(f">> Starting continuous evaluation stream:")
            print(f"  - Grid: {self.grid_size} panes ({self.grid_composer.grid_rows}x{self.grid_composer.grid_cols})")
            print(f"  - Resolution: {self.grid_composer.output_width}x{self.grid_composer.output_height}")
            print(f"  - FPS: {self.fps}")
            print(f"  - Save mode: {self.save_mode}")
            if self.segment_seconds:
                print(f"  - Segment duration: {self.segment_seconds}s")

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
            ffmpeg_available = test_ffmpeg_availability(self.ffmpeg_path)
            if not ffmpeg_available and self.args.verbose >= 1:
                print("WARNING: FFmpeg not available, using mock mode for testing")

            with FFmpegWriter(
                output_path=output_path,
                width=self.grid_composer.output_width,
                height=self.grid_composer.output_height,
                fps=self.fps,
                crf=self.crf,
                preset=self.preset,
                segment_time=segment_time,
                ffmpeg_path=self.ffmpeg_path,
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
                                print("WARNING: Failed to write frame, stopping...")
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
                                print(f">> Streaming: {self.frames_streamed} frames, {elapsed:.0f}s, {actual_fps:.1f} FPS")

                    except KeyboardInterrupt:
                        if self.args.verbose >= 1:
                            print("\nWARNING: Streaming interrupted by user")
                        break
                    except Exception as e:
                        if self.args.verbose >= 1:
                            print(f"ERROR: Streaming error: {e}")
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
                        print(f"\n>> Streaming completed:")
                        print(f"  - Total frames: {self.frames_streamed}")
                        print(f"  - Duration: {elapsed:.1f}s")
                        print(f"  - Average FPS: {actual_fps:.1f}")
                        print(f"  - Episodes completed: {len(self.episode_rewards)}")
                        if self.episode_rewards:
                            print(f"  - Mean reward: {np.mean(self.episode_rewards):.2f}")

                return True

        except Exception as e:
            if self.args.verbose >= 1:
                print(f"ERROR: Streaming failed: {e}")
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
                print(f"ERROR: Evaluation failed: {e}")
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
            print("WARNING: FFmpeg not found. Running in mock mode for testing.")
            print("   For production use, please install FFmpeg and ensure it's in PATH.")

    try:
        # Load configuration
        config = load_config(args.config)
        if args.verbose >= 2:
            print(f">> Configuration loaded from: {args.config}")

        # Create evaluator
        evaluator = ContinuousEvaluator(config, args)

        # Run evaluation
        return evaluator.run()

    except Exception as e:
        if args.verbose >= 1:
            print(f"ERROR: Failed to start evaluator: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
