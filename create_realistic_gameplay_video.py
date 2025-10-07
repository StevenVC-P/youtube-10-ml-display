#!/usr/bin/env python3
"""
Create realistic-looking Atari Breakout gameplay for 1-hour video.
Simulates actual game mechanics and learning progression without requiring RL libraries.
"""

import sys
import time
import yaml
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import argparse
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent))


class BreakoutGameSimulator:
    """Realistic Breakout game simulator with learning progression."""
    
    def __init__(self, width=160, height=210, seed=42):
        self.width = width
        self.height = height
        np.random.seed(seed)
        
        # Game state
        self.reset_game()
        
        # Learning progression parameters
        self.skill_level = 0.0  # 0.0 to 1.0, improves over time
        self.episode_count = 0
        self.total_reward = 0.0
        
    def reset_game(self):
        """Reset game to initial state."""
        # Paddle
        self.paddle_x = self.width // 2
        self.paddle_width = 20
        self.paddle_y = self.height - 20
        
        # Ball
        self.ball_x = float(self.width // 2)
        self.ball_y = float(self.height - 40)
        self.ball_dx = np.random.choice([-2, 2])
        self.ball_dy = -3
        self.ball_radius = 2
        
        # Bricks
        self.bricks = []
        brick_rows = 6
        brick_cols = 8
        brick_width = self.width // brick_cols - 2
        brick_height = 8
        
        colors = [
            [255, 100, 100],  # Red
            [255, 150, 100],  # Orange
            [255, 255, 100],  # Yellow
            [100, 255, 100],  # Green
            [100, 100, 255],  # Blue
            [255, 100, 255],  # Purple
        ]
        
        for row in range(brick_rows):
            for col in range(brick_cols):
                brick = {
                    'x': col * (brick_width + 2) + 1,
                    'y': 50 + row * (brick_height + 2),
                    'width': brick_width,
                    'height': brick_height,
                    'color': colors[row % len(colors)],
                    'active': True
                }
                self.bricks.append(brick)
        
        # Game state
        self.lives = 3
        self.score = 0
        self.episode_reward = 0.0
        self.game_over = False
        self.episode_length = 0
    
    def get_intelligent_action(self):
        """Get action based on current skill level (simulates learning)."""
        # Random action for low skill
        if np.random.random() > self.skill_level:
            return np.random.choice([0, 1, 2, 3])  # NOOP, FIRE, LEFT, RIGHT
        
        # Intelligent action based on ball position
        ball_future_x = self.ball_x + self.ball_dx * 10  # Predict ball position
        
        if ball_future_x < self.paddle_x - 5:
            return 2  # LEFT
        elif ball_future_x > self.paddle_x + 5:
            return 3  # RIGHT
        else:
            return 0  # NOOP (stay in position)
    
    def step(self, action):
        """Step the game simulation."""
        self.episode_length += 1
        
        # Move paddle based on action
        if action == 2:  # LEFT
            self.paddle_x = max(self.paddle_width // 2, self.paddle_x - 4)
        elif action == 3:  # RIGHT
            self.paddle_x = min(self.width - self.paddle_width // 2, self.paddle_x + 4)
        
        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy
        
        # Ball collision with walls
        if self.ball_x <= self.ball_radius or self.ball_x >= self.width - self.ball_radius:
            self.ball_dx = -self.ball_dx
        
        if self.ball_y <= self.ball_radius:
            self.ball_dy = -self.ball_dy
        
        # Ball collision with paddle
        if (self.ball_y >= self.paddle_y - self.ball_radius and
            self.ball_y <= self.paddle_y + 5 and
            self.ball_x >= self.paddle_x - self.paddle_width // 2 and
            self.ball_x <= self.paddle_x + self.paddle_width // 2):
            
            self.ball_dy = -abs(self.ball_dy)
            # Add some angle based on where ball hits paddle
            hit_pos = (self.ball_x - self.paddle_x) / (self.paddle_width // 2)
            self.ball_dx += hit_pos * 1.5
            self.ball_dx = np.clip(self.ball_dx, -4, 4)
        
        # Ball collision with bricks
        reward = 0
        for brick in self.bricks:
            if (brick['active'] and
                self.ball_x >= brick['x'] and
                self.ball_x <= brick['x'] + brick['width'] and
                self.ball_y >= brick['y'] and
                self.ball_y <= brick['y'] + brick['height']):
                
                brick['active'] = False
                self.ball_dy = -self.ball_dy
                reward = 10
                self.score += 10
                break
        
        # Ball goes off bottom
        if self.ball_y >= self.height:
            self.lives -= 1
            reward = -10
            if self.lives <= 0:
                self.game_over = True
            else:
                # Reset ball
                self.ball_x = float(self.width // 2)
                self.ball_y = float(self.height - 40)
                self.ball_dx = np.random.choice([-2, 2])
                self.ball_dy = -3
        
        # Check if all bricks destroyed
        if all(not brick['active'] for brick in self.bricks):
            reward += 100
            self.game_over = True
        
        self.episode_reward += reward
        return reward, self.game_over
    
    def render(self):
        """Render the current game state."""
        # Create frame
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Background
        frame[:, :] = [20, 20, 40]  # Dark blue
        
        # Draw bricks
        for brick in self.bricks:
            if brick['active']:
                cv2.rectangle(
                    frame,
                    (brick['x'], brick['y']),
                    (brick['x'] + brick['width'], brick['y'] + brick['height']),
                    brick['color'],
                    -1
                )
        
        # Draw paddle
        paddle_left = self.paddle_x - self.paddle_width // 2
        paddle_right = self.paddle_x + self.paddle_width // 2
        cv2.rectangle(
            frame,
            (paddle_left, self.paddle_y),
            (paddle_right, self.paddle_y + 5),
            [255, 255, 255],
            -1
        )
        
        # Draw ball
        cv2.circle(
            frame,
            (int(self.ball_x), int(self.ball_y)),
            self.ball_radius,
            [255, 255, 0],
            -1
        )
        
        # Draw score
        score_text = f"Score: {self.score}"
        cv2.putText(frame, score_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
        
        # Draw lives
        lives_text = f"Lives: {self.lives}"
        cv2.putText(frame, lives_text, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
        
        return frame


class RealisticGameplayStreamer:
    """Creates realistic-looking gameplay video with learning progression."""
    
    def __init__(self, duration_minutes=60, grid_size=4):
        self.duration_minutes = duration_minutes
        self.grid_size = grid_size
        
        # Import modules directly
        import importlib.util
        
        # FFmpeg module
        spec1 = importlib.util.spec_from_file_location("ffmpeg_io", "stream/ffmpeg_io.py")
        self.ffmpeg_io = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(self.ffmpeg_io)
        
        # Grid composer module
        spec2 = importlib.util.spec_from_file_location("grid_composer", "stream/grid_composer.py")
        self.grid_composer_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(self.grid_composer_module)
        
        # Setup components
        self.grid_composer = self.grid_composer_module.GridComposer(
            grid_size=grid_size,
            pane_size=(240, 180),
            overlay_hud=True
        )
        
        # Create game simulators
        self.games = []
        for i in range(grid_size):
            game = BreakoutGameSimulator(seed=42 + i)
            self.games.append(game)
        
        print(f"🎮 Realistic Breakout gameplay streamer initialized")
        print(f"   Duration: {duration_minutes} minutes")
        print(f"   Grid: {grid_size} panes")
        print(f"   Resolution: {self.grid_composer.output_width}x{self.grid_composer.output_height}")
    
    def update_skill_progression(self, frame_count, total_frames):
        """Update skill level to simulate learning over time."""
        progress = frame_count / total_frames
        
        # Skill improves over time with some randomness
        base_skill = min(0.8, progress * 1.2)  # Cap at 80% skill
        
        for i, game in enumerate(self.games):
            # Each game learns at slightly different rate
            individual_progress = progress + (i * 0.1)
            game.skill_level = min(0.9, base_skill + np.sin(individual_progress * 3) * 0.1)
    
    def create_realistic_gameplay_video(self):
        """Create the realistic gameplay video."""
        print(f"🎬 Creating {self.duration_minutes}-minute realistic Breakout gameplay...")
        
        # Create output directory
        output_dir = Path("video/realistic_gameplay")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"realistic_breakout_{self.duration_minutes}min_{timestamp}.mp4"
        
        print(f"📹 Output: {output_path}")
        
        # Check FFmpeg availability
        ffmpeg_available = self.ffmpeg_io.test_ffmpeg_availability()
        if not ffmpeg_available:
            print("⚠️ FFmpeg not available, using mock mode")
        
        try:
            with self.ffmpeg_io.FFmpegWriter(
                output_path=output_path,
                width=self.grid_composer.output_width,
                height=self.grid_composer.output_height,
                fps=30,
                crf=23,
                preset="veryfast",
                mock_mode=not ffmpeg_available,
                verbose=True
            ) as writer:
                
                total_frames = self.duration_minutes * 60 * 30  # minutes * seconds * fps
                start_time = time.time()
                
                print(f"🎯 Target: {total_frames:,} frames ({self.duration_minutes} minutes)")
                
                for frame_count in range(total_frames):
                    # Update learning progression
                    self.update_skill_progression(frame_count, total_frames)
                    
                    # Step all games
                    frames = []
                    for game in self.games:
                        if game.game_over:
                            game.reset_game()
                            game.episode_count += 1
                        
                        # Get intelligent action based on skill level
                        action = game.get_intelligent_action()
                        reward, done = game.step(action)
                        
                        # Render game frame
                        frame = game.render()
                        frames.append(frame)
                    
                    # Create HUD info
                    hud_info = {
                        'global_step': frame_count,
                        'checkpoint_time': datetime.now().strftime("%H:%M:%S"),
                        'fps': 30.0,
                        'rewards': [game.episode_reward for game in self.games],
                        'episodes': [game.episode_count for game in self.games],
                        'scores': [game.score for game in self.games],
                        'skill_levels': [f"{game.skill_level:.2f}" for game in self.games]
                    }
                    
                    # Compose grid frame
                    composed_frame = self.grid_composer.compose_frame(frames, hud_info=hud_info)
                    
                    # Add title overlay for first few seconds
                    if frame_count < 150:  # First 5 seconds
                        title_text = f"Realistic Breakout Training - {self.duration_minutes} Minutes"
                        subtitle_text = "Simulated RL Agent Learning Progression"
                        
                        # Semi-transparent overlay
                        overlay = composed_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (self.grid_composer.output_width, 80), (0, 0, 0), -1)
                        cv2.addWeighted(composed_frame, 0.7, overlay, 0.3, 0, composed_frame)
                        
                        # Title text
                        cv2.putText(composed_frame, title_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(composed_frame, subtitle_text, (10, 55), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    # Write frame
                    success = writer.write_frame(composed_frame)
                    if not success:
                        print(f"❌ Failed to write frame {frame_count}")
                        break
                    
                    # Progress updates
                    if frame_count % (30 * 60) == 0:  # Every minute
                        elapsed_minutes = frame_count // (30 * 60)
                        progress = (frame_count / total_frames) * 100
                        print(f"📊 Progress: {elapsed_minutes}/{self.duration_minutes} min ({progress:.1f}%)")
                        
                        # Show game stats
                        for i, game in enumerate(self.games):
                            print(f"   Game {i}: Episodes={game.episode_count}, "
                                  f"Score={game.score}, Skill={game.skill_level:.2f}")
                
                # Final statistics
                elapsed_total = time.time() - start_time
                actual_fps = writer.frames_written / elapsed_total if elapsed_total > 0 else 0
                
                print(f"\n🏁 Video creation completed!")
                print(f"   Frames written: {writer.frames_written:,}")
                print(f"   Duration: {elapsed_total/60:.1f} minutes")
                print(f"   Average FPS: {actual_fps:.1f}")
                
                if ffmpeg_available and output_path.exists():
                    size_mb = output_path.stat().st_size / 1024 / 1024
                    print(f"   File size: {size_mb:.1f} MB")
                    print(f"   Video saved: {output_path}")
                
                return True
                
        except Exception as e:
            print(f"❌ Video creation failed: {e}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Realistic Atari gameplay video creation")
    parser.add_argument("--duration", type=int, default=60, help="Duration in minutes")
    parser.add_argument("--grid", type=int, default=4, choices=[1, 4, 9], help="Grid size")
    args = parser.parse_args()
    
    print("🚀 Realistic Atari Breakout Gameplay Video Creator")
    print("=" * 60)
    
    try:
        streamer = RealisticGameplayStreamer(
            duration_minutes=args.duration,
            grid_size=args.grid
        )
        
        success = streamer.create_realistic_gameplay_video()
        
        if success:
            print("🎉 Realistic gameplay video created successfully!")
        else:
            print("❌ Realistic gameplay video creation failed!")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"💥 Failed to create video: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
