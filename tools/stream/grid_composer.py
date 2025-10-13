#!/usr/bin/env python3
"""
Single-screen frame composer with HUD overlay for Breakout training.
Simplified for single environment training with real emulator.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


class SingleScreenComposer:
    """Composes single environment frame with HUD overlay for training display."""

    def __init__(
        self,
        frame_size: Tuple[int, int] = (480, 360),
        overlay_hud: bool = True,
        background_color: Tuple[int, int, int] = (32, 32, 32)
    ):
        """
        Initialize single screen composer.

        Args:
            frame_size: Size of the output frame (width, height)
            overlay_hud: Whether to overlay HUD information
            background_color: Background color for empty areas (BGR)
        """
        self.frame_width, self.frame_height = frame_size
        self.overlay_hud = overlay_hud
        self.background_color = background_color

        # Calculate output dimensions (same as frame size for single screen)
        self.output_width = self.frame_width
        self.output_height = self.frame_height

        # Add space for HUD if enabled
        if self.overlay_hud:
            self.hud_height = 80
            self.output_height += self.hud_height
        else:
            self.hud_height = 0
    
    def compose_frame(
        self,
        frame: np.ndarray,
        hud_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Compose single frame with optional HUD overlay.

        Args:
            frame: Single environment frame
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

        if frame is not None:
            # Calculate display area (excluding HUD space)
            display_height = self.frame_height
            if self.overlay_hud:
                display_height = self.output_height - self.hud_height

            # Resize frame to fit display area
            if frame.shape[:2] != (display_height, self.frame_width):
                frame = cv2.resize(frame, (self.frame_width, display_height))

            # Ensure frame is BGR format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assume RGB, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif len(frame.shape) == 2:
                # Grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Place frame on canvas
            canvas[0:display_height, 0:self.frame_width] = frame

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


def test_single_screen_composer():
    """Test the single screen composer functionality."""
    print("🎨 Testing Single Screen Composer")

    # Test different frame sizes
    frame_configs = [
        ((480, 360), "Standard resolution"),
        ((640, 480), "VGA resolution"),
        ((320, 240), "Small resolution"),
    ]

    for frame_size, description in frame_configs:
        print(f"  Testing {description}...")

        composer = SingleScreenComposer(
            frame_size=frame_size,
            overlay_hud=True
        )

        # Create test frame (Atari-like)
        frame = np.zeros((84, 84, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel
        frame[20:60, 20:60, 1] = 255  # Green square in middle

        # Test composition with HUD
        hud_data = {
            'global_step': 123456,
            'checkpoint_time': '12:00:00',
            'fps': 29.5,
            'rewards': 42.5
        }

        composed = composer.compose_frame(frame, hud_info=hud_data)

        expected_height = frame_size[1] + 80  # +80 for HUD
        expected_width = frame_size[0]

        print(f"    Output shape: {composed.shape}")
        print(f"    Expected: ({expected_height}, {expected_width}, 3)")

        if composed.shape != (expected_height, expected_width, 3):
            print(f"    ❌ Shape mismatch!")
            return False

        print(f"    ✅ {description} working correctly")

    return True


if __name__ == "__main__":
    success = test_single_screen_composer()
    if success:
        print("🎉 Single screen composer tests passed!")
    else:
        print("❌ Single screen composer tests failed!")
