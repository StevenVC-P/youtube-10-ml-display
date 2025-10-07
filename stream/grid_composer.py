#!/usr/bin/env python3
"""
Grid composer for multiple environment frames with HUD overlay.
Lightweight module without heavy ML library dependencies.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2


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


def test_grid_composer():
    """Test the grid composer functionality."""
    print("🎨 Testing Grid Composer")
    
    # Test different grid sizes
    grid_configs = [
        (1, (240, 180), "Single pane"),
        (4, (120, 90), "2x2 grid"),
        (9, (80, 60), "3x3 grid"),
    ]
    
    for grid_size, pane_size, description in grid_configs:
        print(f"  Testing {description}...")
        
        composer = GridComposer(
            grid_size=grid_size,
            pane_size=pane_size,
            overlay_hud=True
        )
        
        # Create test frames
        frames = []
        for i in range(grid_size):
            # Create colorful test frame
            frame = np.zeros((84, 84, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Different color per frame
            frames.append(frame)
        
        # Test composition with HUD
        hud_data = {
            'global_step': 123456,
            'checkpoint_time': '12:00:00',
            'fps': 29.5,
            'rewards': [float(i * 10) for i in range(grid_size)]
        }
        
        composed = composer.compose_frame(frames, hud_info=hud_data)
        
        expected_height = pane_size[1] * composer.grid_rows + 80  # +80 for HUD
        expected_width = pane_size[0] * composer.grid_cols
        
        print(f"    Output shape: {composed.shape}")
        print(f"    Expected: ({expected_height}, {expected_width}, 3)")
        
        if composed.shape != (expected_height, expected_width, 3):
            print(f"    ❌ Shape mismatch!")
            return False
        
        print(f"    ✅ {description} working correctly")
    
    return True


if __name__ == "__main__":
    success = test_grid_composer()
    if success:
        print("🎉 Grid composer tests passed!")
    else:
        print("❌ Grid composer tests failed!")
