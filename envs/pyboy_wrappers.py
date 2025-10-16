"""
PyBoy Gymnasium wrappers for authentic Gameboy emulation.
Provides pixel-perfect retro gaming experience for ML training.
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PyBoyWrapper(gym.Env):
    """
    Gymnasium wrapper for PyBoy authentic Gameboy emulation.
    Provides authentic retro gaming experience with pixel-perfect graphics.
    """
    
    def __init__(self, rom_path: str, config: Dict[str, Any], render_mode: Optional[str] = None, **kwargs):
        super().__init__()

        self.rom_path = rom_path
        self.config = config
        self.render_mode = render_mode
        
        # Check if ROM file exists
        if not Path(rom_path).exists():
            raise FileNotFoundError(
                f"ROM file not found: {rom_path}\n"
                f"Please place legal ROM files in the appropriate directory.\n"
                f"Supported formats: .gb, .gbc"
            )
        
        # Import PyBoy
        try:
            from pyboy import PyBoy
            self.PyBoy = PyBoy
        except ImportError:
            raise ImportError(
                "PyBoy not installed. Install with: pip install pyboy"
            )
        
        # Initialize PyBoy emulator
        self.pyboy = None
        self.screen = None
        self.cartridge_title = "Unknown"
        self._setup_emulator()

        # Skip auto-start - let model learn from title screen
        print("🎮 Starting from title screen - model will learn menu navigation")

        # Just wait for boot to complete
        for _ in range(500):
            self.pyboy.tick()

        # Define action space (Gameboy buttons)
        # 0: No action, 1: A, 2: B, 3: Start, 4: Select, 5: Up, 6: Down, 7: Left, 8: Right
        self.action_space = gym.spaces.Discrete(9)
        
        # Define observation space (160x144 grayscale screen)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(144, 160, 1), 
            dtype=np.uint8
        )
        
        # Action mapping
        self.action_map = {
            0: [],  # No action
            1: ['a'],
            2: ['b'], 
            3: ['start'],
            4: ['select'],
            5: ['up'],
            6: ['down'],
            7: ['left'],
            8: ['right']
        }
        
        logger.info(f"PyBoy wrapper initialized for ROM: {rom_path}")

    def _auto_start_game(self):
        """Use the CORRECT Game Boy Tetris sequence: Start → A"""
        print("🎮 Auto-starting Tetris with CORRECT sequence...")

        # Step 1: Wait for full boot sequence
        print("   ⏳ Waiting for Tetris to fully boot...")
        for i in range(500):
            self.pyboy.tick()
            if i % 100 == 0:
                current_screen = self.pyboy.screen.ndarray.copy()
                unique_vals = len(np.unique(current_screen))
                print(f"   Boot progress {i}: unique values = {unique_vals}")

        # Step 2: Press Start (gets to "1 player" menu)
        print("   🎯 CORRECT Step 1: Start → '1 player' menu")
        self.pyboy.button_press("start")
        for _ in range(30):
            self.pyboy.tick()
        self.pyboy.button_release("start")

        # Wait for menu to fully appear
        for _ in range(120):
            self.pyboy.tick()

        # Step 3: Press A (selects "GAME A-TYPE")
        print("   🎯 CORRECT Step 2: A → 'GAME A-TYPE'")
        self.pyboy.button_press("a")
        for _ in range(30):
            self.pyboy.tick()
        self.pyboy.button_release("a")

        # Wait for game to start (longer wait)
        print("   ⏳ Waiting for game to fully start...")
        for _ in range(600):  # Much longer wait for game to initialize
            self.pyboy.tick()

        # Simple verification
        print("   🔍 Verifying game started correctly...")

        # Final wait for game to stabilize
        for _ in range(300):  # Longer wait
            self.pyboy.tick()

        # Force some initial gameplay moves to ensure we're deep in the game
        print("   🎯 Forcing initial gameplay moves...")
        for move_cycle in range(5):  # More cycles
            # Soft drop to advance game state
            self.pyboy.button_press("down")
            for _ in range(30):
                self.pyboy.tick()
            self.pyboy.button_release("down")

            # Wait for piece to settle
            for _ in range(90):
                self.pyboy.tick()

            # Move left
            self.pyboy.button_press("left")
            for _ in range(15):
                self.pyboy.tick()
            self.pyboy.button_release("left")

            # Wait
            for _ in range(60):
                self.pyboy.tick()

        # Final verification using our improved detection
        current_screen = self.pyboy.screen.ndarray.copy()
        is_actually_in_game = self._is_actually_in_game(current_screen)

        # Detailed analysis for debugging
        title_area = current_screen[50:100, 50:110] if current_screen.shape[0] > 100 else current_screen[20:60, 20:80]
        title_pixels = np.sum(title_area > 128)

        game_area = current_screen[20:140, 40:120] if current_screen.shape[0] > 140 else current_screen[20:120, 40:120]
        game_pixels = np.sum(game_area > 128)

        ui_area = current_screen[20:140, 120:160] if current_screen.shape[1] > 160 else current_screen[20:120, 100:140]
        ui_pixels = np.sum(ui_area > 128)

        left_border = current_screen[20:140, 39:41] if current_screen.shape[0] > 140 else current_screen[20:120, 39:41]
        right_border = current_screen[20:140, 119:121] if current_screen.shape[0] > 140 else current_screen[20:120, 119:121]
        left_border_pixels = np.sum(left_border > 128)
        right_border_pixels = np.sum(right_border > 128)

        print(f"   📊 Detailed analysis:")
        print(f"      title_pixels={title_pixels}, game_pixels={game_pixels}, ui_pixels={ui_pixels}")
        print(f"      left_border={left_border_pixels}, right_border={right_border_pixels}")
        print(f"      is_actually_in_game={is_actually_in_game}")

        if is_actually_in_game:
            print("   ✅ Successfully started Tetris game!")
        else:
            print("   ❌ FAILED to start game - still in menus!")
            print("   🔧 Auto-start sequence needs improvement")

        print("🎮 Auto-start complete - ML model can now learn Tetris gameplay!")

    def _is_actually_in_game(self, screen: np.ndarray) -> bool:
        """More strict detection of actual gameplay vs menus."""

        # Check for obvious title screen indicators
        title_area = screen[50:100, 50:110] if screen.shape[0] > 100 else screen[20:60, 20:80]
        title_pixels = np.sum(title_area > 128)

        # If we see the TETRIS logo, we're definitely in menus
        if title_pixels > 8000:
            return False

        # Check for "1 PLAYER" / "2 PLAYER" text
        player_region = screen[120:160, 20:140] if screen.shape[0] > 160 else screen[80:120, 20:100]
        player_pixels = np.sum(player_region > 128)

        if player_pixels > 5000:
            return False

        # Check for actual Tetris game board structure
        # The game board should have specific patterns
        game_area = screen[20:140, 40:120] if screen.shape[0] > 140 else screen[20:120, 40:120]

        # Look for the characteristic Tetris board borders/structure
        # In actual gameplay, there should be consistent vertical lines (board edges)
        left_border = screen[20:140, 39:41] if screen.shape[0] > 140 else screen[20:120, 39:41]
        right_border = screen[20:140, 119:121] if screen.shape[0] > 140 else screen[20:120, 119:121]

        left_border_pixels = np.sum(left_border > 128)
        right_border_pixels = np.sum(right_border > 128)

        # In actual gameplay, we should see board borders
        has_board_structure = left_border_pixels > 50 and right_border_pixels > 50

        # Check for score/UI area (should be consistently present in gameplay)
        ui_area = screen[20:140, 120:160] if screen.shape[1] > 160 else screen[20:120, 100:140]
        ui_pixels = np.sum(ui_area > 128)

        # We're in game if we have board structure AND UI elements
        return has_board_structure and ui_pixels > 2000

    def _setup_emulator(self):
        """Initialize the PyBoy emulator."""
        try:
            # Create PyBoy instance in null window mode for training (updated API)
            self.pyboy = self.PyBoy(
                self.rom_path,
                window="null",
                debug=False
            )

            # Get screen interface
            self.screen = self.pyboy.screen

            # Get cartridge title (updated API)
            try:
                self.cartridge_title = self.pyboy.cartridge_title
            except AttributeError:
                self.cartridge_title = "Unknown Game"

            logger.info(f"Emulator initialized for: {self.cartridge_title}")

        except Exception as e:
            logger.error(f"Failed to initialize PyBoy emulator: {e}")
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset any internal state but keep the emulator running
        # (Emulator and auto-start are handled in __init__)
        self.steps_without_action = 0
        if hasattr(self, '_last_screen'):
            delattr(self, '_last_screen')
        if hasattr(self, '_last_menu_pixels'):
            delattr(self, '_last_menu_pixels')

        observation = self._get_observation()
        info = self._get_info()

        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Convert numpy array to int if needed
        if hasattr(action, 'item'):
            action = action.item()
        action = int(action)

        # Release all buttons first
        self.pyboy.button_release("a")
        self.pyboy.button_release("b")
        self.pyboy.button_release("start")
        self.pyboy.button_release("select")
        self.pyboy.button_release("up")
        self.pyboy.button_release("down")
        self.pyboy.button_release("left")
        self.pyboy.button_release("right")

        # Press the selected button(s)
        if action in self.action_map:
            buttons = self.action_map[action]
            if buttons:
                # Normal button press
                for button in buttons:
                    self.pyboy.button_press(button)

                # Advance one frame with button pressed
                self.pyboy.tick()

                # Release the button
                for button in buttons:
                    self.pyboy.button_release(button)
            else:
                # No action - just advance one frame
                self.pyboy.tick()
        else:
            # Invalid action - just advance one frame
            self.pyboy.tick()
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward (game-specific, can be overridden)
        reward = self._calculate_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = False  # Can be implemented based on time limits
        
        # Get additional info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get the current screen as observation."""
        # Get screen image from PyBoy (updated API)
        screen_image = self.pyboy.screen.ndarray.copy()

        # Handle different screen formats
        if len(screen_image.shape) == 3:
            if screen_image.shape[2] == 4:  # RGBA format
                # Convert RGBA to grayscale
                observation = np.dot(screen_image[...,:3], [0.2989, 0.5870, 0.1140])
                observation = observation.astype(np.uint8)
            elif screen_image.shape[2] == 3:  # RGB format
                # Convert RGB to grayscale
                observation = np.dot(screen_image[...,:3], [0.2989, 0.5870, 0.1140])
                observation = observation.astype(np.uint8)
            else:
                # Take first channel
                observation = screen_image[:,:,0].astype(np.uint8)
        else:
            # Already grayscale
            observation = screen_image.astype(np.uint8)

        # Add channel dimension for Gymnasium
        observation = np.expand_dims(observation, axis=2)

        return observation
    
    def _calculate_reward(self) -> float:
        """Calculate reward (to be overridden by game-specific wrappers)."""
        # Default reward: small positive for staying alive, but encourage action
        return 0.1
    
    def _is_terminated(self) -> bool:
        """Check if episode is terminated (to be overridden by game-specific wrappers)."""
        # Default: never terminate (continuous play)
        return False
    
    def _get_info(self) -> Dict:
        """Get additional information."""
        return {
            "game_title": self.cartridge_title,
            "frame_count": getattr(self, '_frame_count', 0)
        }
    
    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """Render the environment."""
        # Use the mode parameter or fall back to the instance render_mode
        render_mode = mode or self.render_mode or "rgb_array"

        if render_mode == "rgb_array":
            # Get fresh screen data (RGBA format) - make sure it's not cached
            screen_data = self.pyboy.screen.ndarray.copy()

            # Convert RGBA to RGB for video recording compatibility
            if len(screen_data.shape) == 3 and screen_data.shape[2] == 4:
                # Remove alpha channel
                screen_data = screen_data[:, :, :3]

            return screen_data
        elif render_mode == "human":
            # For human rendering, we'd need to create a window
            # For now, just return the array
            screen_data = self.pyboy.screen.ndarray.copy()
            if len(screen_data.shape) == 3 and screen_data.shape[2] == 4:
                screen_data = screen_data[:, :, :3]
            return screen_data
        else:
            return None
    
    def close(self):
        """Close the environment."""
        if self.pyboy:
            self.pyboy.stop()
            self.pyboy = None
        logger.info("PyBoy environment closed")


class PyBoyTetrisWrapper(PyBoyWrapper):
    """
    Tetris-specific wrapper with game-specific reward calculation.
    """

    def __init__(self, rom_path: str, config: Dict[str, Any], render_mode: Optional[str] = None, **kwargs):
        super().__init__(rom_path, config, render_mode, **kwargs)
        self.last_score = 0
        self.last_lines = 0
        self.steps_without_action = 0
        self.consecutive_menu_steps = 0  # Track how long we've been in menus
        self.last_was_in_game = True  # Assume we start in game after auto-start

        # Detailed tracking for debugging
        self.step_count = 0
        self.menu_returns = 0
        self.last_action = None
        self.action_history = []  # Track last 10 actions
        self.state_history = []   # Track last 10 states
        self.screen_save_counter = 0

    def _calculate_reward(self) -> float:
        """Calculate Tetris-specific reward based on gameplay."""
        reward = 0.0

        # Get current screen
        current_screen = self.pyboy.screen.ndarray.copy()

        # Check if we're in actual Tetris gameplay (not menus)
        # Look for the game board area (should have consistent structure)
        game_area = current_screen[20:140, 40:120]  # Main game board area
        game_pixels = np.sum(game_area > 128)

        # Detect if we're in menus vs gameplay with better detection
        is_in_game = self._detect_gameplay_state(current_screen)

        if is_in_game:
            # MASSIVE REWARDS for reaching and staying in game
            self.consecutive_menu_steps = 0  # Reset menu counter

            # HUGE bonus for first time reaching gameplay
            if not self.last_was_in_game:
                reward += 50.0  # Massive reward for reaching gameplay
                print(f"🎉 HUGE REWARD: Reached gameplay! Bonus: +50.0")

            # Focus on Tetris gameplay rewards
            if hasattr(self, '_last_screen'):
                screen_diff = np.sum(np.abs(current_screen.astype(float) - self._last_screen.astype(float)))

                if screen_diff > 500:  # Significant change (piece movement, line clear, etc.)
                    reward += 5.0  # Increased gameplay rewards
                    self.steps_without_action = 0
                elif screen_diff > 100:  # Small change (piece falling)
                    reward += 2.0  # Increased gameplay rewards
                    self.steps_without_action = 0
                else:
                    self.steps_without_action += 1

                # Penalize inactivity (encourage active play)
                if self.steps_without_action > 60:  # 2 seconds of no activity
                    reward -= 0.5
            else:
                self.steps_without_action = 0

            # Large bonus for staying in gameplay
            reward += 2.0  # Large bonus for staying in game
            self.last_was_in_game = True

        else:
            # ENCOURAGE menu navigation to reach gameplay
            self.consecutive_menu_steps += 1

            # Reward menu activity (encourage exploration)
            if hasattr(self, '_last_screen'):
                screen_diff = np.sum(np.abs(current_screen.astype(float) - self._last_screen.astype(float)))
                if screen_diff > 100:  # Menu navigation activity
                    reward += 1.0  # Good reward for menu activity
                    print(f"🎯 Menu navigation activity! Reward: +1.0")

            # Only punish if stuck in menus for too long
            if self.consecutive_menu_steps > 500:  # Much more patient
                menu_punishment = -0.5  # Gentle punishment
                reward += menu_punishment

            # Extra punishment if we were in game and returned to menu
            if self.last_was_in_game:
                reward -= 10.0  # Punishment for leaving gameplay
                print(f"🚨 PUNISHMENT: Left gameplay for menu! Penalty: -10.0")

            self.last_was_in_game = False

        # Store state for next comparison
        self._last_screen = current_screen.copy()

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with detailed tracking."""
        self.step_count += 1
        self.last_action = action

        # Track action history
        self.action_history.append(action)
        if len(self.action_history) > 10:
            self.action_history.pop(0)

        # Smart pause punishment - only punish Start during gameplay, not menus
        if action == 3:  # Start button
            # Check if we're likely in gameplay (not menus)
            current_screen = self.pyboy.screen.ndarray.copy()
            is_in_game = self._detect_gameplay_state(current_screen)

            if is_in_game:
                reward -= 1.0  # Stronger punishment for pausing during gameplay
                print(f"⏸️ Gameplay pause penalty: -1.0 (Start during game)")
            else:
                # Small reward for Start during menus (helps navigation)
                reward += 0.2
                print(f"🎯 Menu navigation: +0.2 (Start in menu)")

        # Execute the step
        obs, reward, terminated, truncated, info = super().step(action)

        # Get current screen for state analysis
        current_screen = self.pyboy.screen.ndarray.copy()
        is_in_game = self._detect_gameplay_state(current_screen)

        # Track state history
        state = "GAME" if is_in_game else "MENU"
        self.state_history.append(state)
        if len(self.state_history) > 10:
            self.state_history.pop(0)

        # Detect menu returns and log detailed info
        if not is_in_game and self.last_was_in_game:
            self.menu_returns += 1
            self._log_menu_return(action, current_screen)

        return obs, reward, terminated, truncated, info

    def _log_menu_return(self, triggering_action: int, screen: np.ndarray):
        """Log detailed information when model returns to menu."""
        action_names = {0: "NONE", 1: "A", 2: "B", 3: "START", 4: "SELECT", 5: "DOWN", 6: "UP", 7: "LEFT", 8: "RIGHT"}

        print(f"\n🚨 MENU RETURN #{self.menu_returns} at step {self.step_count}")
        print(f"   Triggering action: {triggering_action} ({action_names.get(triggering_action, 'UNKNOWN')})")
        print(f"   Last 10 actions: {[action_names.get(a, str(a)) for a in self.action_history]}")
        print(f"   Last 10 states: {self.state_history}")

        # Analyze screen regions
        title_area = screen[50:100, 50:110] if screen.shape[0] > 100 else screen[20:60, 20:80]
        title_pixels = np.sum(title_area > 128)

        game_area = screen[20:140, 40:120] if screen.shape[0] > 140 else screen[20:120, 40:120]
        game_pixels = np.sum(game_area > 128)

        ui_area = screen[20:140, 120:160] if screen.shape[1] > 160 else screen[20:120, 100:140]
        ui_pixels = np.sum(ui_area > 128)

        print(f"   Screen analysis: title_pixels={title_pixels}, game_pixels={game_pixels}, ui_pixels={ui_pixels}")

        # Save screenshot for analysis
        self.screen_save_counter += 1
        if len(screen.shape) == 3 and screen.shape[2] >= 3:
            screen_rgb = screen[:, :, :3]
        else:
            screen_rgb = np.stack([screen, screen, screen], axis=2)

        import cv2
        screen_large = cv2.resize(screen_rgb, (320, 288), interpolation=cv2.INTER_NEAREST)
        filename = f"menu_return_{self.menu_returns}_step_{self.step_count}.png"
        cv2.imwrite(filename, screen_large)
        print(f"   💾 Screenshot saved: {filename}")
        print()

    def _detect_gameplay_state(self, screen: np.ndarray) -> bool:
        """Detect if we're in actual gameplay vs menus with improved detection."""
        # Use the more accurate detection function
        return self._is_actually_in_game(screen)

    def _is_terminated(self) -> bool:
        """Check if Tetris game is over."""
        # This would need to read Tetris-specific game over state
        # For now, never terminate
        return False


def make_pyboy_env(rom_path: str, config: Dict[str, Any], **kwargs) -> gym.Env:
    """
    Create a PyBoy environment with appropriate wrappers.
    
    Args:
        rom_path: Path to the ROM file
        config: Configuration dictionary
        **kwargs: Additional arguments
    
    Returns:
        Wrapped PyBoy environment
    """
    
    # Determine game type from ROM path
    rom_name = Path(rom_path).stem.lower()
    
    if "tetris" in rom_name:
        env = PyBoyTetrisWrapper(rom_path, config, **kwargs)
    else:
        # Generic wrapper for other games
        env = PyBoyWrapper(rom_path, config, **kwargs)
    
    # Apply standard wrappers if needed
    if config.get("frame_stack", 0) > 1:
        from gymnasium.wrappers import FrameStack
        env = FrameStack(env, config["frame_stack"])
    
    logger.info(f"PyBoy environment created for: {rom_path}")
    return env


def get_rom_path(game_name: str) -> str:
    """
    Get the ROM path for a given game name.
    
    Args:
        game_name: Name of the game
    
    Returns:
        Path to the ROM file
    """
    
    # ROM directory
    rom_dir = Path("envs/gameboy_retro/roms/GameBoy")
    
    # Game name to ROM file mapping
    rom_mapping = {
        "tetris_gb_authentic": "Tetris (JUE) (V1.1) [!].gb",
        "tetris_authentic": "Tetris (JUE) (V1.1) [!].gb",
        "mario_land_authentic": "SuperMarioLand.gb",
        "super_mario_land_authentic": "SuperMarioLand.gb",
        "kirby_authentic": "KirbysDreamLand.gb",
        "kirbys_dream_land_authentic": "KirbysDreamLand.gb"
    }
    
    # Get ROM filename
    rom_filename = rom_mapping.get(game_name.lower())
    if not rom_filename:
        # Try direct mapping
        rom_filename = f"{game_name}.gb"
    
    rom_path = rom_dir / rom_filename
    
    # Also check for .gbc files
    if not rom_path.exists():
        rom_path = rom_dir / f"{Path(rom_filename).stem}.gbc"
    
    return str(rom_path)
