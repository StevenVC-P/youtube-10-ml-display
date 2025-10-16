#!/usr/bin/env python3
"""
Gameboy environment wrappers for stable-retro integration.
Provides compatibility layer between stable-retro and the main training system.
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np

class GameboyRetroProxy:
    """
    Proxy class that runs Gameboy games in a separate Python 3.12 environment
    while maintaining compatibility with the main training system.
    """
    
    def __init__(self, game_name: str, config: Dict[str, Any], **kwargs):
        self.game_name = game_name
        self.config = config
        self.gameboy_env_path = Path("envs/gameboy_retro")
        self.python_path = self.gameboy_env_path / "venv" / "Scripts" / "python.exe"
        
        # Check if gameboy environment exists
        if not self.python_path.exists():
            raise RuntimeError(
                f"Gameboy environment not found at {self.python_path}. "
                "Run setup_gameboy_emulation.py first."
            )
        
        # Create temporary communication files
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gameboy_proxy_"))
        self.command_file = self.temp_dir / "command.json"
        self.response_file = self.temp_dir / "response.json"
        
        # Start the gameboy environment process
        self._start_gameboy_process()
    
    def _start_gameboy_process(self):
        """Start the gameboy environment in a separate process."""
        proxy_script = self._create_proxy_script()
        
        # Start the process
        self.process = subprocess.Popen([
            str(self.python_path),
            str(proxy_script),
            str(self.command_file),
            str(self.response_file),
            self.game_name
        ], cwd=str(self.gameboy_env_path))
    
    def _create_proxy_script(self) -> Path:
        """Create the proxy script for the gameboy environment."""
        script_path = self.temp_dir / "gameboy_proxy.py"
        
        script_content = '''
import sys
import json
import time
import retro
import numpy as np
from pathlib import Path

def main():
    command_file = Path(sys.argv[1])
    response_file = Path(sys.argv[2])
    game_name = sys.argv[3]
    
    # Create environment
    env = retro.make(game_name)
    obs, info = env.reset()
    
    # Send initial observation
    response = {
        "type": "reset",
        "observation": obs.tolist(),
        "info": info,
        "observation_space": {
            "shape": obs.shape,
            "dtype": str(obs.dtype)
        },
        "action_space": {
            "n": env.action_space.n
        }
    }
    
    with open(response_file, 'w') as f:
        json.dump(response, f)
    
    # Main loop
    while True:
        if command_file.exists():
            with open(command_file, 'r') as f:
                command = json.load(f)
            
            command_file.unlink()  # Remove command file
            
            if command["type"] == "step":
                obs, reward, terminated, truncated, info = env.step(command["action"])
                response = {
                    "type": "step",
                    "observation": obs.tolist(),
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info
                }
            elif command["type"] == "reset":
                obs, info = env.reset()
                response = {
                    "type": "reset",
                    "observation": obs.tolist(),
                    "info": info
                }
            elif command["type"] == "close":
                env.close()
                break
            else:
                response = {"type": "error", "message": f"Unknown command: {command['type']}"}
            
            with open(response_file, 'w') as f:
                json.dump(response, f)
        
        time.sleep(0.001)  # Small delay to prevent busy waiting

if __name__ == "__main__":
    main()
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return script_path
    
    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send a command to the gameboy process and wait for response."""
        # Write command
        with open(self.command_file, 'w') as f:
            json.dump(command, f)
        
        # Wait for response
        timeout = 10.0  # 10 second timeout
        start_time = time.time()
        
        while not self.response_file.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError("Gameboy process did not respond in time")
            time.sleep(0.01)
        
        # Read response
        with open(self.response_file, 'r') as f:
            response = json.load(f)
        
        # Remove response file
        self.response_file.unlink()
        
        return response
    
    def reset(self, **kwargs):
        """Reset the environment."""
        response = self._send_command({"type": "reset"})
        obs = np.array(response["observation"], dtype=np.uint8)
        return obs, response["info"]
    
    def step(self, action):
        """Step the environment."""
        response = self._send_command({"type": "step", "action": int(action)})
        obs = np.array(response["observation"], dtype=np.uint8)
        return obs, response["reward"], response["terminated"], response["truncated"], response["info"]
    
    def close(self):
        """Close the environment."""
        try:
            self._send_command({"type": "close"})
        except:
            pass  # Process might already be closed
        
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait()
        
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

class GameboyEnvironmentWrapper(gym.Env):
    """
    Gymnasium wrapper for Gameboy games using stable-retro.
    Makes Gameboy games compatible with the main training system.
    """
    
    def __init__(self, game_name: str, config: Dict[str, Any], **kwargs):
        super().__init__()
        
        self.game_name = game_name
        self.config = config
        
        # Create the proxy
        self.proxy = GameboyRetroProxy(game_name, config, **kwargs)
        
        # Get initial observation to set up spaces
        initial_obs, _ = self.proxy.reset()
        
        # Set up observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=initial_obs.shape, 
            dtype=np.uint8
        )
        
        # Gameboy typically has 8 actions (A, B, Select, Start, Up, Down, Left, Right)
        self.action_space = gym.spaces.Discrete(8)
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.proxy.reset(**kwargs)
    
    def step(self, action):
        """Step the environment."""
        return self.proxy.step(action)
    
    def close(self):
        """Close the environment."""
        self.proxy.close()

def make_gameboy_env(env_id: str, config: Dict[str, Any], seed: int = None, **kwargs) -> gym.Env:
    """
    Create and wrap a Gameboy environment with standard preprocessing.
    
    Args:
        env_id: Environment ID (e.g., "Tetris-GameBoy")
        config: Configuration dictionary
        seed: Random seed
        **kwargs: Additional arguments
    
    Returns:
        Wrapped Gameboy environment
    """
    # Extract game name from env_id
    game_name = env_id
    
    # Create the base environment
    env = GameboyEnvironmentWrapper(game_name, config, **kwargs)
    
    if seed is not None:
        env.reset(seed=seed)
    
    # Apply standard preprocessing (similar to Atari)
    game_config = config.get('game', {})
    
    # Apply grayscale conversion if requested
    if game_config.get('grayscale', True):
        env = GameboyGrayscaleWrapper(env)
    
    # Apply resizing if requested
    resize = game_config.get('resize', [84, 84])
    if resize:
        env = GameboyResizeWrapper(env, tuple(resize))
    
    # Apply frame stacking if requested
    frame_stack = game_config.get('frame_stack', 4)
    if frame_stack > 1:
        env = GameboyFrameStackWrapper(env, frame_stack)
    
    return env

# Additional wrapper classes (simplified versions of the Tetris wrappers)
class GameboyGrayscaleWrapper(gym.ObservationWrapper):
    """Convert Gameboy observations to grayscale."""
    
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8
        )
    
    def observation(self, observation):
        # Convert RGB to grayscale
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            gray = np.dot(observation[...,:3], [0.2989, 0.5870, 0.1140])
            return np.expand_dims(gray.astype(np.uint8), axis=2)
        return observation

class GameboyResizeWrapper(gym.ObservationWrapper):
    """Resize Gameboy observations."""
    
    def __init__(self, env, size):
        super().__init__(env)
        self.size = size
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(size[0], size[1], obs_shape[2]),
            dtype=np.uint8
        )
    
    def observation(self, observation):
        import cv2
        return cv2.resize(observation, self.size, interpolation=cv2.INTER_AREA)

class GameboyFrameStackWrapper(gym.Wrapper):
    """Stack multiple frames for Gameboy environments."""
    
    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(num_stack, obs_shape[0], obs_shape[1]),
            dtype=np.uint8
        )
        
        self.frames = np.zeros((num_stack, obs_shape[0], obs_shape[1]), dtype=np.uint8)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.num_stack):
            self.frames[i] = obs[:, :, 0]  # Take first channel
        return self.frames.copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs[:, :, 0]  # Take first channel
        return self.frames.copy(), reward, terminated, truncated, info
