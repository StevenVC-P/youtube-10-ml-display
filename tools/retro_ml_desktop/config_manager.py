"""
Configuration Manager for Retro ML Trainer

Handles all application configuration, paths, and settings.
Supports first-run detection and setup wizard integration.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
import shutil


class ConfigManager:
    """Manages application configuration and paths."""
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        'app': {
            'name': 'Retro ML Trainer',
            'version': '1.0.0',
            'first_run_completed': False
        },
        'paths': {
            'install_dir': None,  # Set at runtime
            'models_dir': 'models',
            'videos_dir': 'videos',
            'database_dir': 'data',
            'logs_dir': 'logs',
            'config_dir': 'config'
        },
        'training': {
            'default_game': 'breakout',
            'default_algorithm': 'PPO',
            'default_timesteps': 1000000,
            'use_gpu': True,  # Will be auto-detected
            'auto_save_interval': 300  # seconds
        },
        'video': {
            'fps': 30,
            'quality': 'high',
            'auto_generate': True,
            'milestone_intervals': [0, 1, 5, 10, 25, 50, 75, 90, 100]
        },
        'system': {
            'gpu_detected': False,
            'cuda_available': False,
            'ffmpeg_available': False,
            'atari_roms_installed': False
        }
    }
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to config file. If None, uses default location.
        """
        # Determine install directory (where the executable/script is running from)
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            self.install_dir = Path(sys.executable).parent
        else:
            # Running as script - use project root
            self.install_dir = Path(__file__).parent.parent.parent
        
        # Set config file path
        if config_file is None:
            self.config_dir = self.install_dir / 'config'
            self.config_dir.mkdir(exist_ok=True)
            self.config_file = self.config_dir / 'settings.yaml'
        else:
            self.config_file = Path(config_file)
            self.config_dir = self.config_file.parent
        
        # Load or create configuration
        self.config = self._load_or_create_config()
        
        # Update install_dir in config
        self.config['paths']['install_dir'] = str(self.install_dir)
    
    def _load_or_create_config(self) -> Dict[str, Any]:
        """Load existing config or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_configs(self.DEFAULT_CONFIG.copy(), config)
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Using default configuration")
                return self.DEFAULT_CONFIG.copy()
        else:
            # First run - create default config
            return self.DEFAULT_CONFIG.copy()
    
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = self._merge_configs(default[key], value)
            else:
                default[key] = value
        return default
    
    def save(self):
        """Save current configuration to file."""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def is_first_run(self) -> bool:
        """Check if this is the first run of the application."""
        return not self.config['app']['first_run_completed']
    
    def mark_first_run_complete(self):
        """Mark first run as completed."""
        print(f"Marking first run complete...")
        print(f"Config file: {self.config_file}")
        self.config['app']['first_run_completed'] = True
        result = self.save()
        if result:
            print(f"✓ First run marked complete and saved successfully")
        else:
            print(f"✗ ERROR: Failed to save first run completion!")
        return result
    
    def get_path(self, path_key: str, create: bool = True) -> Path:
        """
        Get a configured path, optionally creating it.
        
        Args:
            path_key: Key from paths config (e.g., 'models_dir', 'videos_dir')
            create: Whether to create the directory if it doesn't exist
            
        Returns:
            Absolute Path object
        """
        if path_key not in self.config['paths']:
            raise ValueError(f"Unknown path key: {path_key}")
        
        path_value = self.config['paths'][path_key]
        
        # Handle install_dir specially (already absolute)
        if path_key == 'install_dir':
            return Path(path_value)
        
        # Convert relative paths to absolute (relative to install_dir)
        if not Path(path_value).is_absolute():
            full_path = self.install_dir / path_value
        else:
            full_path = Path(path_value)
        
        # Create directory if requested
        if create:
            full_path.mkdir(parents=True, exist_ok=True)
        
        return full_path
    
    def set_path(self, path_key: str, path_value: str):
        """Set a path in configuration."""
        if path_key not in self.config['paths']:
            raise ValueError(f"Unknown path key: {path_key}")
        
        self.config['paths'][path_key] = path_value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'training.default_game')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """
        Set a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'training.default_game')
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def detect_system_capabilities(self) -> Dict[str, bool]:
        """
        Detect system capabilities (GPU, CUDA, FFmpeg, etc.).
        
        Returns:
            Dictionary of detected capabilities
        """
        capabilities = {
            'gpu_detected': False,
            'cuda_available': False,
            'ffmpeg_available': False,
            'atari_roms_installed': False
        }
        
        # Check for GPU/CUDA
        try:
            import torch
            capabilities['cuda_available'] = torch.cuda.is_available()
            capabilities['gpu_detected'] = torch.cuda.device_count() > 0
        except ImportError:
            pass
        
        # Check for FFmpeg
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  timeout=5)
            capabilities['ffmpeg_available'] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Check for Atari ROMs
        try:
            import ale_py.roms as roms
            # Check if at least one ROM is available
            rom_dir = Path(roms.__file__).parent
            rom_files = list(rom_dir.glob('*.bin'))
            capabilities['atari_roms_installed'] = len(rom_files) > 0
        except (ImportError, AttributeError):
            pass
        
        # Update config
        self.config['system'].update(capabilities)
        
        return capabilities
    
    def get_database_path(self) -> Path:
        """Get the path to the SQLite database."""
        db_dir = self.get_path('database_dir')
        return db_dir / 'ml_experiments.db'
    
    def initialize_directories(self):
        """Create all configured directories."""
        for key in ['models_dir', 'videos_dir', 'database_dir', 'logs_dir', 'config_dir']:
            self.get_path(key, create=True)
    
    def export_config(self) -> str:
        """Export configuration as YAML string."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_file={self.config_file})"


# Import sys for frozen executable detection
import sys

