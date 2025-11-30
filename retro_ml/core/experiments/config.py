"""
Experiment Configuration - Single Source of Truth

This module defines the ExperimentConfig class, which serves as the single
source of truth for all experiment parameters.
"""

import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field, field_validator, model_validator


class RunType(str, Enum):
    """Training run duration presets"""
    QUICK = "quick"      # ~5-10 minutes (for testing)
    SHORT = "short"      # 1-2 hours
    MEDIUM = "medium"    # 4-6 hours
    LONG = "long"        # 10+ hours
    CUSTOM = "custom"    # User-defined duration


class ExperimentConfig(BaseModel):
    """
    Experiment configuration object - single source of truth.
    
    This class encapsulates all parameters needed to run a training experiment,
    including game selection, algorithm, hyperparameters, and output paths.
    """
    
    # Unique experiment identifier
    experiment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this experiment"
    )
    
    # Game configuration
    game_id: str = Field(
        ...,
        description="Atari game ID (e.g., 'BreakoutNoFrameskip-v4')"
    )
    
    # Algorithm configuration
    algorithm: str = Field(
        default="PPO",
        description="RL algorithm to use (PPO, A2C, DQN)"
    )
    
    # Training configuration
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility (None = random)"
    )
    
    total_timesteps: int = Field(
        default=10000,
        gt=0,
        description="Total number of timesteps to train"
    )
    
    run_type: RunType = Field(
        default=RunType.QUICK,
        description="Training run duration preset"
    )
    
    # Output configuration
    output_dir: Path = Field(
        default_factory=lambda: Path("D:/RetroML/experiments"),
        description="Base directory for experiment outputs"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when config was created"
    )
    
    # GPU configuration
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU if available"
    )
    
    # Advanced configuration (for future phases)
    learning_rate: float = Field(
        default=0.0003,
        gt=0,
        description="Learning rate for the optimizer"
    )
    
    n_envs: int = Field(
        default=4,
        gt=0,
        description="Number of parallel environments"
    )

    # PPO-specific hyperparameters
    n_steps: int = Field(
        default=128,
        gt=0,
        description="Number of steps per update (PPO)"
    )

    batch_size: int = Field(
        default=256,
        gt=0,
        description="Minibatch size for training"
    )

    gamma: float = Field(
        default=0.99,
        ge=0,
        le=1,
        description="Discount factor"
    )

    gae_lambda: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description="GAE lambda parameter (PPO)"
    )

    ent_coef: float = Field(
        default=0.01,
        ge=0,
        description="Entropy coefficient"
    )

    vf_coef: float = Field(
        default=0.5,
        ge=0,
        description="Value function coefficient"
    )

    max_grad_norm: float = Field(
        default=0.5,
        ge=0,
        description="Maximum gradient norm for clipping"
    )

    # Video generation configuration
    video_length_hours: float = Field(
        default=1.0,
        gt=0,
        description="Target video length in hours"
    )

    milestone_percentages: List[int] = Field(
        default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        description="Training progress percentages for milestone clips"
    )

    clip_seconds: int = Field(
        default=90,
        gt=0,
        description="Length of milestone clips in seconds"
    )

    # Environment configuration
    frame_stack: int = Field(
        default=4,
        gt=0,
        description="Number of frames to stack"
    )

    # Model architecture
    policy: str = Field(
        default="CnnPolicy",
        description="Policy network architecture"
    )

    net_arch: Optional[List] = Field(
        default=None,
        description="Custom network architecture (None = default)"
    )

    @field_validator("game_id")
    @classmethod
    def validate_game_id(cls, v: str) -> str:
        """Validate game ID format"""
        if not v:
            raise ValueError("game_id cannot be empty")
        # For now, just check it's not empty. Later we can validate against known games
        return v
    
    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate algorithm is supported"""
        supported = ["PPO", "A2C", "DQN"]
        if v.upper() not in supported:
            raise ValueError(f"Algorithm must be one of {supported}, got {v}")
        return v.upper()
    
    @model_validator(mode="after")
    def set_timesteps_from_run_type(self):
        """Set total_timesteps based on run_type if not explicitly set"""
        # Only auto-set if total_timesteps is still the default
        if self.total_timesteps == 10000 and self.run_type != RunType.CUSTOM:
            timestep_map = {
                RunType.QUICK: 10_000,      # ~5-10 minutes
                RunType.SHORT: 500_000,     # ~1-2 hours
                RunType.MEDIUM: 2_000_000,  # ~4-6 hours
                RunType.LONG: 10_000_000,   # ~10+ hours
            }
            self.total_timesteps = timestep_map.get(self.run_type, 10_000)
        return self
    
    @property
    def experiment_dir(self) -> Path:
        """Get the directory for this specific experiment"""
        return self.output_dir / self.experiment_id
    
    @property
    def metrics_path(self) -> Path:
        """Get path to metrics file"""
        return self.experiment_dir / "metrics.json"
    
    @property
    def model_path(self) -> Path:
        """Get path to model checkpoint"""
        return self.experiment_dir / "model.zip"
    
    @property
    def log_path(self) -> Path:
        """Get path to log file"""
        return self.experiment_dir / "training.log"
    
    @property
    def video_path(self) -> Path:
        """Get path to video file"""
        return self.experiment_dir / "video.mp4"
    
    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist"""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_preset(
        cls,
        game_id: str,
        preset: str = "quick",
        video_length_hours: float = 1.0,
        **kwargs
    ) -> "ExperimentConfig":
        """
        Create config from preset name.

        Args:
            game_id: Atari game ID
            preset: 'quick' (30m), 'short' (1-4h), 'medium' (4-6h), 'long' (10h)
            video_length_hours: Target video length
            **kwargs: Additional config overrides

        Returns:
            ExperimentConfig instance
        """
        # Calculate timesteps based on video length
        # Assuming ~30 FPS, 8 envs, and want real-time video
        seconds = video_length_hours * 3600
        timesteps = int(seconds * 30 * 8)  # FPS * envs

        preset_configs = {
            "quick": {
                "algorithm": "PPO",
                "total_timesteps": min(timesteps, 1_000_000),  # Cap at 1M for quick
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "video_length_hours": video_length_hours,
                "n_envs": 8,
                "run_type": RunType.QUICK,
            },
            "short": {
                "algorithm": "PPO",
                "total_timesteps": timesteps,
                "learning_rate": 2.5e-4,
                "n_steps": 128,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "video_length_hours": video_length_hours,
                "n_envs": 8,
                "run_type": RunType.SHORT,
            },
            "medium": {
                "algorithm": "PPO",
                "total_timesteps": timesteps,
                "learning_rate": 2.5e-4,
                "n_steps": 128,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "video_length_hours": video_length_hours,
                "n_envs": 8,
                "run_type": RunType.MEDIUM,
            },
            "long": {
                "algorithm": "PPO",
                "total_timesteps": timesteps,
                "learning_rate": 2.5e-4,
                "n_steps": 128,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "video_length_hours": video_length_hours,
                "n_envs": 8,
                "run_type": RunType.LONG,
            },
        }

        if preset not in preset_configs:
            raise ValueError(f"Unknown preset: {preset}. Must be one of {list(preset_configs.keys())}")

        # Merge preset config with kwargs
        config_dict = {"game_id": game_id, **preset_configs[preset], **kwargs}
        return cls(**config_dict)

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }

