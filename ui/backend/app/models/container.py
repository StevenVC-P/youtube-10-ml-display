"""
Container data models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ContainerStatus(str, Enum):
    """Container lifecycle states."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DELETED = "deleted"


class ResourceSpec(BaseModel):
    """Resource allocation specification."""
    cpu_cores: float = Field(default=2.0, ge=0.1, le=32.0)
    memory_gb: float = Field(default=4.0, ge=0.5, le=128.0)
    gpu_memory_gb: Optional[float] = Field(default=None, ge=0.1, le=80.0)
    disk_space_gb: float = Field(default=10.0, ge=1.0, le=1000.0)
    
    class Config:
        schema_extra = {
            "example": {
                "cpu_cores": 4.0,
                "memory_gb": 8.0,
                "gpu_memory_gb": 4.0,
                "disk_space_gb": 20.0
            }
        }


class TrainingConfig(BaseModel):
    """ML training configuration."""
    game: str = Field(..., pattern="^[a-zA-Z0-9_-]+$")
    algorithm: str = Field(default="ppo", pattern="^(ppo|dqn)$")
    total_timesteps: int = Field(default=1000000, ge=1000, le=100000000)
    vec_envs: int = Field(default=4, ge=1, le=16)
    learning_rate: float = Field(default=2.5e-4, ge=1e-6, le=1e-1)
    checkpoint_every_sec: int = Field(default=60, ge=10, le=3600)
    video_recording: bool = Field(default=True)
    resource_limits: ResourceSpec = Field(default_factory=ResourceSpec)
    
    # Additional training parameters
    batch_size: int = Field(default=256, ge=32, le=2048)
    n_steps: int = Field(default=128, ge=32, le=2048)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    gae_lambda: float = Field(default=0.95, ge=0.9, le=0.99)
    clip_range: float = Field(default=0.1, ge=0.05, le=0.5)
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.1)
    vf_coef: float = Field(default=0.5, ge=0.1, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "game": "breakout",
                "algorithm": "ppo",
                "total_timesteps": 2000000,
                "vec_envs": 8,
                "learning_rate": 2.5e-4,
                "checkpoint_every_sec": 120,
                "video_recording": True,
                "resource_limits": {
                    "cpu_cores": 4.0,
                    "memory_gb": 8.0,
                    "gpu_memory_gb": 4.0,
                    "disk_space_gb": 20.0
                }
            }
        }
    
    def to_yaml_config(self) -> Dict[str, Any]:
        """Convert to format compatible with existing config.yaml."""
        return {
            "project_name": f"container-{self.game}-{self.algorithm}",
            "seed": 42,
            "paths": {
                "videos_milestones": "video/milestones",
                "videos_eval": "video/eval",
                "videos_parts": "video/render/parts",
                "logs_tb": "logs/tb",
                "models": "models/checkpoints",
                "ffmpeg_path": "ffmpeg"
            },
            "game": {
                "env_id": f"{self.game.title()}NoFrameskip-v4",
                "frame_stack": 4,
                "resize": [84, 84],
                "grayscale": True,
                "max_skip": 4
            },
            "train": {
                "algo": self.algorithm,
                "policy": "CnnPolicy",
                "total_timesteps": self.total_timesteps,
                "vec_envs": self.vec_envs,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "vf_coef": self.vf_coef,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "checkpoint_every_sec": self.checkpoint_every_sec,
                "keep_last": 5
            },
            "recording": {
                "fps": 30,
                "crf": 23,
                "milestone_clip_seconds": 90,
                "eval_clip_seconds": 120,
                "milestones_pct": [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "eval_every_steps": 100000,
                "eval_episodes": 2,
                "enabled": self.video_recording
            },
            "stream": {
                "enabled": self.video_recording,
                "grid": 4,
                "pane_size": [480, 360],
                "fps": 30,
                "overlay_hud": True,
                "checkpoint_poll_sec": 30,
                "save_mode": "segments",
                "segment_seconds": 1800,
                "output_basename": f"container_{self.game}_{self.algorithm}",
                "preset": "veryfast",
                "crf": 23
            }
        }


class Container(BaseModel):
    """ML training container model."""
    id: str
    name: str
    config: TrainingConfig
    status: ContainerStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    process_id: Optional[int] = None
    working_directory: Optional[str] = None
    log_file: Optional[str] = None
    error_message: Optional[str] = None
    
    # Runtime information
    current_timesteps: int = 0
    current_episodes: int = 0
    current_reward: float = 0.0
    checkpoint_count: int = 0
    video_count: int = 0
    
    class Config:
        schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Breakout Training Session",
                "status": "running",
                "created_at": "2023-10-16T10:00:00Z",
                "started_at": "2023-10-16T10:01:00Z",
                "process_id": 12345,
                "current_timesteps": 150000,
                "current_episodes": 75,
                "current_reward": 45.2
            }
        }
    
    def get_runtime_duration(self) -> Optional[float]:
        """Get runtime duration in seconds."""
        if not self.started_at:
            return None
        
        end_time = self.stopped_at or datetime.now()
        return (end_time - self.started_at).total_seconds()
    
    def get_progress_percentage(self) -> float:
        """Get training progress as percentage."""
        if self.config.total_timesteps == 0:
            return 0.0
        
        return min(100.0, (self.current_timesteps / self.config.total_timesteps) * 100.0)
    
    def is_active(self) -> bool:
        """Check if container is in an active state."""
        return self.status in [ContainerStatus.STARTING, ContainerStatus.RUNNING]
    
    def can_start(self) -> bool:
        """Check if container can be started."""
        return self.status in [ContainerStatus.CREATED, ContainerStatus.STOPPED]
    
    def can_stop(self) -> bool:
        """Check if container can be stopped."""
        return self.status in [ContainerStatus.RUNNING, ContainerStatus.PAUSED]
    
    def can_delete(self) -> bool:
        """Check if container can be deleted."""
        return self.status in [
            ContainerStatus.CREATED, 
            ContainerStatus.STOPPED, 
            ContainerStatus.ERROR
        ]
