"""
Configuration management for the ML Container Management API.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    api_title: str = "ML Container Management API"
    api_version: str = "1.0.0"
    api_description: str = "API for managing multiple ML training containers with resource monitoring"
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=True, env="RELOAD")
    
    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./containers.db", env="DATABASE_URL")
    
    # Redis Configuration (for caching and real-time features)
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_enabled: bool = Field(default=False, env="REDIS_ENABLED")
    
    # Container Management
    containers_base_dir: str = Field(default="./containers", env="CONTAINERS_BASE_DIR")
    max_containers: int = Field(default=10, env="MAX_CONTAINERS")
    default_container_timeout: int = Field(default=3600, env="DEFAULT_CONTAINER_TIMEOUT")  # 1 hour
    
    # Resource Monitoring
    monitoring_interval: float = Field(default=5.0, env="MONITORING_INTERVAL")
    monitoring_history_hours: int = Field(default=24, env="MONITORING_HISTORY_HOURS")
    
    # WebSocket Configuration
    websocket_broadcast_interval: float = Field(default=2.0, env="WEBSOCKET_BROADCAST_INTERVAL")
    
    # ML Training Configuration
    ml_models_path: str = Field(default="../models", env="ML_MODELS_PATH")
    ml_videos_path: str = Field(default="../video", env="ML_VIDEOS_PATH")
    ml_logs_path: str = Field(default="../logs", env="ML_LOGS_PATH")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Resource Limits
    default_cpu_cores: float = Field(default=2.0, env="DEFAULT_CPU_CORES")
    default_memory_gb: float = Field(default=4.0, env="DEFAULT_MEMORY_GB")
    default_disk_space_gb: float = Field(default=10.0, env="DEFAULT_DISK_SPACE_GB")
    max_cpu_cores: float = Field(default=16.0, env="MAX_CPU_CORES")
    max_memory_gb: float = Field(default=64.0, env="MAX_MEMORY_GB")
    max_disk_space_gb: float = Field(default=500.0, env="MAX_DISK_SPACE_GB")
    
    # GPU Configuration
    gpu_enabled: bool = Field(default=True, env="GPU_ENABLED")
    max_gpu_memory_gb: float = Field(default=24.0, env="MAX_GPU_MEMORY_GB")
    
    # Video Streaming
    video_streaming_enabled: bool = Field(default=True, env="VIDEO_STREAMING_ENABLED")
    video_stream_port_start: int = Field(default=8100, env="VIDEO_STREAM_PORT_START")
    video_stream_port_end: int = Field(default=8200, env="VIDEO_STREAM_PORT_END")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    testing: bool = Field(default=False, env="TESTING")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        Path(self.containers_base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.ml_models_path).mkdir(parents=True, exist_ok=True)
        Path(self.ml_videos_path).mkdir(parents=True, exist_ok=True)
        Path(self.ml_logs_path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    reload: bool = True
    log_level: str = "DEBUG"


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    reload: bool = False
    log_level: str = "INFO"
    
    # More restrictive defaults for production
    max_containers: int = 20
    monitoring_interval: float = 10.0
    websocket_broadcast_interval: float = 5.0


class TestingSettings(Settings):
    """Testing environment settings."""
    testing: bool = True
    database_url: str = "sqlite:///./test_containers.db"
    containers_base_dir: str = "./test_containers"
    monitoring_interval: float = 1.0
    websocket_broadcast_interval: float = 0.5


def get_settings() -> Settings:
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export the appropriate settings
settings = get_settings()
