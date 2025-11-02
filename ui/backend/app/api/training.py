"""
API endpoints for training session management.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.container_manager import container_manager
from ..core.training_wrapper import training_manager
from ..core.resource_monitor import resource_monitor

router = APIRouter(prefix="/api/training", tags=["training"])


# Request/Response Models
class TrainingMetricsResponse(BaseModel):
    """Training metrics response model."""
    container_id: str
    timesteps: int
    episodes: int
    reward: float
    progress: float
    runtime_seconds: Optional[float]
    
    
class TrainingProgressResponse(BaseModel):
    """Training progress response model."""
    container_id: str
    container_name: str
    game: str
    algorithm: str
    status: str
    progress_percentage: float
    current_timesteps: int
    total_timesteps: int
    current_episodes: int
    current_reward: float
    runtime_duration: Optional[float]
    estimated_completion: Optional[datetime]


class SessionSummaryResponse(BaseModel):
    """Session summary response model."""
    total_sessions: int
    active_sessions: int
    completed_sessions: int
    failed_sessions: int
    total_runtime_hours: float
    total_timesteps: int
    average_reward: float


class GameConfigResponse(BaseModel):
    """Game configuration response model."""
    game: str
    display_name: str
    description: str
    default_algorithm: str
    supported_algorithms: List[str]
    default_timesteps: int
    estimated_training_time: str


# API Endpoints
@router.get("/sessions", response_model=List[TrainingProgressResponse])
async def list_training_sessions():
    """List all training sessions with progress information."""
    containers = container_manager.list_containers()
    sessions = []
    
    for container in containers:
        # Get training metrics
        metrics = training_manager.get_training_metrics(container.id)
        
        # Calculate estimated completion
        estimated_completion = None
        if (container.started_at and 
            container.current_timesteps > 0 and 
            container.config.total_timesteps > container.current_timesteps):
            
            elapsed = (datetime.now() - container.started_at).total_seconds()
            timesteps_per_second = container.current_timesteps / elapsed
            remaining_timesteps = container.config.total_timesteps - container.current_timesteps
            remaining_seconds = remaining_timesteps / timesteps_per_second
            estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
        
        sessions.append(TrainingProgressResponse(
            container_id=container.id,
            container_name=container.name,
            game=container.config.game,
            algorithm=container.config.algorithm,
            status=container.status.value,
            progress_percentage=container.get_progress_percentage(),
            current_timesteps=container.current_timesteps,
            total_timesteps=container.config.total_timesteps,
            current_episodes=container.current_episodes,
            current_reward=container.current_reward,
            runtime_duration=container.get_runtime_duration(),
            estimated_completion=estimated_completion
        ))
    
    return sessions


@router.get("/sessions/{container_id}", response_model=TrainingProgressResponse)
async def get_training_session(container_id: str):
    """Get detailed training session information."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Get training metrics
    metrics = training_manager.get_training_metrics(container_id)
    
    # Calculate estimated completion
    estimated_completion = None
    if (container.started_at and 
        container.current_timesteps > 0 and 
        container.config.total_timesteps > container.current_timesteps):
        
        elapsed = (datetime.now() - container.started_at).total_seconds()
        timesteps_per_second = container.current_timesteps / elapsed
        remaining_timesteps = container.config.total_timesteps - container.current_timesteps
        remaining_seconds = remaining_timesteps / timesteps_per_second
        estimated_completion = datetime.now() + timedelta(seconds=remaining_seconds)
    
    return TrainingProgressResponse(
        container_id=container.id,
        container_name=container.name,
        game=container.config.game,
        algorithm=container.config.algorithm,
        status=container.status.value,
        progress_percentage=container.get_progress_percentage(),
        current_timesteps=container.current_timesteps,
        total_timesteps=container.config.total_timesteps,
        current_episodes=container.current_episodes,
        current_reward=container.current_reward,
        runtime_duration=container.get_runtime_duration(),
        estimated_completion=estimated_completion
    )


@router.get("/sessions/{container_id}/metrics", response_model=TrainingMetricsResponse)
async def get_training_metrics(container_id: str):
    """Get training metrics for a specific session."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    metrics = training_manager.get_training_metrics(container_id)
    
    return TrainingMetricsResponse(
        container_id=container_id,
        timesteps=metrics.get("timesteps", container.current_timesteps),
        episodes=metrics.get("episodes", container.current_episodes),
        reward=metrics.get("reward", container.current_reward),
        progress=metrics.get("progress", container.get_progress_percentage()),
        runtime_seconds=container.get_runtime_duration()
    )


@router.get("/sessions/{container_id}/logs")
async def get_training_logs(
    container_id: str,
    lines: int = Query(default=100, ge=1, le=10000),
    follow: bool = Query(default=False)
):
    """Get training logs for a specific session."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    if not container.log_file:
        return {"logs": "No log file available"}
    
    try:
        import subprocess
        
        if follow:
            # For real-time log following, this would typically use WebSocket
            # For now, just return recent logs
            cmd = ["tail", "-n", str(lines), container.log_file]
        else:
            cmd = ["tail", "-n", str(lines), container.log_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {"logs": result.stdout, "container_id": container_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@router.post("/sessions/{container_id}/pause")
async def pause_training_session(container_id: str, background_tasks: BackgroundTasks):
    """Pause a training session (save checkpoint and stop)."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    if container.status.value != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot pause session in {container.status.value} state"
        )
    
    # TODO: Implement pause functionality (save checkpoint then stop)
    background_tasks.add_task(container_manager.stop_container, container_id, False)
    
    return {"message": "Training session pause initiated", "container_id": container_id}


@router.post("/sessions/{container_id}/resume")
async def resume_training_session(container_id: str, background_tasks: BackgroundTasks):
    """Resume a paused training session."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    if container.status.value != "stopped":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume session in {container.status.value} state"
        )
    
    # TODO: Implement resume functionality (load from checkpoint)
    background_tasks.add_task(container_manager.start_container, container_id)
    
    return {"message": "Training session resume initiated", "container_id": container_id}


@router.get("/summary", response_model=SessionSummaryResponse)
async def get_training_summary():
    """Get summary of all training sessions."""
    containers = container_manager.list_containers()
    
    total_sessions = len(containers)
    active_sessions = len([c for c in containers if c.status.value == "running"])
    completed_sessions = len([c for c in containers if c.status.value == "stopped" and c.get_progress_percentage() >= 100])
    failed_sessions = len([c for c in containers if c.status.value == "error"])
    
    total_runtime_hours = 0.0
    total_timesteps = 0
    total_reward = 0.0
    reward_count = 0
    
    for container in containers:
        runtime = container.get_runtime_duration()
        if runtime:
            total_runtime_hours += runtime / 3600.0
        
        total_timesteps += container.current_timesteps
        
        if container.current_reward > 0:
            total_reward += container.current_reward
            reward_count += 1
    
    average_reward = total_reward / reward_count if reward_count > 0 else 0.0
    
    return SessionSummaryResponse(
        total_sessions=total_sessions,
        active_sessions=active_sessions,
        completed_sessions=completed_sessions,
        failed_sessions=failed_sessions,
        total_runtime_hours=total_runtime_hours,
        total_timesteps=total_timesteps,
        average_reward=average_reward
    )


@router.get("/games", response_model=List[GameConfigResponse])
async def list_supported_games():
    """List supported games and their configurations."""
    # This would typically come from a configuration file or database
    games = [
        GameConfigResponse(
            game="breakout",
            display_name="Breakout",
            description="Classic Atari Breakout game - break bricks with a ball and paddle",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=2000000,
            estimated_training_time="2-4 hours"
        ),
        GameConfigResponse(
            game="pong",
            display_name="Pong",
            description="Classic Atari Pong game - tennis-like game with paddles",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=1000000,
            estimated_training_time="1-2 hours"
        ),
        GameConfigResponse(
            game="space_invaders",
            display_name="Space Invaders",
            description="Classic Atari Space Invaders - shoot alien invaders",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=3000000,
            estimated_training_time="3-6 hours"
        ),
        GameConfigResponse(
            game="asteroids",
            display_name="Asteroids",
            description="Classic Atari Asteroids - navigate and destroy asteroids",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=2500000,
            estimated_training_time="2-5 hours"
        ),
        GameConfigResponse(
            game="frogger",
            display_name="Frogger",
            description="Cross roads and rivers while avoiding obstacles",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=2000000,
            estimated_training_time="2-4 hours"
        ),
        GameConfigResponse(
            game="pacman",
            display_name="Pac-Man",
            description="Navigate mazes, eat dots, and avoid ghosts",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=2500000,
            estimated_training_time="3-5 hours"
        ),
        GameConfigResponse(
            game="tetris",
            display_name="Tetris",
            description="Arrange falling blocks to clear lines",
            default_algorithm="ppo",
            supported_algorithms=["ppo", "dqn"],
            default_timesteps=3000000,
            estimated_training_time="4-8 hours"
        )
    ]
    
    return games


@router.get("/algorithms")
async def list_supported_algorithms():
    """List supported ML algorithms and their configurations."""
    algorithms = {
        "ppo": {
            "name": "Proximal Policy Optimization",
            "description": "On-policy algorithm that balances exploration and exploitation",
            "default_params": {
                "learning_rate": 2.5e-4,
                "n_steps": 128,
                "batch_size": 256,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.1,
                "ent_coef": 0.01,
                "vf_coef": 0.5
            },
            "recommended_for": ["breakout", "pong", "space_invaders"]
        },
        "dqn": {
            "name": "Deep Q-Network",
            "description": "Off-policy algorithm using experience replay and target networks",
            "default_params": {
                "learning_rate": 1e-4,
                "buffer_size": 100000,
                "learning_starts": 50000,
                "batch_size": 32,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1
            },
            "recommended_for": ["asteroids", "frogger", "pacman"]
        }
    }
    
    return algorithms
