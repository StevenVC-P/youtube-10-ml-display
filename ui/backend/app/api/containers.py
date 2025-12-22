"""
API endpoints for container management.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from ..core.container_manager import (
    container_manager, 
    TrainingConfig, 
    ResourceSpec, 
    Container,
    ContainerStatus
)

router = APIRouter(prefix="/api/containers", tags=["containers"])


# Request/Response Models
class ResourceSpecRequest(BaseModel):
    cpu_cores: float = Field(default=2.0, ge=0.1, le=32.0)
    memory_gb: float = Field(default=4.0, ge=0.5, le=128.0)
    gpu_memory_gb: Optional[float] = Field(default=None, ge=0.1, le=80.0)
    disk_space_gb: float = Field(default=10.0, ge=1.0, le=1000.0)


class TrainingConfigRequest(BaseModel):
    game: str = Field(..., pattern="^[a-zA-Z0-9_-]+$")
    algorithm: str = Field(default="ppo", pattern="^(ppo|dqn)$")
    total_timesteps: int = Field(default=1000000, ge=1000, le=100000000)
    vec_envs: int = Field(default=4, ge=1, le=16)
    learning_rate: float = Field(default=2.5e-4, ge=1e-6, le=1e-1)
    checkpoint_every_sec: int = Field(default=60, ge=10, le=3600)
    video_recording: bool = Field(default=True)
    fast_mode: bool = Field(default=False)
    resource_limits: ResourceSpecRequest = Field(default_factory=ResourceSpecRequest)


class CreateContainerRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    config: TrainingConfigRequest


class ContainerResponse(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    stopped_at: Optional[datetime]
    process_id: Optional[int]
    error_message: Optional[str]
    config: dict
    
    @classmethod
    def from_container(cls, container: Container) -> "ContainerResponse":
        return cls(
            id=container.id,
            name=container.name,
            status=container.status.value,
            created_at=container.created_at,
            started_at=container.started_at,
            stopped_at=container.stopped_at,
            process_id=container.process_id,
            error_message=container.error_message,
            config=container.config.to_dict()
        )


class ResourceUsageResponse(BaseModel):
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    num_threads: int
    gpu_processes: List[dict]


class SystemResourcesResponse(BaseModel):
    total: dict
    available: dict
    allocated: dict


# API Endpoints
@router.get("/", response_model=List[ContainerResponse])
async def list_containers():
    """List all containers."""
    containers = container_manager.list_containers()
    return [ContainerResponse.from_container(c) for c in containers]


@router.post("/", response_model=ContainerResponse)
async def create_container(request: CreateContainerRequest):
    """Create a new training container."""
    try:
        # Convert request to internal models
        resource_spec = ResourceSpec(
            cpu_cores=request.config.resource_limits.cpu_cores,
            memory_gb=request.config.resource_limits.memory_gb,
            gpu_memory_gb=request.config.resource_limits.gpu_memory_gb,
            disk_space_gb=request.config.resource_limits.disk_space_gb
        )
        
        training_config = TrainingConfig(
            game=request.config.game,
            algorithm=request.config.algorithm,
            total_timesteps=request.config.total_timesteps,
            vec_envs=request.config.vec_envs,
            learning_rate=request.config.learning_rate,
            checkpoint_every_sec=request.config.checkpoint_every_sec,
            video_recording=request.config.video_recording,
            fast_mode=request.config.fast_mode,
            resource_limits=resource_spec
        )
        
        container = container_manager.create_container(request.name, training_config)
        return ContainerResponse.from_container(container)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create container: {str(e)}")


@router.get("/{container_id}", response_model=ContainerResponse)
async def get_container(container_id: str):
    """Get container details."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    return ContainerResponse.from_container(container)


@router.post("/{container_id}/start")
async def start_container(container_id: str, background_tasks: BackgroundTasks):
    """Start a container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    if container.status not in [ContainerStatus.CREATED, ContainerStatus.STOPPED]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot start container in {container.status.value} state"
        )
    
    # Start container in background
    background_tasks.add_task(container_manager.start_container, container_id)
    
    return {"message": "Container start initiated", "container_id": container_id}


@router.post("/{container_id}/stop")
async def stop_container(container_id: str, background_tasks: BackgroundTasks, force: bool = False):
    """Stop a container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    if container.status != ContainerStatus.RUNNING:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot stop container in {container.status.value} state"
        )
    
    # Stop container in background
    background_tasks.add_task(container_manager.stop_container, container_id, force)
    
    return {"message": "Container stop initiated", "container_id": container_id}


@router.delete("/{container_id}")
async def delete_container(container_id: str, background_tasks: BackgroundTasks):
    """Delete a container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    # Delete container in background
    background_tasks.add_task(container_manager.delete_container, container_id)
    
    return {"message": "Container deletion initiated", "container_id": container_id}


@router.get("/{container_id}/metrics", response_model=Optional[ResourceUsageResponse])
async def get_container_metrics(container_id: str):
    """Get resource usage metrics for a container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    metrics = container_manager.get_container_metrics(container_id)
    if not metrics:
        return None
    
    return ResourceUsageResponse(
        cpu_percent=metrics.cpu_percent,
        memory_used_gb=metrics.memory_used_gb,
        memory_percent=metrics.memory_percent,
        num_threads=metrics.num_threads,
        gpu_processes=metrics.gpu_processes
    )


@router.get("/{container_id}/logs")
async def get_container_logs(container_id: str, lines: int = 100):
    """Get container logs."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    if not container.log_file:
        return {"logs": "No log file available"}
    
    try:
        import subprocess
        result = subprocess.run(
            ["tail", "-n", str(lines), container.log_file],
            capture_output=True,
            text=True
        )
        return {"logs": result.stdout}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {str(e)}")


@router.get("/system/resources", response_model=SystemResourcesResponse)
async def get_system_resources():
    """Get system resource information."""
    total_resources = container_manager.get_system_resources()
    available_resources = container_manager.get_available_resources()
    
    # Calculate allocated resources
    allocated_resources = ResourceSpec(
        cpu_cores=total_resources.cpu_cores - available_resources.cpu_cores,
        memory_gb=total_resources.memory_gb - available_resources.memory_gb,
        gpu_memory_gb=(
            (total_resources.gpu_memory_gb or 0) - (available_resources.gpu_memory_gb or 0)
            if total_resources.gpu_memory_gb else None
        ),
        disk_space_gb=total_resources.disk_space_gb - available_resources.disk_space_gb
    )
    
    return SystemResourcesResponse(
        total=total_resources.to_dict(),
        available=available_resources.to_dict(),
        allocated=allocated_resources.to_dict()
    )
