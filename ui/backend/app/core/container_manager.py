"""
Container management system for ML training sessions.
Handles creation, lifecycle, and resource allocation of training containers.
"""

import asyncio
import os
import signal
import subprocess
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import json
import psutil

from .resource_monitor import resource_monitor, ResourceMetrics, ProcessMetrics
from .training_wrapper import training_manager

logger = logging.getLogger(__name__)


class ContainerStatus(Enum):
    """Container lifecycle states."""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DELETED = "deleted"


@dataclass
class ResourceSpec:
    """Resource allocation specification for a container."""
    cpu_cores: float = 2.0
    memory_gb: float = 4.0
    gpu_memory_gb: Optional[float] = None
    disk_space_gb: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for ML training session."""
    game: str
    algorithm: str = "ppo"
    total_timesteps: int = 1000000
    vec_envs: int = 4
    learning_rate: float = 2.5e-4
    checkpoint_every_sec: int = 60
    video_recording: bool = True
    fast_mode: bool = False
    resource_limits: ResourceSpec = None
    
    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = ResourceSpec()
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['resource_limits'] = self.resource_limits.to_dict()
        return data
    
    def to_yaml_config(self) -> Dict[str, Any]:
        """Convert to format compatible with existing config.yaml."""
        return {
            'project_name': f"container-{self.game}-{self.algorithm}",
            'seed': 42,
            'game': {
                'env_id': f"{self.game.title()}NoFrameskip-v4"
            },
            'train': {
                'algo': self.algorithm,
                'total_timesteps': self.total_timesteps,
                'vec_envs': self.vec_envs,
                'learning_rate': self.learning_rate,
                'checkpoint_every_sec': self.checkpoint_every_sec
            },
            'recording': {
                'enabled': self.video_recording and not self.fast_mode,
                'milestone_clip_seconds': 0 if self.fast_mode else 10
            }
        }


@dataclass
class Container:
    """ML training container representation."""
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
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['config'] = self.config.to_dict()
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['stopped_at'] = self.stopped_at.isoformat() if self.stopped_at else None
        return data


class ResourceAllocator:
    """Manages resource allocation across containers."""
    
    def __init__(self):
        self.allocated_resources: Dict[str, ResourceSpec] = {}
        self.system_resources = self._detect_system_resources()
        
    def _detect_system_resources(self) -> ResourceSpec:
        """Detect available system resources."""
        # CPU cores
        cpu_cores = float(psutil.cpu_count(logical=True))
        
        # Memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # GPU memory (if available)
        gpu_memory_gb = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass
            
        # Disk space (root partition)
        disk = psutil.disk_usage('/')
        disk_space_gb = disk.total / (1024**3)
        
        return ResourceSpec(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            disk_space_gb=disk_space_gb
        )
    
    def can_allocate(self, resources: ResourceSpec) -> bool:
        """Check if resources can be allocated."""
        # Calculate currently allocated resources
        total_allocated = ResourceSpec()
        for allocated in self.allocated_resources.values():
            total_allocated.cpu_cores += allocated.cpu_cores
            total_allocated.memory_gb += allocated.memory_gb
            if allocated.gpu_memory_gb and total_allocated.gpu_memory_gb:
                total_allocated.gpu_memory_gb += allocated.gpu_memory_gb
            total_allocated.disk_space_gb += allocated.disk_space_gb
        
        # Check if new allocation would exceed system resources
        if (total_allocated.cpu_cores + resources.cpu_cores > self.system_resources.cpu_cores or
            total_allocated.memory_gb + resources.memory_gb > self.system_resources.memory_gb * 0.9 or  # Leave 10% buffer
            total_allocated.disk_space_gb + resources.disk_space_gb > self.system_resources.disk_space_gb * 0.9):
            return False
            
        # Check GPU memory if both are specified
        if (resources.gpu_memory_gb and self.system_resources.gpu_memory_gb and
            total_allocated.gpu_memory_gb and
            total_allocated.gpu_memory_gb + resources.gpu_memory_gb > self.system_resources.gpu_memory_gb * 0.9):
            return False
            
        return True
    
    def allocate(self, container_id: str, resources: ResourceSpec) -> bool:
        """Allocate resources to a container."""
        if not self.can_allocate(resources):
            return False
            
        self.allocated_resources[container_id] = resources
        return True
    
    def deallocate(self, container_id: str) -> bool:
        """Deallocate resources from a container."""
        if container_id in self.allocated_resources:
            del self.allocated_resources[container_id]
            return True
        return False
    
    def get_available_resources(self) -> ResourceSpec:
        """Get currently available resources."""
        total_allocated = ResourceSpec()
        for allocated in self.allocated_resources.values():
            total_allocated.cpu_cores += allocated.cpu_cores
            total_allocated.memory_gb += allocated.memory_gb
            if allocated.gpu_memory_gb and total_allocated.gpu_memory_gb:
                total_allocated.gpu_memory_gb += allocated.gpu_memory_gb
            total_allocated.disk_space_gb += allocated.disk_space_gb
        
        return ResourceSpec(
            cpu_cores=max(0, self.system_resources.cpu_cores - total_allocated.cpu_cores),
            memory_gb=max(0, self.system_resources.memory_gb - total_allocated.memory_gb),
            gpu_memory_gb=(max(0, self.system_resources.gpu_memory_gb - (total_allocated.gpu_memory_gb or 0))
                          if self.system_resources.gpu_memory_gb else None),
            disk_space_gb=max(0, self.system_resources.disk_space_gb - total_allocated.disk_space_gb)
        )


class ContainerManager:
    """Main container management service."""
    
    def __init__(self, base_directory: str = "./containers"):
        self.containers: Dict[str, Container] = {}
        self.resource_allocator = ResourceAllocator()
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True)
        
    def create_container(self, name: str, config: TrainingConfig) -> Container:
        """Create a new training container."""
        container_id = str(uuid.uuid4())
        
        # Check resource allocation
        if not self.resource_allocator.can_allocate(config.resource_limits):
            raise ValueError("Insufficient resources to create container")
        
        # Create container
        container = Container(
            id=container_id,
            name=name,
            config=config,
            status=ContainerStatus.CREATED,
            created_at=datetime.now()
        )
        
        # Allocate resources
        self.resource_allocator.allocate(container_id, config.resource_limits)
        
        # Create working directory
        container_dir = self.base_directory / container_id
        container_dir.mkdir(exist_ok=True)
        container.working_directory = str(container_dir)
        
        # Create config file
        config_file = container_dir / "config.yaml"
        with open(config_file, 'w') as f:
            import yaml
            yaml.dump(config.to_yaml_config(), f)
        
        # Create log file
        container.log_file = str(container_dir / "training.log")
        
        self.containers[container_id] = container
        logger.info(f"Created container {container_id} ({name})")
        
        return container
    
    async def start_container(self, container_id: str) -> bool:
        """Start a training container."""
        if container_id not in self.containers:
            return False
            
        container = self.containers[container_id]
        
        if container.status not in [ContainerStatus.CREATED, ContainerStatus.STOPPED]:
            return False
        
        try:
            container.status = ContainerStatus.STARTING

            # Start training using the training wrapper
            success = await training_manager.start_container_training(
                container_id,
                container.working_directory,
                container.config.to_yaml_config()
            )

            if success:
                # Get the process ID from the training wrapper
                wrapper = training_manager.get_training_wrapper(container_id)
                if wrapper:
                    container.process_id = wrapper.get_process_id()

                container.started_at = datetime.now()
                container.status = ContainerStatus.RUNNING

                logger.info(f"Started container {container_id} with PID {container.process_id}")
                return True
            else:
                container.status = ContainerStatus.ERROR
                container.error_message = "Failed to start training process"
                return False
            
        except Exception as e:
            container.status = ContainerStatus.ERROR
            container.error_message = str(e)
            logger.error(f"Failed to start container {container_id}: {e}")
            return False
    
    async def stop_container(self, container_id: str, force: bool = False) -> bool:
        """Stop a training container."""
        if container_id not in self.containers:
            return False
            
        container = self.containers[container_id]
        
        if container.status != ContainerStatus.RUNNING:
            return False
        
        try:
            container.status = ContainerStatus.STOPPING

            # Stop training using the training wrapper
            success = await training_manager.stop_container_training(container_id, force)

            if success:
                container.stopped_at = datetime.now()
                container.status = ContainerStatus.STOPPED
                container.process_id = None

                logger.info(f"Stopped container {container_id}")
                return True
            else:
                container.status = ContainerStatus.ERROR
                container.error_message = "Failed to stop training process"
                return False
            
        except Exception as e:
            container.status = ContainerStatus.ERROR
            container.error_message = str(e)
            logger.error(f"Failed to stop container {container_id}: {e}")
            return False
    
    async def delete_container(self, container_id: str) -> bool:
        """Delete a container and clean up resources."""
        if container_id not in self.containers:
            return False
            
        container = self.containers[container_id]
        
        # Stop container if running
        if container.status == ContainerStatus.RUNNING:
            await self.stop_container(container_id, force=True)
        
        try:
            # Deallocate resources
            self.resource_allocator.deallocate(container_id)
            
            # Clean up working directory
            if container.working_directory:
                import shutil
                shutil.rmtree(container.working_directory, ignore_errors=True)
            
            # Remove from containers
            del self.containers[container_id]
            
            logger.info(f"Deleted container {container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete container {container_id}: {e}")
            return False
    
    def get_container(self, container_id: str) -> Optional[Container]:
        """Get container by ID."""
        return self.containers.get(container_id)
    
    def list_containers(self) -> List[Container]:
        """List all containers."""
        return list(self.containers.values())
    
    def get_container_metrics(self, container_id: str) -> Optional[ProcessMetrics]:
        """Get resource metrics for a container."""
        container = self.get_container(container_id)
        if not container or not container.process_id:
            return None
            
        return resource_monitor.get_current_process_metrics(container.process_id)
    
    def get_system_resources(self) -> ResourceSpec:
        """Get total system resources."""
        return self.resource_allocator.system_resources
    
    def get_available_resources(self) -> ResourceSpec:
        """Get available system resources."""
        return self.resource_allocator.get_available_resources()


# Global container manager instance
container_manager = ContainerManager()
