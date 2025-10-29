"""
Docker container management for ML training runs.
"""

import docker
import os
import secrets
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ContainerInfo:
    """Information about a Docker container."""
    id: str
    name: str
    image: str
    status: str
    created: datetime
    labels: Dict[str, str]


@dataclass
class ResourceLimits:
    """Resource limits for container creation."""
    cpus: Optional[float] = None
    nano_cpus: Optional[int] = None
    mem_limit: Optional[str] = None
    shm_size: Optional[str] = None
    gpu: str = "auto"  # "auto", "none", "0", "1", "0,1", etc.


class DockerManager:
    """Manages Docker containers for ML training."""
    
    def __init__(self):
        self.client = None
        self._log_streams: Dict[str, threading.Thread] = {}
        self._log_callbacks: Dict[str, List[Callable[[str], None]]] = {}
        self._connect()
    
    def _connect(self):
        """Connect to Docker daemon."""
        try:
            self.client = docker.from_env()
            # Test connection
            self.client.ping()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Docker daemon: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Docker daemon."""
        try:
            if self.client:
                self.client.ping()
                return True
        except:
            pass
        return False
    
    def get_containers(self, all_containers: bool = True) -> List[ContainerInfo]:
        """Get list of containers."""
        if not self.client:
            return []
        
        try:
            containers = self.client.containers.list(all=all_containers)
            result = []
            
            for container in containers:
                # Parse creation time
                created_str = container.attrs['Created']
                created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                
                result.append(ContainerInfo(
                    id=container.id[:12],  # Short ID
                    name=container.name,
                    image=container.image.tags[0] if container.image.tags else container.image.id[:12],
                    status=container.status,
                    created=created,
                    labels=container.labels
                ))
            
            return result
        except Exception:
            return []
    
    def get_training_containers(self) -> List[ContainerInfo]:
        """Get containers created by this app (with retro-ml label)."""
        containers = self.get_containers()
        return [c for c in containers if c.labels.get('retro-ml') == 'true']
    
    def create_container(
        self,
        image: str,
        command: str,
        name: str,
        workdir: str = "/workspace",
        volumes: Optional[List[str]] = None,
        environment: Optional[List[str]] = None,
        resources: Optional[ResourceLimits] = None,
        run_id: str = None
    ) -> str:
        """Create and start a new container."""
        if not self.client:
            raise RuntimeError("Not connected to Docker daemon")
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"run-{secrets.token_hex(4)}"
        
        # Prepare volumes
        volume_dict = {}
        if volumes:
            for volume in volumes:
                parts = volume.split(':')
                if len(parts) >= 2:
                    host_path = os.path.abspath(os.path.expanduser(parts[0]))
                    container_path = parts[1]
                    mode = parts[2] if len(parts) > 2 else 'rw'
                    volume_dict[host_path] = {'bind': container_path, 'mode': mode}
        
        # Prepare environment
        env_dict = {}
        if environment:
            for env_var in environment:
                if '=' in env_var:
                    key, value = env_var.split('=', 1)
                    env_dict[key] = value
        
        # Prepare resource limits
        kwargs = {
            'image': image,
            'command': command,
            'name': name,
            'working_dir': workdir,
            'volumes': volume_dict,
            'environment': env_dict,
            'detach': True,
            'labels': {
                'retro-ml': 'true',
                'run-id': run_id,
                'created-by': 'retro-ml-desktop'
            }
        }
        
        if resources:
            # Handle CPU limits
            if resources.nano_cpus:
                kwargs['nano_cpus'] = resources.nano_cpus
            elif resources.cpus:
                kwargs['nano_cpus'] = int(resources.cpus * 1_000_000_000)
            
            # Handle memory limit
            if resources.mem_limit:
                kwargs['mem_limit'] = resources.mem_limit
            
            # Handle shared memory
            if resources.shm_size:
                kwargs['shm_size'] = resources.shm_size
            
            # Handle GPU
            if resources.gpu and resources.gpu != "none":
                device_requests = []
                if resources.gpu == "auto":
                    device_requests.append(
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    )
                else:
                    # Parse specific GPU IDs (e.g., "0", "1", "0,1")
                    gpu_ids = [id.strip() for id in resources.gpu.split(',')]
                    device_requests.append(
                        docker.types.DeviceRequest(
                            device_ids=gpu_ids,
                            capabilities=[["gpu"]]
                        )
                    )
                kwargs['device_requests'] = device_requests
        
        try:
            container = self.client.containers.run(**kwargs)
            return container.id[:12]
        except docker.errors.ImageNotFound:
            raise RuntimeError(f"Docker image '{image}' not found. Please build or pull the image first.")
        except docker.errors.APIError as e:
            if "nvidia" in str(e).lower() or "gpu" in str(e).lower():
                raise RuntimeError(f"GPU/NVIDIA runtime error: {e}\nEnsure NVIDIA Docker runtime is installed and configured.")
            raise RuntimeError(f"Docker API error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create container: {e}")
    
    def stop_container(self, container_id: str, timeout: int = 30) -> bool:
        """Stop a container."""
        if not self.client:
            return False
        
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=timeout)
            return True
        except Exception:
            return False
    
    def remove_container(self, container_id: str, force: bool = False) -> bool:
        """Remove a container."""
        if not self.client:
            return False
        
        try:
            container = self.client.containers.get(container_id)
            container.remove(force=force)
            
            # Stop log streaming if active
            self._stop_log_stream(container_id)
            
            return True
        except Exception:
            return False
    
    def get_container_logs(self, container_id: str, tail: int = 50) -> str:
        """Get container logs."""
        if not self.client:
            return ""
        
        try:
            container = self.client.containers.get(container_id)
            logs = container.logs(tail=tail, timestamps=True)
            return logs.decode('utf-8', errors='replace')
        except Exception:
            return ""
    
    def start_log_stream(self, container_id: str, callback: Callable[[str], None]):
        """Start streaming logs from a container."""
        if container_id in self._log_streams:
            return  # Already streaming
        
        if container_id not in self._log_callbacks:
            self._log_callbacks[container_id] = []
        self._log_callbacks[container_id].append(callback)
        
        thread = threading.Thread(
            target=self._log_stream_worker,
            args=(container_id,),
            daemon=True
        )
        self._log_streams[container_id] = thread
        thread.start()
    
    def stop_log_stream(self, container_id: str):
        """Stop streaming logs from a container."""
        self._stop_log_stream(container_id)
    
    def _stop_log_stream(self, container_id: str):
        """Internal method to stop log streaming."""
        if container_id in self._log_streams:
            # Thread will stop when container is removed from _log_streams
            del self._log_streams[container_id]
        
        if container_id in self._log_callbacks:
            del self._log_callbacks[container_id]
    
    def _log_stream_worker(self, container_id: str):
        """Worker thread for streaming container logs."""
        if not self.client:
            return
        
        try:
            container = self.client.containers.get(container_id)
            
            # Stream logs
            for log_line in container.logs(stream=True, follow=True, timestamps=True):
                # Check if we should stop streaming
                if container_id not in self._log_streams:
                    break
                
                line = log_line.decode('utf-8', errors='replace').strip()
                
                # Send to all callbacks
                callbacks = self._log_callbacks.get(container_id, [])
                for callback in callbacks:
                    try:
                        callback(line)
                    except Exception:
                        pass  # Don't let callback errors stop streaming
        
        except Exception:
            pass  # Container might have been removed or stopped
        
        finally:
            # Clean up
            if container_id in self._log_streams:
                del self._log_streams[container_id]


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run-{secrets.token_hex(4)}"


def expand_volume_path(path: str) -> str:
    """Expand and normalize a volume path."""
    return os.path.abspath(os.path.expanduser(path))
