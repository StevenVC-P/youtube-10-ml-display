"""
Training wrapper for integrating existing ML training pipeline with container management.
"""

import os
import sys
import subprocess
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import psutil

from .resource_monitor import resource_monitor
from .config import settings

logger = logging.getLogger(__name__)


class TrainingWrapper:
    """Wrapper for existing ML training pipeline to work with container management."""
    
    def __init__(self, container_id: str, working_directory: str, config: Dict[str, Any]):
        self.container_id = container_id
        self.working_directory = Path(working_directory)
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
    async def start_training(self) -> bool:
        """Start the training process."""
        try:
            # Prepare the training environment
            await self._prepare_environment()
            
            # Create config file
            config_file = self.working_directory / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f)
            
            # Prepare training command
            cmd = self._build_training_command(config_file)
            
            # Set environment variables
            env = self._prepare_environment_variables()
            
            # Start the training process
            log_file = self.working_directory / "training.log"
            with open(log_file, 'w') as log:
                self.process = subprocess.Popen(
                    cmd,
                    cwd=self.working_directory,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group
                )
            
            # Add to resource monitoring
            resource_monitor.add_process_to_monitoring(self.process.pid)
            
            # Start monitoring task
            self.monitoring_task = asyncio.create_task(self._monitor_training())
            
            logger.info(f"Started training for container {self.container_id} with PID {self.process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training for container {self.container_id}: {e}")
            return False
    
    async def stop_training(self, force: bool = False) -> bool:
        """Stop the training process."""
        if not self.process:
            return True
            
        try:
            # Cancel monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Try graceful shutdown first
            if not force:
                self.process.send_signal(signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process_end()),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    force = True
            
            # Force kill if still running
            if force and self.process.poll() is None:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                await asyncio.sleep(1)
            
            # Remove from resource monitoring
            resource_monitor.remove_process_from_monitoring(self.process.pid)
            
            logger.info(f"Stopped training for container {self.container_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop training for container {self.container_id}: {e}")
            return False
    
    async def _prepare_environment(self):
        """Prepare the training environment."""
        # Create necessary directories
        directories = [
            "models/checkpoints",
            "logs/tb",
            "video/milestones",
            "video/eval",
            "video/render/parts"
        ]
        
        for directory in directories:
            (self.working_directory / directory).mkdir(parents=True, exist_ok=True)
        
        # Copy any necessary files from the main project
        # This would include copying the training scripts, environment wrappers, etc.
        await self._copy_training_files()
    
    async def _copy_training_files(self):
        """Copy necessary training files to the container directory."""
        import shutil
        
        # Get the main project directory (assuming we're in ui/backend)
        main_project_dir = Path(__file__).parent.parent.parent.parent
        
        # Files and directories to copy
        items_to_copy = [
            "training",
            "agents", 
            "envs",
            "conf",
            "tools"
        ]
        
        for item in items_to_copy:
            src = main_project_dir / item
            dst = self.working_directory / item
            
            if src.exists():
                if src.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
    
    def _build_training_command(self, config_file: Path) -> list:
        """Build the training command."""
        # Use the existing training script
        cmd = [
            sys.executable, "-m", "training.train",
            "--config", str(config_file),
            "--container-id", self.container_id
        ]
        
        return cmd
    
    def _prepare_environment_variables(self) -> Dict[str, str]:
        """Prepare environment variables for the training process."""
        env = os.environ.copy()
        
        # Resource limits
        resource_limits = self.config.get('resource_limits', {})
        if 'cpu_cores' in resource_limits:
            env['OMP_NUM_THREADS'] = str(int(resource_limits['cpu_cores']))
        
        if 'memory_gb' in resource_limits:
            env['CONTAINER_MEMORY_LIMIT'] = str(int(resource_limits['memory_gb'] * 1024))
        
        # Container identification
        env['CONTAINER_ID'] = self.container_id
        env['CONTAINER_WORKING_DIR'] = str(self.working_directory)
        
        # GPU configuration
        if settings.gpu_enabled:
            env['CUDA_VISIBLE_DEVICES'] = '0'  # TODO: Implement proper GPU allocation
        else:
            env['CUDA_VISIBLE_DEVICES'] = ''
        
        # Python path
        env['PYTHONPATH'] = str(self.working_directory)
        
        return env
    
    async def _wait_for_process_end(self):
        """Wait for the process to end."""
        while self.process and self.process.poll() is None:
            await asyncio.sleep(0.1)
    
    async def _monitor_training(self):
        """Monitor the training process and update container status."""
        try:
            while self.process and self.process.poll() is None:
                # Check if process is still running
                try:
                    process = psutil.Process(self.process.pid)
                    if not process.is_running():
                        break
                except psutil.NoSuchProcess:
                    break
                
                # TODO: Parse training logs to extract metrics
                # This would involve reading the log file and extracting:
                # - Current timesteps
                # - Current episodes
                # - Current reward
                # - Training progress
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error monitoring training for container {self.container_id}: {e}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics by parsing logs."""
        log_file = self.working_directory / "training.log"
        
        if not log_file.exists():
            return {}
        
        # TODO: Implement log parsing to extract training metrics
        # This would parse the training log to extract:
        # - Current timesteps
        # - Episodes completed
        # - Average reward
        # - Loss values
        # - Training progress
        
        return {
            "timesteps": 0,
            "episodes": 0,
            "reward": 0.0,
            "progress": 0.0
        }
    
    def is_running(self) -> bool:
        """Check if the training process is running."""
        if not self.process:
            return False
        
        return self.process.poll() is None
    
    def get_process_id(self) -> Optional[int]:
        """Get the process ID."""
        return self.process.pid if self.process else None


class TrainingManager:
    """Manages multiple training wrappers."""
    
    def __init__(self):
        self.training_wrappers: Dict[str, TrainingWrapper] = {}
    
    async def start_container_training(self, container_id: str, working_directory: str, config: Dict[str, Any]) -> bool:
        """Start training for a container."""
        if container_id in self.training_wrappers:
            logger.warning(f"Training already exists for container {container_id}")
            return False
        
        wrapper = TrainingWrapper(container_id, working_directory, config)
        success = await wrapper.start_training()
        
        if success:
            self.training_wrappers[container_id] = wrapper
        
        return success
    
    async def stop_container_training(self, container_id: str, force: bool = False) -> bool:
        """Stop training for a container."""
        if container_id not in self.training_wrappers:
            return True
        
        wrapper = self.training_wrappers[container_id]
        success = await wrapper.stop_training(force)
        
        if success:
            del self.training_wrappers[container_id]
        
        return success
    
    def get_training_wrapper(self, container_id: str) -> Optional[TrainingWrapper]:
        """Get training wrapper for a container."""
        return self.training_wrappers.get(container_id)
    
    def get_training_metrics(self, container_id: str) -> Dict[str, Any]:
        """Get training metrics for a container."""
        wrapper = self.training_wrappers.get(container_id)
        if not wrapper:
            return {}
        
        return wrapper.get_training_metrics()
    
    def is_training_running(self, container_id: str) -> bool:
        """Check if training is running for a container."""
        wrapper = self.training_wrappers.get(container_id)
        if not wrapper:
            return False
        
        return wrapper.is_running()


# Global training manager instance
training_manager = TrainingManager()
