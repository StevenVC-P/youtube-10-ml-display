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
                
                await asyncio.sleep(10)  # Check every 10 seconds

            # Process finished
            exit_code = self.process.poll()
            if exit_code == 0:
                logger.info(f"Training completed successfully for container {self.container_id}")
                
                # Check for Fast Mode auto-render
                training_config = self.config.get('train', {})
                fast_mode = self.config.get('recording', {}).get('milestone_clip_seconds', 10) == 0
                
                # Note: We infer fast_mode from the 0 clip duration set in container_manager
                if fast_mode:
                    logger.info(f"Fast Mode detected for {self.container_id}. Triggering post-training render...")
                    await self._trigger_post_training_render()
            else:
                logger.error(f"Training failed for container {self.container_id} with exit code {exit_code}")
                
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

    async def _trigger_post_training_render(self):
        """Trigger post-training video generation for Fast Mode."""
        try:
            # Construct path to video generator script relative to backend
            # Backend: ui/backend/app/core/training_wrapper.py
            # Root:    ui/backend/app/core/../../../..  -> .
            # Script:  training/post_training_video_generator.py
            
            # We are running inside the container working directory context
            # So we can just call python -m training.post_training_video_generator
            
            # Use 'total_timesteps' to calculate hours for video length
            # Assuming 100k steps ~ 1 hour (rough estimate, but we need seconds)
            # Actually, standard epic is 10h. Let's look for hours in config or default to 10h.
            # For simplicity, we'll generate a continuous video for the full duration.
            
            # Calculate total seconds based on Total Timesteps / Est SPS (e.g. 1000) ??
            # Or just use the --total-seconds arg if we can infer it.
            # Let's rely on finding 'total_timesteps' and converting to estimated seconds?
            # Or better: check config for 'hours'.
            # TrainingConfig has 'total_timesteps'.
            # Standard: 1M steps for ~1-2 hours.
            
            # Safer bet: Generate based on checkpoints found.
            # But post_training_video_generator requires --total-seconds for continuous mode.
            # Let's hardcode a reasonable default or try to get it from config if passed.
            # The UI config sets 'total_timesteps'.
            
            # Let's assume 36000 seconds (10 hours) for now as default 'Epic' length?
            # Or just generate for whatever length exists.
            
            # Wait, post_training_video_generator needs --config.
            config_file = self.working_directory / "config.yaml"
            model_dir = self.working_directory / "models/checkpoints"
            output_dir = self.working_directory / "video/merged_epic"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Infer duration from total_timesteps 
            # (Assuming standard Atari 4-frame skip, so 1 step = 4 frames. 60 FPS game speed.
            # Realtime: 60 frames/sec. 15 steps/sec game time.
            # Total seconds = total_timesteps / 15 ?)
            # Actually, let's just use 36000 (10h) as intended for "Fast Mode" epic training.
            
            total_timesteps = self.config.get('train', {}).get('total_timesteps', 1000000)
            # Approx 10M steps = 10 hours for standard PPO?
            # Let's use 10 hours (36000s) as safeguard, ensuring complete coverage.
            total_seconds = 36000 
            
            cmd = [
                sys.executable, "-m", "training.post_training_video_generator",
                "--config", str(config_file),
                "--model-dir", str(model_dir),
                "--output-dir", str(output_dir),
                "--total-seconds", str(total_seconds),
                "--verbose", "1"
            ]
            
            log_file = self.working_directory / "rendering.log"
            
            with open(log_file, 'w') as log:
                logger.info(f"Starting post-render: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    cwd=self.working_directory,
                    stdout=log,
                    stderr=subprocess.STDOUT
                )
                
                # We won't block the main loop, but we could add it to a tracking list if needed.
                # For now, fire and forget (logging to file).
                
        except Exception as e:
            logger.error(f"Failed to trigger post-render: {e}")


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
