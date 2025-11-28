"""
Direct Python process management for ML training runs.
No Docker required - runs training/train.py directly.

Now integrated with Event Bus and Experiment Manager for real-time updates.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import psutil
import secrets
import yaml
import tempfile
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import new Phase 1 systems
from tools.retro_ml_desktop.metric_event_bus import get_event_bus, EventTypes
from tools.retro_ml_desktop.experiment_manager import ExperimentManager, Experiment

logger = logging.getLogger(__name__)


@dataclass
class ProcessInfo:
    """Information about a training process."""
    id: str
    name: str
    command: List[str]
    status: str
    created: datetime
    pid: Optional[int] = None
    process: Optional[subprocess.Popen] = None
    config_data: Optional[Dict] = None


@dataclass
class ResourceLimits:
    """Resource limits for process creation."""
    cpu_affinity: Optional[List[int]] = None  # CPU cores to use
    memory_limit_gb: Optional[float] = None   # Memory limit in GB
    priority: str = "normal"  # "low", "normal", "high"
    gpu_id: Optional[str] = None  # GPU ID to use


class ProcessManager:
    """Manages Python training processes directly."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.processes: Dict[str, ProcessInfo] = {}
        self._log_streams: Dict[str, threading.Thread] = {}
        self._log_callbacks: Dict[str, List[Callable[[str], None]]] = {}
        self._log_buffers: Dict[str, str] = {}  # Store logs for progress parsing
        self._temp_configs: Dict[str, str] = {}  # Track temp config files
        self.ml_database = None  # Will be set by the main application

        # Phase 1: Event Bus and Experiment Manager integration
        self.event_bus = get_event_bus()
        self.experiment_manager = None  # Will be initialized when database is set
        self._experiment_map: Dict[str, str] = {}  # Maps process_id -> experiment_id

        logger.info("ProcessManager initialized with Event Bus integration")

        # Find training script and base config - handle frozen executables
        if getattr(sys, 'frozen', False):
            # Running as frozen executable - use _MEIPASS
            base_path = Path(sys._MEIPASS)
        else:
            # Running as normal Python script
            base_path = self.project_root

        self.train_script = base_path / "training" / "train.py"
        self.base_config = base_path / "conf" / "config.yaml"

        if not self.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {self.train_script}")
        if not self.base_config.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config}")

        # Load base configuration
        with open(self.base_config, 'r') as f:
            self.base_config_data = yaml.safe_load(f)

    def set_database(self, database):
        """Set the database and initialize Experiment Manager."""
        self.ml_database = database
        if database:
            self.experiment_manager = ExperimentManager(database)
            logger.info("Experiment Manager initialized")

    def get_processes(self) -> List[ProcessInfo]:
        """Get list of all training processes."""
        # Update process statuses (but preserve paused status)
        for process_info in self.processes.values():
            if process_info.process:
                if process_info.process.poll() is None:
                    # Process is alive - check if it's actually paused
                    if process_info.status != "paused":
                        # Only update to running if not explicitly paused
                        try:
                            proc = psutil.Process(process_info.process.pid)
                            if proc.status() == psutil.STATUS_STOPPED:
                                process_info.status = "paused"
                            else:
                                process_info.status = "running"
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            process_info.status = "running"  # Fallback
                else:
                    # Process has exited - check exit code to determine if it was successful
                    exit_code = process_info.process.poll()
                    if exit_code == 0:
                        # Check if we've already marked this as finished (to avoid duplicate video generation)
                        was_already_finished = process_info.status == "finished"
                        process_info.status = "finished"

                        # Mark run as inactive in database
                        if self.ml_database:
                            self.ml_database.mark_run_inactive(process_info.id, 'completed')

                        # Phase 1: Update experiment status and publish event
                        if not was_already_finished and self.experiment_manager and process_info.id in self._experiment_map:
                            experiment_id = self._experiment_map[process_info.id]
                            try:
                                # Update experiment status to completed
                                self.experiment_manager.update_experiment_status(
                                    experiment_id,
                                    "completed"
                                )

                                # Publish TRAINING_COMPLETE event
                                self.event_bus.publish(EventTypes.TRAINING_COMPLETE, {
                                    'experiment_id': experiment_id,
                                    'run_id': process_info.id,
                                    'total_time': (datetime.now() - process_info.created).total_seconds()
                                })

                                logger.debug(f"Training completed for experiment {experiment_id}")

                            except Exception as e:
                                logger.error(f"Failed to update experiment {experiment_id}: {e}")

                        # AUTO-GENERATE VIDEO: If training just completed and has target_hours set
                        # This makes video generation automatic instead of manual
                        if not was_already_finished and hasattr(process_info, 'target_hours') and process_info.target_hours:
                            self._auto_generate_video(process_info)
                    else:
                        process_info.status = "failed"
                        # Mark run as inactive in database
                        if self.ml_database:
                            self.ml_database.mark_run_inactive(process_info.id, 'failed')

                        # Enhanced error analysis for CUDA/GPU issues
                        recent_logs = self.get_recent_logs(process_info.id)

                        # Check for CUDA-related errors
                        cuda_error_detected = False
                        error_summary = f"Process failed with exit code: {exit_code}"

                        if recent_logs:
                            cuda_keywords = ['cuda', 'gpu', 'memory allocation', 'device', 'kernel', 'nvidia', 'out of memory']
                            log_lower = recent_logs.lower()
                            cuda_error_detected = any(keyword in log_lower for keyword in cuda_keywords)

                            if cuda_error_detected:
                                error_summary = f"CUDA/GPU Error (exit code {exit_code})"
                                # Extract the most relevant error line
                                lines = recent_logs.split('\n')
                                error_lines = [line.strip() for line in lines if line.strip() and any(keyword in line.lower() for keyword in ['error', 'failed', 'cuda', 'memory', 'allocation'])]
                                if error_lines:
                                    error_summary += f" - {error_lines[-1][:100]}..."

                        process_info.error_message = error_summary
                        print(f"[ERROR] {error_summary}")

                        # Phase 1: Update experiment status and publish event
                        if self.experiment_manager and process_info.id in self._experiment_map:
                            experiment_id = self._experiment_map[process_info.id]
                            try:
                                # Update experiment status to failed
                                self.experiment_manager.update_experiment_status(
                                    experiment_id,
                                    "failed"
                                )

                                # Publish TRAINING_FAILED event
                                self.event_bus.publish(EventTypes.TRAINING_FAILED, {
                                    'experiment_id': experiment_id,
                                    'run_id': process_info.id,
                                    'error': error_summary,
                                    'exit_code': exit_code
                                })

                                logger.debug(f"Training failed for experiment {experiment_id}")

                            except Exception as e:
                                logger.error(f"Failed to update experiment {experiment_id}: {e}")

                        if recent_logs and len(recent_logs) > 100:
                            print(f"Recent logs from failed process:\n{recent_logs[-500:]}")  # Last 500 chars
                    process_info.process = None

        return list(self.processes.values())

    def get_process_output_paths(self, process_id: str) -> Dict[str, str]:
        """Get the output paths for a specific process.

        Returns absolute paths resolved relative to the project root.
        """
        if process_id not in self.processes:
            return {}

        process_info = self.processes[process_id]
        if not hasattr(process_info, 'config_data') or not process_info.config_data:
            return {}

        config = process_info.config_data
        paths = config.get('paths', {})

        # Helper function to resolve paths to absolute
        def resolve_path(path_str: str) -> str:
            if not path_str:
                return ''
            path = Path(path_str)
            # If already absolute, return as-is; otherwise resolve relative to project root
            if path.is_absolute():
                return str(path)
            else:
                return str(self.project_root / path)

        # Extract and resolve all paths to absolute paths
        videos_milestones = resolve_path(paths.get('videos_milestones', ''))
        videos_base = str(Path(videos_milestones).parent) if videos_milestones else ''

        return {
            'videos_base': videos_base,
            'videos_milestones': videos_milestones,
            'videos_eval': resolve_path(paths.get('videos_eval', '')),
            'videos_parts': resolve_path(paths.get('videos_parts', '')),
            'models': resolve_path(paths.get('models', '')),
            'logs_tb': resolve_path(paths.get('logs_tb', ''))
        }

    def create_process(
        self,
        game: str,
        algorithm: str,
        run_id: str = None,
        total_timesteps: int = 4000000,
        vec_envs: int = 16,
        save_freq: int = 200000,
        resources: Optional[ResourceLimits] = None,
        extra_args: Optional[List[str]] = None,
        custom_output_path: str = None,
        resume_from_checkpoint: str = None,
        target_hours: float = None
    ) -> str:
        """
        Create and start a new training process.

        Args:
            game: Game environment ID
            algorithm: Algorithm to use (ppo, dqn, etc.)
            run_id: Run ID (generated if not provided)
            total_timesteps: Total timesteps to train
            vec_envs: Number of vectorized environments
            save_freq: Checkpoint save frequency
            resources: Resource limits
            extra_args: Extra command-line arguments
            custom_output_path: Custom output path for videos
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            target_hours: Target video length in hours (optional)

        Returns:
            Process ID

        Raises:
            ValueError: If run_id is already active
        """

        # Generate run ID if not provided
        if not run_id:
            run_id = f"run-{secrets.token_hex(4)}"

        # Validate that run_id is not already active
        if self.ml_database and self.ml_database.is_run_active(run_id):
            raise ValueError(
                f"Run ID '{run_id}' is already active. "
                "Cannot start the same run multiple times to prevent data corruption."
            )

        # Create custom config for this training run
        config_data = self.base_config_data.copy()

        # Update config with training parameters
        config_data['game']['env_id'] = game
        config_data['train']['algo'] = algorithm
        config_data['train']['total_timesteps'] = total_timesteps
        config_data['train']['vec_envs'] = vec_envs
        config_data['train']['save_freq'] = save_freq

        # CRITICAL FIX: Disable video recording during training for desktop app
        # Set milestone_clip_seconds to 0 to skip video recording and use checkpoints instead
        # Videos can be generated post-training using PostTrainingVideoGenerator
        config_data['recording']['milestone_clip_seconds'] = 0   # 0 = skip video recording, save checkpoints
        config_data['recording']['eval_clip_seconds'] = 0       # 0 = skip eval videos too

        # Save target video length for post-training video generation
        if target_hours is not None:
            config_data['render']['target_hours'] = target_hours

        # Update paths for this run (use custom output path if provided)
        if custom_output_path:
            # Use custom path for videos, keep logs and models in project
            base_output_path = Path(custom_output_path) / run_id
            config_data['paths']['videos_milestones'] = str(base_output_path / "milestones")
            config_data['paths']['videos_eval'] = str(base_output_path / "eval")
            config_data['paths']['videos_parts'] = str(base_output_path / "parts")
            config_data['paths']['logs_tb'] = f"logs/tb/{run_id}"
            config_data['paths']['models'] = f"models/checkpoints/{run_id}"

            # Create output directories (custom path for videos)
            output_dirs = [
                base_output_path / "milestones",
                base_output_path / "eval",
                base_output_path / "parts",
                self.project_root / "logs" / "tb" / run_id,
                self.project_root / "models" / "checkpoints" / run_id
            ]
        else:
            # Use default project paths
            config_data['paths']['videos_milestones'] = f"outputs/{run_id}/milestones"
            config_data['paths']['videos_eval'] = f"outputs/{run_id}/eval"
            config_data['paths']['videos_parts'] = f"outputs/{run_id}/parts"
            config_data['paths']['logs_tb'] = f"logs/tb/{run_id}"
            config_data['paths']['models'] = f"models/checkpoints/{run_id}"

            # Create output directories (default paths)
            output_dirs = [
                self.project_root / "outputs" / run_id / "milestones",
                self.project_root / "outputs" / run_id / "eval",
                self.project_root / "outputs" / run_id / "parts",
                self.project_root / "logs" / "tb" / run_id,
                self.project_root / "models" / "checkpoints" / run_id
            ]

        for dir_path in output_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Save run metadata to a JSON file in the checkpoint directory
        metadata_path = self.project_root / "models" / "checkpoints" / run_id / "run_metadata.json"
        metadata = {
            'run_id': run_id,
            'game': game,
            'algorithm': algorithm,
            'total_timesteps': total_timesteps,
            'target_hours': target_hours,
            'created': datetime.now().isoformat(),
            'custom_output_path': custom_output_path
        }
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)

        # Create temporary config file
        temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            prefix=f'config_{run_id}_'
        )

        with temp_config as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Store temp config path for cleanup
        self._temp_configs[run_id] = temp_config.name

        # Prepare command - much simpler now!
        command = [
            sys.executable,  # Use current Python interpreter
            str(self.train_script),
            "--config", temp_config.name,
            "--verbose", "1"
        ]

        # Add checkpoint path if resuming
        if resume_from_checkpoint:
            command.extend(["--resume-from", resume_from_checkpoint])

        # Add extra arguments
        if extra_args:
            command.extend(extra_args)
        
        # Prepare environment variables
        env = os.environ.copy()
        
        # Set GPU if specified
        if resources and resources.gpu_id:
            env["CUDA_VISIBLE_DEVICES"] = resources.gpu_id
        
        # Set OMP threads based on CPU affinity
        if resources and resources.cpu_affinity:
            env["OMP_NUM_THREADS"] = str(len(resources.cpu_affinity))
        else:
            env["OMP_NUM_THREADS"] = "2"  # Default
        
        try:
            # Start the process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                encoding='utf-8',  # Force UTF-8 encoding on Windows
                errors='replace',  # Replace encoding errors instead of crashing
                env=env,
                cwd=str(self.project_root)
            )
            
            # Apply resource limits
            if resources:
                self._apply_resource_limits(process.pid, resources)
            
            # Create process info
            process_name = f"{algorithm}-{game}-{run_id}"
            process_info = ProcessInfo(
                id=run_id,
                name=process_name,
                command=command,
                status="running",
                created=datetime.now(),
                pid=process.pid,
                process=process,
                config_data=config_data
            )

            # Store target_hours for automatic video generation
            if target_hours is not None:
                process_info.target_hours = target_hours

            # Store total_timesteps for progress tracking
            process_info.total_timesteps = total_timesteps

            self.processes[run_id] = process_info

            # Phase 1: Create experiment and publish TRAINING_STARTED event
            if self.experiment_manager:
                try:
                    # Determine preset based on target_hours
                    preset = "quick" if target_hours and target_hours < 1 else "standard"
                    if target_hours and target_hours >= 8:
                        preset = "epic"

                    # Create experiment
                    experiment = self.experiment_manager.create_experiment(
                        name=f"{game} - {algorithm}",
                        game=game,
                        algorithm=algorithm,
                        preset=preset,
                        video_length_hours=target_hours or 4.0,
                        notes=f"Training run {run_id}"
                    )

                    # Map process to experiment
                    self._experiment_map[run_id] = experiment.id

                    # Update experiment status to running
                    self.experiment_manager.update_experiment_status(
                        experiment.id,
                        "running"
                    )

                    # Publish TRAINING_STARTED event
                    self.event_bus.publish(EventTypes.TRAINING_STARTED, {
                        'experiment_id': experiment.id,
                        'run_id': run_id,
                        'game': game,
                        'algorithm': algorithm,
                        'preset': preset,
                        'total_timesteps': total_timesteps,
                        'target_hours': target_hours
                    })

                    print(f"[Experiment] Created: {experiment.id} ({preset} preset, {target_hours or 4.0}h target)")
                    logger.debug(f"Created experiment {experiment.id} for run {run_id}")

                except Exception as e:
                    logger.error(f"Failed to create experiment for run {run_id}: {e}")

            # Start log streaming automatically for progress tracking
            self._start_log_stream_for_process(run_id)

            return run_id

        except Exception as e:
            raise RuntimeError(f"Failed to start training process: {e}")
    
    def _apply_resource_limits(self, pid: int, resources: ResourceLimits):
        """Apply resource limits to a process."""
        try:
            proc = psutil.Process(pid)

            # Set CPU affinity
            if resources.cpu_affinity:
                try:
                    proc.cpu_affinity(resources.cpu_affinity)
                except (psutil.AccessDenied, OSError):
                    pass  # Might not have permission or invalid cores

            # Set process priority (Windows uses different values)
            try:
                if resources.priority == "low":
                    if sys.platform == "win32":
                        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    else:
                        proc.nice(10)  # Lower priority on Unix
                elif resources.priority == "high":
                    if sys.platform == "win32":
                        proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    else:
                        proc.nice(-5)  # Higher priority on Unix
            except (psutil.AccessDenied, OSError):
                pass  # Might not have permission to change priority

            # Memory limit is harder to enforce directly in Python
            # We'll just monitor it and warn if exceeded

        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass  # Process might have finished or we don't have permissions
    
    def stop_process(self, process_id: str, timeout: int = 30) -> bool:
        """Stop a training process."""
        if process_id not in self.processes:
            return False

        process_info = self.processes[process_id]
        if not process_info.process:
            return False

        try:
            # If process is paused, resume it first so it can handle termination properly
            if process_info.status == "paused":
                try:
                    proc = psutil.Process(process_info.process.pid)
                    proc.resume()
                    # Give it a moment to resume
                    import time
                    time.sleep(0.5)
                except:
                    pass

            # Try graceful shutdown first
            process_info.process.terminate()

            # Wait for graceful shutdown
            try:
                process_info.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                try:
                    # Kill all child processes first
                    proc = psutil.Process(process_info.process.pid)
                    children = proc.children(recursive=True)
                    for child in children:
                        try:
                            child.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass

                    # Then kill the main process
                    process_info.process.kill()
                    process_info.process.wait()
                except:
                    pass

            process_info.status = "stopped"
            process_info.process = None

            # Mark run as inactive in database
            if self.ml_database:
                self.ml_database.mark_run_inactive(process_id, 'stopped')

            # Phase 1: Update experiment and publish event
            if self.experiment_manager and process_id in self._experiment_map:
                experiment_id = self._experiment_map[process_id]
                try:
                    # Update experiment status to stopped
                    self.experiment_manager.update_experiment_status(
                        experiment_id,
                        "paused"  # Using paused status for stopped
                    )

                    # Publish TRAINING_STOPPED event
                    self.event_bus.publish(EventTypes.TRAINING_STOPPED, {
                        'experiment_id': experiment_id,
                        'run_id': process_id
                    })

                    logger.debug(f"Training stopped for experiment {experiment_id}")

                except Exception as e:
                    logger.error(f"Failed to update experiment {experiment_id}: {e}")

            # Stop log streaming
            self._stop_log_stream(process_id)

            print(f"Successfully stopped process {process_id}")
            return True

        except Exception as e:
            print(f"Error stopping process {process_id}: {e}")
            return False

    def pause_process(self, process_id: str) -> bool:
        """Pause a running process (suspend)."""
        if process_id not in self.processes:
            return False

        process_info = self.processes[process_id]
        if not process_info.process or process_info.process.poll() is not None:
            return False

        # Don't pause if already paused
        if process_info.status == "paused":
            return True

        try:
            proc = psutil.Process(process_info.process.pid)

            # Suspend the main process and all its children
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.suspend()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            proc.suspend()
            process_info.status = "paused"

            # Phase 1: Publish TRAINING_PAUSED event
            if self.experiment_manager and process_id in self._experiment_map:
                experiment_id = self._experiment_map[process_id]
                try:
                    self.event_bus.publish(EventTypes.TRAINING_PAUSED, {
                        'experiment_id': experiment_id,
                        'run_id': process_id
                    })
                    logger.debug(f"Training paused for experiment {experiment_id}")
                except Exception as e:
                    logger.error(f"Failed to publish pause event for {experiment_id}: {e}")

            print(f"Successfully paused process {process_id} (PID: {proc.pid})")
            return True
        except Exception as e:
            print(f"Error pausing process {process_id}: {e}")
            return False

    def resume_process(self, process_id: str) -> bool:
        """Resume a paused process."""
        if process_id not in self.processes:
            return False

        process_info = self.processes[process_id]
        if not process_info.process or process_info.process.poll() is not None:
            return False

        # Don't resume if not paused
        if process_info.status != "paused":
            return True

        try:
            proc = psutil.Process(process_info.process.pid)

            # Resume the main process and all its children
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.resume()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            proc.resume()
            process_info.status = "running"

            # Phase 1: Publish TRAINING_RESUMED event
            if self.experiment_manager and process_id in self._experiment_map:
                experiment_id = self._experiment_map[process_id]
                try:
                    self.event_bus.publish(EventTypes.TRAINING_RESUMED, {
                        'experiment_id': experiment_id,
                        'run_id': process_id
                    })
                    logger.debug(f"Training resumed for experiment {experiment_id}")
                except Exception as e:
                    logger.error(f"Failed to publish resume event for {experiment_id}: {e}")

            print(f"Successfully resumed process {process_id} (PID: {proc.pid})")
            return True
        except Exception as e:
            print(f"Error resuming process {process_id}: {e}")
            return False

    def clear_training_data(self, process_id: str = None, clear_all: bool = False) -> bool:
        """Clear training data (models, logs, videos) for a specific run or all runs."""
        try:
            if clear_all:
                # Clear all training data
                paths_to_clear = [
                    self.project_root / "outputs",
                    self.project_root / "models" / "checkpoints",
                    self.project_root / "logs" / "tb"
                ]

                for path in paths_to_clear:
                    if path.exists():
                        import shutil
                        shutil.rmtree(path)
                        path.mkdir(parents=True, exist_ok=True)

                print("Cleared all training data")
                return True

            elif process_id:
                # Clear data for specific process
                paths_to_clear = [
                    self.project_root / "outputs" / process_id,
                    self.project_root / "models" / "checkpoints" / process_id,
                    self.project_root / "logs" / "tb" / process_id
                ]

                for path in paths_to_clear:
                    if path.exists():
                        import shutil
                        shutil.rmtree(path)

                print(f"Cleared training data for {process_id}")
                return True

            return False

        except Exception as e:
            print(f"Error clearing training data: {e}")
            return False

    def remove_process(self, process_id: str) -> bool:
        """Remove a process from tracking."""
        if process_id not in self.processes:
            return False

        # Stop the process first if it's running
        self.stop_process(process_id)

        # Clean up temporary config file
        if process_id in self._temp_configs:
            try:
                os.unlink(self._temp_configs[process_id])
            except OSError:
                pass  # File might already be deleted
            del self._temp_configs[process_id]

        # Remove from tracking
        del self.processes[process_id]

        return True
    
    def get_process_logs(self, process_id: str, tail_lines: int = 50) -> str:
        """Get recent logs from a process (not implemented for live processes)."""
        # For live processes, logs are streamed via callbacks
        # This could be enhanced to read from log files if needed
        return f"Live log streaming active for process {process_id}"

    def get_recent_logs(self, process_id: str) -> str:
        """Get recent logs from the log buffer for progress parsing."""
        import logging

        if process_id not in self._log_buffers:
            logging.info(f"No log buffer found for process {process_id}")
            return ""

        # Return the last 2000 characters of logs for parsing
        log_buffer = self._log_buffers[process_id]
        buffer_length = len(log_buffer)
        result = log_buffer[-2000:] if buffer_length > 2000 else log_buffer

        logging.info(f"get_recent_logs for {process_id}: buffer_length={buffer_length}, returning {len(result)} chars")
        if result:
            # Show first 200 chars of logs for debugging
            preview = result[:200].replace('\n', '\\n')
            logging.info(f"Log preview: {preview}...")

        return result
    
    def start_log_stream(self, process_id: str, callback: Callable[[str], None]):
        """Start streaming logs from a process."""
        if process_id in self._log_streams:
            return  # Already streaming
        
        if process_id not in self._log_callbacks:
            self._log_callbacks[process_id] = []
        self._log_callbacks[process_id].append(callback)
        
        thread = threading.Thread(
            target=self._log_stream_worker,
            args=(process_id,),
            daemon=True
        )
        self._log_streams[process_id] = thread
        thread.start()
    
    def stop_log_stream(self, process_id: str):
        """Stop streaming logs from a process."""
        self._stop_log_stream(process_id)
    
    def _stop_log_stream(self, process_id: str):
        """Internal method to stop log streaming."""
        if process_id in self._log_streams:
            del self._log_streams[process_id]

        if process_id in self._log_callbacks:
            del self._log_callbacks[process_id]

        if process_id in self._log_buffers:
            del self._log_buffers[process_id]

    def _start_log_stream_for_process(self, process_id: str):
        """Start log streaming for a process to capture stdout for progress tracking."""
        if process_id not in self.processes:
            return

        # Initialize log buffer
        self._log_buffers[process_id] = ""

        # Start log streaming thread
        thread = threading.Thread(
            target=self._log_stream_worker,
            args=(process_id,),
            daemon=True
        )
        self._log_streams[process_id] = thread
        thread.start()
    
    def _log_stream_worker(self, process_id: str):
        """Worker thread for streaming process logs."""
        if process_id not in self.processes:
            return
        
        process_info = self.processes[process_id]
        if not process_info.process or not process_info.process.stdout:
            return
        
        try:
            # Stream stdout line by line
            for line in iter(process_info.process.stdout.readline, ''):
                # Check if we should stop streaming
                if process_id not in self._log_streams:
                    break
                
                line = line.strip()
                if line:  # Only send non-empty lines
                    # Store in log buffer for progress parsing
                    if process_id not in self._log_buffers:
                        self._log_buffers[process_id] = ""

                    self._log_buffers[process_id] += line + "\n"

                    # Keep buffer size manageable (last 10000 characters)
                    if len(self._log_buffers[process_id]) > 10000:
                        self._log_buffers[process_id] = self._log_buffers[process_id][-8000:]

                    # Parse metrics from log line and publish TRAINING_PROGRESS event
                    self._parse_and_publish_metrics(process_id, line)

                    # Send to all callbacks
                    callbacks = self._log_callbacks.get(process_id, [])
                    for callback in callbacks:
                        try:
                            callback(line)
                        except Exception:
                            pass  # Don't let callback errors stop streaming
        
        except Exception:
            pass  # Process might have been terminated
        
        finally:
            # Clean up
            if process_id in self._log_streams:
                del self._log_streams[process_id]


    def _parse_and_publish_metrics(self, process_id: str, log_line: str):
        """
        Parse training metrics from log line and publish TRAINING_PROGRESS event.

        Extracts metrics from stable-baselines3 PPO/DQN log output.
        Example log format:
        -----------------------------------------
        | time/              |                |
        |    fps             | 150            |
        |    iterations      | 100            |
        |    time_elapsed    | 68             |
        |    total_timesteps | 102400         |
        | train/             |                |
        |    approx_kl       | 0.009258024    |
        |    clip_fraction   | 0.0833         |
        |    entropy_loss    | -0.619         |
        |    explained_variance | 0.912       |
        |    learning_rate   | 0.0003         |
        |    loss            | 14.1           |
        |    policy_gradient_loss | -0.0134   |
        |    value_loss      | 41.4           |
        | rollout/           |                |
        |    ep_len_mean     | 1440           |
        |    ep_rew_mean     | 189.5          |
        """
        import re

        # Only parse if this process has an experiment mapped
        if process_id not in self._experiment_map:
            return

        experiment_id = self._experiment_map[process_id]

        # Parse key metrics using regex
        metrics = {}

        # FPS (training speed)
        if 'fps' in log_line.lower():
            match = re.search(r'fps\s+\|\s+(\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['fps'] = float(match.group(1))

        # Total timesteps (progress)
        if 'total_timesteps' in log_line.lower():
            match = re.search(r'total_timesteps\s+\|\s+(\d+)', log_line, re.IGNORECASE)
            if match:
                metrics['timesteps'] = int(match.group(1))

        # Mean episode reward
        if 'ep_rew_mean' in log_line.lower():
            match = re.search(r'ep_rew_mean\s+\|\s+(-?\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['reward_mean'] = float(match.group(1))

        # Policy loss
        if 'policy_gradient_loss' in log_line.lower():
            match = re.search(r'policy_gradient_loss\s+\|\s+(-?\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['policy_loss'] = float(match.group(1))

        # Value loss
        if 'value_loss' in log_line.lower():
            match = re.search(r'value_loss\s+\|\s+(-?\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['value_loss'] = float(match.group(1))

        # Episode length
        if 'ep_len_mean' in log_line.lower():
            match = re.search(r'ep_len_mean\s+\|\s+(-?\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['episode_length'] = float(match.group(1))

        # Time elapsed
        if 'time_elapsed' in log_line.lower():
            match = re.search(r'time_elapsed\s+\|\s+(\d+(?:\.\d+)?)', log_line, re.IGNORECASE)
            if match:
                metrics['elapsed_time'] = float(match.group(1))

        # If we found any metrics, publish TRAINING_PROGRESS event
        if metrics:
            # Get process info for total timesteps
            process_info = self.processes.get(process_id)
            if process_info:
                # Calculate progress percentage if we have timesteps
                progress_data = {
                    'experiment_id': experiment_id,
                    'run_id': process_id,
                    'metrics': metrics
                }

                # Add progress calculation if we have timesteps
                if 'timesteps' in metrics and hasattr(process_info, 'total_timesteps'):
                    total_ts = getattr(process_info, 'total_timesteps', 0)
                    if total_ts > 0:
                        progress_pct = (metrics['timesteps'] / total_ts) * 100
                        progress_data['progress_pct'] = progress_pct
                        progress_data['timestep'] = metrics['timesteps']
                        progress_data['total_timesteps'] = total_ts

                        # Estimate time remaining
                        if 'elapsed_time' in metrics and metrics['elapsed_time'] > 0:
                            remaining_ts = total_ts - metrics['timesteps']
                            time_per_ts = metrics['elapsed_time'] / metrics['timesteps']
                            eta = remaining_ts * time_per_ts
                            progress_data['estimated_time_remaining'] = eta

                # Publish event
                try:
                    self.event_bus.publish(EventTypes.TRAINING_PROGRESS, progress_data)
                    logger.debug(f"Published TRAINING_PROGRESS for {experiment_id}: {metrics}")
                except Exception as e:
                    logger.error(f"Failed to publish TRAINING_PROGRESS: {e}")

    def _auto_generate_video(self, process_info: ProcessInfo):
        """
        Automatically generate video after training completes.
        This runs in a background thread to avoid blocking the UI.
        """
        import threading

        def generate_video_thread():
            try:
                run_id = process_info.id
                target_hours = process_info.target_hours

                # Convert target hours to seconds
                total_seconds = int(target_hours * 3600)

                print(f"[AutoVideo] Generating {target_hours}h video...")

                # Phase 1: Publish VIDEO_GENERATION_STARTED event
                experiment_id = self._experiment_map.get(run_id)
                if experiment_id:
                    self.event_bus.publish(EventTypes.VIDEO_GENERATION_STARTED, {
                        'experiment_id': experiment_id,
                        'run_id': run_id,
                        'target_duration': total_seconds
                    })
                    logger.debug(f"Video generation started for experiment {experiment_id}")

                # Get checkpoint directory
                checkpoint_dir = self.project_root / "models" / "checkpoints" / run_id / "milestones"

                if not checkpoint_dir.exists():
                    print(f"[AutoVideo] No checkpoints found at {checkpoint_dir}")
                    return

                # Get output directory from config or use default
                config_data = process_info.config_data
                custom_output_path = config_data.get('paths', {}).get('videos_milestones')

                if custom_output_path:
                    # Use the parent directory of videos_milestones
                    output_dir = Path(custom_output_path).parent
                else:
                    # Use default video directory
                    output_dir = self.project_root / "video" / "post_training"

                output_dir.mkdir(parents=True, exist_ok=True)

                # Get config path
                config_path = self._temp_configs.get(run_id)
                if not config_path or not Path(config_path).exists():
                    config_path = str(self.project_root / "conf" / "config.yaml")

                # Generate the video
                success = generate_post_training_videos(
                    config_path=config_path,
                    model_dir=str(checkpoint_dir),
                    output_dir=str(output_dir),
                    clip_seconds=90,  # Not used when total_seconds is provided
                    total_seconds=total_seconds,
                    verbose=1
                )

                if success:
                    # Phase 1: Register video artifact and publish event
                    if experiment_id and self.ml_database:
                        try:
                            # Find the generated video file
                            video_files = list(output_dir.glob("*.mp4"))
                            if video_files:
                                video_path = video_files[-1]  # Get the most recent video
                                video_size_mb = video_path.stat().st_size / (1024 * 1024)

                                # Register video artifact in database
                                video_id = f"video_{run_id}_{int(time.time())}"
                                self.ml_database.add_video_artifact(
                                    video_id=video_id,
                                    experiment_id=experiment_id,
                                    path=str(video_path),
                                    video_type="hour",  # Post-training continuous video
                                    metadata={
                                        'duration': total_seconds,
                                        'size_mb': video_size_mb,
                                        'target_hours': target_hours
                                    }
                                )

                                # Publish VIDEO_GENERATED event
                                self.event_bus.publish(EventTypes.VIDEO_GENERATED, {
                                    'experiment_id': experiment_id,
                                    'run_id': run_id,
                                    'video_path': str(video_path),
                                    'video_type': 'hour',
                                    'duration': total_seconds,
                                    'size_mb': video_size_mb
                                })

                                print(f"[AutoVideo] Complete! Video saved and registered in database ({video_size_mb:.1f} MB)")
                                logger.debug(f"Video artifact registered for experiment {experiment_id}")

                        except Exception as e:
                            logger.error(f"Failed to register video artifact: {e}")
                            print(f"[AutoVideo] Complete! (Note: video file created but database registration failed)")

                else:
                    print(f"[AutoVideo] Failed - check logs for details")

                    # Publish failure event
                    if experiment_id:
                        self.event_bus.publish(EventTypes.VIDEO_GENERATION_FAILED, {
                            'experiment_id': experiment_id,
                            'run_id': run_id
                        })

            except Exception as e:
                print(f"[AutoVideo] Error: {e}")
                logger.error(f"Video generation error: {e}", exc_info=True)

        # Start video generation in background thread
        thread = threading.Thread(target=generate_video_thread, daemon=True)
        thread.start()

def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run-{secrets.token_hex(4)}"


@dataclass
class CPUResource:
    """CPU resource information."""
    core_id: int
    is_physical: bool
    frequency_mhz: float
    usage_percent: float
    available: bool

@dataclass
class GPUResource:
    """GPU resource information."""
    gpu_id: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_percent: float
    temperature_c: float
    available: bool
    driver_version: str = ""

def get_detailed_cpu_info() -> List[CPUResource]:
    """Get detailed information about available CPU cores."""
    cpu_resources = []

    try:
        logical_cores = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        cpu_freq = psutil.cpu_freq()
        cpu_percent_per_core = psutil.cpu_percent(percpu=True, interval=0.1)

        base_freq = cpu_freq.current if cpu_freq else 2000.0

        for i in range(logical_cores):
            # Determine if this is a physical core (simplified heuristic)
            is_physical = i < physical_cores

            # Get per-core usage
            usage = cpu_percent_per_core[i] if i < len(cpu_percent_per_core) else 0.0

            # Consider core available if usage is below 80%
            available = usage < 80.0

            cpu_resources.append(CPUResource(
                core_id=i,
                is_physical=is_physical,
                frequency_mhz=base_freq,
                usage_percent=usage,
                available=available
            ))

    except Exception as e:
        print(f"Error getting CPU info: {e}")
        # Fallback to basic info
        for i in range(psutil.cpu_count() or 4):
            cpu_resources.append(CPUResource(
                core_id=i,
                is_physical=True,
                frequency_mhz=2000.0,
                usage_percent=0.0,
                available=True
            ))

    return cpu_resources

def get_detailed_gpu_info() -> List[GPUResource]:
    """Get detailed information about available GPUs."""
    gpu_resources = []

    # Try pynvml first (more detailed)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = mem_info.total / (1024**2)  # MB
            memory_used = mem_info.used / (1024**2)   # MB
            memory_free = memory_total - memory_used

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = util.gpu

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0.0

            # Driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                driver_version = "Unknown"

            # Consider GPU available if utilization < 90% and memory usage < 90%
            available = utilization < 90.0 and (memory_used / memory_total) < 0.9

            gpu_resources.append(GPUResource(
                gpu_id=i,
                name=name,
                memory_total_mb=memory_total,
                memory_used_mb=memory_used,
                memory_free_mb=memory_free,
                utilization_percent=utilization,
                temperature_c=temp,
                available=available,
                driver_version=driver_version
            ))

    except Exception:
        # Fallback to GPUtil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()

            for gpu in gpus:
                memory_total = gpu.memoryTotal
                memory_used = gpu.memoryUsed
                memory_free = gpu.memoryFree
                utilization = gpu.load * 100

                available = utilization < 90.0 and (memory_used / memory_total) < 0.9

                gpu_resources.append(GPUResource(
                    gpu_id=gpu.id,
                    name=gpu.name,
                    memory_total_mb=memory_total,
                    memory_used_mb=memory_used,
                    memory_free_mb=memory_free,
                    utilization_percent=utilization,
                    temperature_c=getattr(gpu, 'temperature', 0.0),
                    available=available,
                    driver_version="Unknown"
                ))

        except Exception:
            pass  # No GPUs available

    return gpu_resources

def get_available_cpus() -> List[int]:
    """Get list of available CPU cores (legacy function)."""
    cpu_info = get_detailed_cpu_info()
    return [cpu.core_id for cpu in cpu_info if cpu.available]

def get_available_gpus() -> List[str]:
    """Get list of available GPU IDs (legacy function)."""
    gpu_info = get_detailed_gpu_info()
    return [str(gpu.gpu_id) for gpu in gpu_info if gpu.available]

def get_recommended_resources() -> Dict[str, any]:
    """Get recommended resource allocation based on system capabilities."""
    cpu_info = get_detailed_cpu_info()
    gpu_info = get_detailed_gpu_info()

    # CPU recommendations
    available_cpus = [cpu for cpu in cpu_info if cpu.available]
    physical_cpus = [cpu for cpu in available_cpus if cpu.is_physical]

    # Recommend using 75% of available physical cores, minimum 2
    recommended_cpu_cores = max(2, int(len(physical_cpus) * 0.75))

    # GPU recommendations
    available_gpus = [gpu for gpu in gpu_info if gpu.available]

    if available_gpus:
        # Recommend GPU with most free memory
        best_gpu = max(available_gpus, key=lambda g: g.memory_free_mb)
        recommended_gpu = best_gpu.gpu_id
    else:
        recommended_gpu = None

    # Memory recommendations (leave 25% free)
    memory_info = psutil.virtual_memory()
    recommended_memory_gb = int((memory_info.total / (1024**3)) * 0.75)

    return {
        'cpu_cores': recommended_cpu_cores,
        'gpu_id': recommended_gpu,
        'memory_gb': recommended_memory_gb,
        'total_cpus': len(cpu_info),
        'available_cpus': len(available_cpus),
        'total_gpus': len(gpu_info),
        'available_gpus': len(available_gpus)
    }


def generate_post_training_videos(
    config_path: str,
    model_dir: str,
    output_dir: str = "video/post_training",
    clip_seconds: int = 10,
    verbose: int = 1,
    total_seconds: int = None,
    db=None,
    run_id: str = None
) -> bool:
    """
    Generate videos after training completes using saved checkpoints.

    Args:
        config_path: Path to the training config file
        model_dir: Directory containing model checkpoints
        output_dir: Output directory for generated videos
        clip_seconds: Length of each video clip in seconds (used if total_seconds is None)
        verbose: Verbosity level
        total_seconds: If provided, generates a single continuous video of this length
        db: MetricsDatabase instance for progress tracking (optional)
        run_id: Training run ID for database tracking (optional)

    Returns:
        True if video generation succeeded, False otherwise
    """
    try:
        # Import the post-training video generator
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        from training.post_training_video_generator import PostTrainingVideoGenerator

        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Create video generator
        generator = PostTrainingVideoGenerator(
            config=config,
            model_dir=Path(model_dir),
            output_dir=Path(output_dir),
            clip_seconds=clip_seconds,
            fps=30,
            verbose=verbose,
            db=db,
            run_id=run_id
        )

        # Generate videos
        if total_seconds is not None:
            # Generate single continuous video
            video_path = generator.generate_continuous_video(total_seconds)
            if video_path:
                if verbose >= 1:
                    print(f"[PostVideo] Generated continuous video: {video_path.name}")
                return True
            else:
                return False
        else:
            # Generate separate milestone videos
            generated_videos = generator.generate_all_videos()

            if verbose >= 1:
                print(f"[PostVideo] Generated {len(generated_videos)} videos")
                for video_path in generated_videos:
                    print(f"[PostVideo] 📹 {video_path.name}")

            return len(generated_videos) > 0

    except Exception as e:
        if verbose >= 1:
            print(f"[PostVideo] Error generating videos: {e}")
            import traceback
            traceback.print_exc()
        return False
