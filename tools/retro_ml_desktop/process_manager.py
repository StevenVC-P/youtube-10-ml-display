"""
Direct Python process management for ML training runs.
No Docker required - runs training/train.py directly.
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
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


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

        # Ensure we can find the training script and base config
        self.train_script = self.project_root / "training" / "train.py"
        self.base_config = self.project_root / "conf" / "config.yaml"

        if not self.train_script.exists():
            raise FileNotFoundError(f"Training script not found: {self.train_script}")
        if not self.base_config.exists():
            raise FileNotFoundError(f"Base config not found: {self.base_config}")

        # Load base configuration
        with open(self.base_config, 'r') as f:
            self.base_config_data = yaml.safe_load(f)
    
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
                        process_info.status = "finished"
                    else:
                        process_info.status = "failed"

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
                        print(f"❌ {error_summary}")

                        if recent_logs and len(recent_logs) > 100:
                            print(f"Recent logs from failed process:\n{recent_logs[-500:]}")  # Last 500 chars
                    process_info.process = None

        return list(self.processes.values())

    def get_process_output_paths(self, process_id: str) -> Dict[str, str]:
        """Get the output paths for a specific process."""
        if process_id not in self.processes:
            return {}

        process_info = self.processes[process_id]
        if not hasattr(process_info, 'config_data') or not process_info.config_data:
            return {}

        config = process_info.config_data
        paths = config.get('paths', {})

        # Extract base paths
        videos_milestones = paths.get('videos_milestones', '')
        videos_base = str(Path(videos_milestones).parent) if videos_milestones else ''

        return {
            'videos_base': videos_base,
            'videos_milestones': paths.get('videos_milestones', ''),
            'videos_eval': paths.get('videos_eval', ''),
            'videos_parts': paths.get('videos_parts', ''),
            'models': paths.get('models', ''),
            'logs_tb': paths.get('logs_tb', '')
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
        custom_output_path: str = None
    ) -> str:
        """Create and start a new training process."""

        # Generate run ID if not provided
        if not run_id:
            run_id = f"run-{secrets.token_hex(4)}"

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
                universal_newlines=True,
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
            
            self.processes[run_id] = process_info

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
        if process_id not in self._log_buffers:
            return ""

        # Return the last 2000 characters of logs for parsing
        log_buffer = self._log_buffers[process_id]
        return log_buffer[-2000:] if len(log_buffer) > 2000 else log_buffer
    
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
    verbose: int = 1
) -> bool:
    """
    Generate milestone videos after training completes using saved checkpoints.

    Args:
        config_path: Path to the training config file
        model_dir: Directory containing model checkpoints
        output_dir: Output directory for generated videos
        clip_seconds: Length of each video clip in seconds
        verbose: Verbosity level

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
            verbose=verbose
        )

        # Generate videos
        generated_videos = generator.generate_all_videos()

        if verbose >= 1:
            print(f"[PostVideo] Generated {len(generated_videos)} videos")
            for video_path in generated_videos:
                print(f"[PostVideo] 📹 {video_path.name}")

        return len(generated_videos) > 0

    except Exception as e:
        if verbose >= 1:
            print(f"[PostVideo] Error generating videos: {e}")
        return False
