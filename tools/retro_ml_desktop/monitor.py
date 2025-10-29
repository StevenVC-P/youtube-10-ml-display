"""
System monitoring utilities for CPU, RAM, and GPU metrics.
"""

import psutil
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False


@dataclass
class CPUInfo:
    """CPU information and metrics."""
    percent: float
    frequency: float
    logical_cores: int
    physical_cores: int


@dataclass
class MemoryInfo:
    """Memory information and metrics."""
    used_gb: float
    total_gb: float
    percent: float
    available_gb: float


@dataclass
class GPUInfo:
    """GPU information and metrics."""
    id: int
    name: str
    load_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    temperature: float


@dataclass
class SystemMetrics:
    """Complete system metrics snapshot."""
    cpu: CPUInfo
    memory: MemoryInfo
    gpus: List[GPUInfo]
    timestamp: float


class SystemMonitor:
    """Real-time system monitoring with callback support."""
    
    def __init__(self, update_interval: float = 2.0):
        self.update_interval = update_interval
        self._running = False
        self._thread = None
        self._callbacks: List[Callable[[SystemMetrics], None]] = []
        self._latest_metrics: Optional[SystemMetrics] = None
    
    def add_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add a callback to be called when metrics are updated."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[SystemMetrics], None]):
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent metrics snapshot."""
        return self._latest_metrics
    
    def get_cpu_info(self) -> CPUInfo:
        """Get current CPU information."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        frequency = cpu_freq.current if cpu_freq else 0.0
        
        return CPUInfo(
            percent=cpu_percent,
            frequency=frequency,
            logical_cores=psutil.cpu_count(logical=True),
            physical_cores=psutil.cpu_count(logical=False)
        )
    
    def get_memory_info(self) -> MemoryInfo:
        """Get current memory information."""
        memory = psutil.virtual_memory()
        
        return MemoryInfo(
            used_gb=memory.used / (1024**3),
            total_gb=memory.total / (1024**3),
            percent=memory.percent,
            available_gb=memory.available / (1024**3)
        )
    
    def get_gpu_info(self) -> List[GPUInfo]:
        """Get current GPU information."""
        gpus = []
        
        if GPUTIL_AVAILABLE:
            try:
                gpu_list = GPUtil.getGPUs()
                for gpu in gpu_list:
                    gpus.append(GPUInfo(
                        id=gpu.id,
                        name=gpu.name,
                        load_percent=gpu.load * 100,
                        memory_used_mb=gpu.memoryUsed,
                        memory_total_mb=gpu.memoryTotal,
                        memory_percent=(gpu.memoryUsed / gpu.memoryTotal) * 100 if gpu.memoryTotal > 0 else 0,
                        temperature=gpu.temperature if hasattr(gpu, 'temperature') else 0.0
                    ))
            except Exception:
                pass  # GPU monitoring failed, return empty list
        
        elif PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    
                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    except:
                        temp = 0.0
                    
                    gpus.append(GPUInfo(
                        id=i,
                        name=name,
                        load_percent=util.gpu,
                        memory_used_mb=mem_info.used / (1024**2),
                        memory_total_mb=mem_info.total / (1024**2),
                        memory_percent=(mem_info.used / mem_info.total) * 100,
                        temperature=temp
                    ))
            except Exception:
                pass  # GPU monitoring failed, return empty list
        
        return gpus
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get a complete snapshot of current system metrics."""
        return SystemMetrics(
            cpu=self.get_cpu_info(),
            memory=self.get_memory_info(),
            gpus=self.get_gpu_info(),
            timestamp=time.time()
        )
    
    def start_monitoring(self):
        """Start continuous monitoring in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self._running:
            try:
                metrics = self.get_current_metrics()
                self._latest_metrics = metrics
                
                # Call all registered callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception:
                        pass  # Don't let callback errors stop monitoring
                
                time.sleep(self.update_interval)
            except Exception:
                time.sleep(self.update_interval)  # Continue monitoring even if there's an error


def format_bytes(bytes_value: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def get_gpu_status_message() -> str:
    """Get a status message about GPU availability."""
    if PYNVML_AVAILABLE or GPUTIL_AVAILABLE:
        monitor = SystemMonitor()
        gpus = monitor.get_gpu_info()
        if gpus:
            return f"{len(gpus)} GPU(s) detected"
        else:
            return "No GPUs detected"
    else:
        return "GPU monitoring unavailable (install nvidia-ml-py or GPUtil)"
