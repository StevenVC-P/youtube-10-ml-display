"""
GPU Monitoring Module

Provides real-time GPU metrics monitoring for NVIDIA GPUs using PyTorch's
built-in CUDA functions (which use pynvml under the hood).

Features:
- GPU detection and information
- Real-time utilization monitoring
- VRAM usage tracking
- Temperature monitoring
- Power draw monitoring (if supported)
- Background monitoring thread
- Event bus integration
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Callable, List

import torch


logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU device information."""
    index: int
    name: str
    total_memory_gb: float
    cuda_capability: tuple
    is_available: bool


@dataclass
class GPUMetrics:
    """Real-time GPU metrics."""
    timestamp: float
    gpu_index: int
    utilization_percent: int  # 0-100
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float  # 0-100
    temperature_c: Optional[int] = None  # Celsius
    power_draw_w: Optional[float] = None  # Watts
    clock_speed_mhz: Optional[int] = None  # MHz


class GPUMonitor:
    """
    GPU monitoring service with background thread.
    
    Monitors NVIDIA GPU metrics and publishes updates via callback.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize GPU monitor.
        
        Args:
            update_interval: Seconds between metric updates (default: 1.0)
        """
        self.update_interval = update_interval
        self.is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[GPUMetrics], None]] = []
        self._lock = threading.Lock()
        
        # Detect GPU on initialization
        self.gpu_info = self._detect_gpu()
        
    def _detect_gpu(self) -> Optional[GPUInfo]:
        """Detect GPU and get device information."""
        try:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available - no GPU detected")
                return None
            
            # Get first GPU (index 0)
            device_index = 0
            name = torch.cuda.get_device_name(device_index)
            total_memory = torch.cuda.get_device_properties(device_index).total_memory
            total_memory_gb = total_memory / (1024 ** 3)  # Convert bytes to GB
            cuda_capability = torch.cuda.get_device_capability(device_index)
            
            gpu_info = GPUInfo(
                index=device_index,
                name=name,
                total_memory_gb=total_memory_gb,
                cuda_capability=cuda_capability,
                is_available=True
            )
            
            logger.info(f"GPU detected: {name} ({total_memory_gb:.1f} GB)")
            return gpu_info
            
        except Exception as e:
            logger.error(f"Failed to detect GPU: {e}")
            return None
    
    def get_current_metrics(self) -> Optional[GPUMetrics]:
        """
        Get current GPU metrics snapshot.
        
        Returns:
            GPUMetrics if GPU is available, None otherwise
        """
        if not self.gpu_info or not self.gpu_info.is_available:
            return None
        
        try:
            device_index = self.gpu_info.index
            
            # Get memory info
            memory_allocated = torch.cuda.memory_allocated(device_index)
            memory_used_gb = memory_allocated / (1024 ** 3)
            memory_percent = (memory_used_gb / self.gpu_info.total_memory_gb) * 100
            
            # Get utilization (requires pynvml, available through torch.cuda)
            utilization = 0
            temperature = None
            power_draw = None
            clock_speed = None
            
            try:
                utilization = torch.cuda.utilization(device_index)
            except Exception as e:
                logger.debug(f"Could not get GPU utilization: {e}")
            
            try:
                temperature = torch.cuda.temperature(device_index)
            except Exception as e:
                logger.debug(f"Could not get GPU temperature: {e}")
            
            try:
                power_draw_mw = torch.cuda.power_draw(device_index)
                power_draw = power_draw_mw / 1000.0  # Convert mW to W
            except Exception as e:
                logger.debug(f"Could not get GPU power draw: {e}")
            
            try:
                clock_speed = torch.cuda.clock_rate(device_index)
            except Exception as e:
                logger.debug(f"Could not get GPU clock speed: {e}")
            
            return GPUMetrics(
                timestamp=time.time(),
                gpu_index=device_index,
                utilization_percent=utilization,
                memory_used_gb=memory_used_gb,
                memory_total_gb=self.gpu_info.total_memory_gb,
                memory_percent=memory_percent,
                temperature_c=temperature,
                power_draw_w=power_draw,
                clock_speed_mhz=clock_speed
            )
            
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {e}")
            return None

    def add_callback(self, callback: Callable[[GPUMetrics], None]):
        """
        Add a callback to be called when metrics are updated.

        Args:
            callback: Function that takes GPUMetrics as argument
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.debug(f"Added GPU metrics callback: {callback.__name__}")

    def remove_callback(self, callback: Callable[[GPUMetrics], None]):
        """
        Remove a callback.

        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.debug(f"Removed GPU metrics callback: {callback.__name__}")

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info("GPU monitoring thread started")

        while self.is_running:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()

                if metrics:
                    # Call all registered callbacks
                    with self._lock:
                        for callback in self._callbacks:
                            try:
                                callback(metrics)
                            except Exception as e:
                                logger.error(f"Error in GPU metrics callback {callback.__name__}: {e}")

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.update_interval)

        logger.info("GPU monitoring thread stopped")

    def start(self):
        """Start background monitoring thread."""
        if self.is_running:
            logger.warning("GPU monitor is already running")
            return

        if not self.gpu_info or not self.gpu_info.is_available:
            logger.warning("Cannot start GPU monitor - no GPU available")
            return

        self.is_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="GPUMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("GPU monitor started")

    def stop(self):
        """Stop background monitoring thread."""
        if not self.is_running:
            return

        self.is_running = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

        logger.info("GPU monitor stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Global GPU monitor instance (singleton pattern)
_gpu_monitor_instance: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """
    Get the global GPU monitor instance (singleton).

    Returns:
        GPUMonitor instance
    """
    global _gpu_monitor_instance

    if _gpu_monitor_instance is None:
        _gpu_monitor_instance = GPUMonitor()

    return _gpu_monitor_instance

