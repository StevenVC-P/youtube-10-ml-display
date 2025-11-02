"""
Resource monitoring system for ML training containers.
Tracks CPU, GPU, memory, and disk usage in real-time.
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import psutil
import logging
from datetime import datetime, timedelta

try:
    import torch
    import GPUtil
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    torch = None
    GPUtil = None

logger = logging.getLogger(__name__)


@dataclass
class ResourceMetrics:
    """Resource usage metrics for a container or system."""
    timestamp: datetime
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    gpu_count: int = 0
    gpu_usage_percent: List[float] = None
    gpu_memory_used_gb: List[float] = None
    gpu_memory_total_gb: List[float] = None
    gpu_memory_percent: List[float] = None
    gpu_temperature: List[float] = None
    
    def __post_init__(self):
        if self.gpu_usage_percent is None:
            self.gpu_usage_percent = []
        if self.gpu_memory_used_gb is None:
            self.gpu_memory_used_gb = []
        if self.gpu_memory_total_gb is None:
            self.gpu_memory_total_gb = []
        if self.gpu_memory_percent is None:
            self.gpu_memory_percent = []
        if self.gpu_temperature is None:
            self.gpu_temperature = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ProcessMetrics:
    """Resource usage metrics for a specific process."""
    pid: int
    name: str
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    num_threads: int
    status: str
    create_time: datetime
    gpu_processes: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.gpu_processes is None:
            self.gpu_processes = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['create_time'] = self.create_time.isoformat()
        return data


class SystemResourceMonitor:
    """Monitor system-wide resource usage."""
    
    def __init__(self):
        self.last_cpu_times = None
        self.monitoring = False
        
    def get_system_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        
        # Disk metrics (root partition)
        disk = psutil.disk_usage('/')
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)
        disk_percent = (disk.used / disk.total) * 100
        
        # GPU metrics
        gpu_count = 0
        gpu_usage = []
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_memory_percent = []
        gpu_temperature = []
        
        if CUDA_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                
                if GPUtil:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_usage.append(gpu.load * 100)
                        gpu_memory_used.append(gpu.memoryUsed / 1024)  # Convert MB to GB
                        gpu_memory_total.append(gpu.memoryTotal / 1024)
                        gpu_memory_percent.append((gpu.memoryUsed / gpu.memoryTotal) * 100)
                        gpu_temperature.append(gpu.temperature)
                else:
                    # Fallback to PyTorch CUDA functions
                    for i in range(gpu_count):
                        gpu_usage.append(torch.cuda.utilization(i))
                        memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                        gpu_memory_used.append(memory_used)
                        gpu_memory_total.append(memory_total)
                        gpu_memory_percent.append((memory_used / memory_total) * 100)
                        gpu_temperature.append(0)  # Not available via PyTorch
                        
            except Exception as e:
                logger.warning(f"Failed to get GPU metrics: {e}")
        
        return ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            memory_percent=memory_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
            gpu_count=gpu_count,
            gpu_usage_percent=gpu_usage,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temperature=gpu_temperature
        )


class ProcessResourceMonitor:
    """Monitor resource usage for specific processes."""
    
    def __init__(self):
        self.tracked_processes: Dict[int, psutil.Process] = {}
        
    def add_process(self, pid: int) -> bool:
        """Add a process to monitoring."""
        try:
            process = psutil.Process(pid)
            self.tracked_processes[pid] = process
            return True
        except psutil.NoSuchProcess:
            logger.warning(f"Process {pid} not found")
            return False
            
    def remove_process(self, pid: int) -> bool:
        """Remove a process from monitoring."""
        if pid in self.tracked_processes:
            del self.tracked_processes[pid]
            return True
        return False
        
    def get_process_metrics(self, pid: int) -> Optional[ProcessMetrics]:
        """Get resource metrics for a specific process."""
        if pid not in self.tracked_processes:
            if not self.add_process(pid):
                return None
                
        process = self.tracked_processes[pid]
        
        try:
            # Basic process info
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_used_gb = memory_info.rss / (1024**3)
            memory_percent = process.memory_percent()
            num_threads = process.num_threads()
            status = process.status()
            create_time = datetime.fromtimestamp(process.create_time())
            
            # GPU process info
            gpu_processes = []
            if CUDA_AVAILABLE and GPUtil:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        for proc in gpu.processes:
                            if proc['pid'] == pid:
                                gpu_processes.append({
                                    'gpu_id': gpu.id,
                                    'gpu_name': gpu.name,
                                    'memory_used_mb': proc['memory_used'],
                                    'memory_used_gb': proc['memory_used'] / 1024
                                })
                except Exception as e:
                    logger.warning(f"Failed to get GPU process info: {e}")
            
            return ProcessMetrics(
                pid=pid,
                name=process.name(),
                cpu_percent=cpu_percent,
                memory_used_gb=memory_used_gb,
                memory_percent=memory_percent,
                num_threads=num_threads,
                status=status,
                create_time=create_time,
                gpu_processes=gpu_processes
            )
            
        except psutil.NoSuchProcess:
            # Process no longer exists
            self.remove_process(pid)
            return None
        except Exception as e:
            logger.error(f"Error getting metrics for process {pid}: {e}")
            return None
            
    def get_all_process_metrics(self) -> Dict[int, ProcessMetrics]:
        """Get metrics for all tracked processes."""
        metrics = {}
        dead_processes = []
        
        for pid in self.tracked_processes:
            process_metrics = self.get_process_metrics(pid)
            if process_metrics:
                metrics[pid] = process_metrics
            else:
                dead_processes.append(pid)
                
        # Clean up dead processes
        for pid in dead_processes:
            self.remove_process(pid)
            
        return metrics


class ResourceMonitorService:
    """Main service for resource monitoring with caching and history."""
    
    def __init__(self, history_duration: timedelta = timedelta(hours=1)):
        self.system_monitor = SystemResourceMonitor()
        self.process_monitor = ProcessResourceMonitor()
        self.history_duration = history_duration
        self.system_history: List[ResourceMetrics] = []
        self.process_history: Dict[int, List[ProcessMetrics]] = {}
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        
    async def start_monitoring(self, interval: float = 5.0):
        """Start continuous monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("Resource monitoring started")
        
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")
        
    async def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_system_metrics()
                self.system_history.append(system_metrics)
                
                # Collect process metrics
                process_metrics = self.process_monitor.get_all_process_metrics()
                for pid, metrics in process_metrics.items():
                    if pid not in self.process_history:
                        self.process_history[pid] = []
                    self.process_history[pid].append(metrics)
                
                # Clean up old history
                self._cleanup_history()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
                
    def _cleanup_history(self):
        """Remove old entries from history."""
        cutoff_time = datetime.now() - self.history_duration
        
        # Clean system history
        self.system_history = [
            metrics for metrics in self.system_history
            if metrics.timestamp > cutoff_time
        ]
        
        # Clean process history
        for pid in list(self.process_history.keys()):
            self.process_history[pid] = [
                metrics for metrics in self.process_history[pid]
                if metrics.create_time > cutoff_time
            ]
            # Remove empty process histories
            if not self.process_history[pid]:
                del self.process_history[pid]
                
    def get_current_system_metrics(self) -> ResourceMetrics:
        """Get current system metrics."""
        return self.system_monitor.get_system_metrics()
        
    def get_current_process_metrics(self, pid: int) -> Optional[ProcessMetrics]:
        """Get current metrics for a specific process."""
        return self.process_monitor.get_process_metrics(pid)
        
    def get_system_history(self, duration: Optional[timedelta] = None) -> List[ResourceMetrics]:
        """Get system metrics history."""
        if duration is None:
            return self.system_history.copy()
            
        cutoff_time = datetime.now() - duration
        return [
            metrics for metrics in self.system_history
            if metrics.timestamp > cutoff_time
        ]
        
    def get_process_history(self, pid: int, duration: Optional[timedelta] = None) -> List[ProcessMetrics]:
        """Get process metrics history."""
        if pid not in self.process_history:
            return []
            
        if duration is None:
            return self.process_history[pid].copy()
            
        cutoff_time = datetime.now() - duration
        return [
            metrics for metrics in self.process_history[pid]
            if metrics.create_time > cutoff_time
        ]
        
    def add_process_to_monitoring(self, pid: int) -> bool:
        """Add a process to monitoring."""
        return self.process_monitor.add_process(pid)
        
    def remove_process_from_monitoring(self, pid: int) -> bool:
        """Remove a process from monitoring."""
        return self.process_monitor.remove_process(pid)


# Global resource monitor instance
resource_monitor = ResourceMonitorService()
