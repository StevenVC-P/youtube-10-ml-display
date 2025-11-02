"""
API endpoints for resource monitoring.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.resource_monitor import resource_monitor, ResourceMetrics, ProcessMetrics
from ..core.container_manager import container_manager

router = APIRouter(prefix="/api/resources", tags=["resources"])


# Response Models
class SystemMetricsResponse(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    gpu_count: int
    gpu_usage_percent: List[float]
    gpu_memory_used_gb: List[float]
    gpu_memory_total_gb: List[float]
    gpu_memory_percent: List[float]
    gpu_temperature: List[float]
    
    @classmethod
    def from_resource_metrics(cls, metrics: ResourceMetrics) -> "SystemMetricsResponse":
        return cls(
            timestamp=metrics.timestamp,
            cpu_percent=metrics.cpu_percent,
            memory_used_gb=metrics.memory_used_gb,
            memory_total_gb=metrics.memory_total_gb,
            memory_percent=metrics.memory_percent,
            disk_used_gb=metrics.disk_used_gb,
            disk_total_gb=metrics.disk_total_gb,
            disk_percent=metrics.disk_percent,
            gpu_count=metrics.gpu_count,
            gpu_usage_percent=metrics.gpu_usage_percent,
            gpu_memory_used_gb=metrics.gpu_memory_used_gb,
            gpu_memory_total_gb=metrics.gpu_memory_total_gb,
            gpu_memory_percent=metrics.gpu_memory_percent,
            gpu_temperature=metrics.gpu_temperature
        )


class ProcessMetricsResponse(BaseModel):
    pid: int
    name: str
    cpu_percent: float
    memory_used_gb: float
    memory_percent: float
    num_threads: int
    status: str
    create_time: datetime
    gpu_processes: List[Dict[str, Any]]
    
    @classmethod
    def from_process_metrics(cls, metrics: ProcessMetrics) -> "ProcessMetricsResponse":
        return cls(
            pid=metrics.pid,
            name=metrics.name,
            cpu_percent=metrics.cpu_percent,
            memory_used_gb=metrics.memory_used_gb,
            memory_percent=metrics.memory_percent,
            num_threads=metrics.num_threads,
            status=metrics.status,
            create_time=metrics.create_time,
            gpu_processes=metrics.gpu_processes
        )


class ContainerMetricsResponse(BaseModel):
    container_id: str
    container_name: str
    container_status: str
    process_metrics: Optional[ProcessMetricsResponse]


class ResourceSummaryResponse(BaseModel):
    system_metrics: SystemMetricsResponse
    container_metrics: List[ContainerMetricsResponse]
    total_containers: int
    running_containers: int
    resource_utilization: Dict[str, float]


# API Endpoints
@router.get("/system", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get current system resource metrics."""
    metrics = resource_monitor.get_current_system_metrics()
    return SystemMetricsResponse.from_resource_metrics(metrics)


@router.get("/system/history", response_model=List[SystemMetricsResponse])
async def get_system_metrics_history(
    duration_minutes: int = Query(default=60, ge=1, le=1440)
):
    """Get system resource metrics history."""
    duration = timedelta(minutes=duration_minutes)
    history = resource_monitor.get_system_history(duration)
    return [SystemMetricsResponse.from_resource_metrics(m) for m in history]


@router.get("/containers", response_model=List[ContainerMetricsResponse])
async def get_container_metrics():
    """Get resource metrics for all containers."""
    containers = container_manager.list_containers()
    container_metrics = []
    
    for container in containers:
        process_metrics = None
        if container.process_id:
            metrics = resource_monitor.get_current_process_metrics(container.process_id)
            if metrics:
                process_metrics = ProcessMetricsResponse.from_process_metrics(metrics)
        
        container_metrics.append(ContainerMetricsResponse(
            container_id=container.id,
            container_name=container.name,
            container_status=container.status.value,
            process_metrics=process_metrics
        ))
    
    return container_metrics


@router.get("/containers/{container_id}", response_model=ContainerMetricsResponse)
async def get_container_metrics_by_id(container_id: str):
    """Get resource metrics for a specific container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    process_metrics = None
    if container.process_id:
        metrics = resource_monitor.get_current_process_metrics(container.process_id)
        if metrics:
            process_metrics = ProcessMetricsResponse.from_process_metrics(metrics)
    
    return ContainerMetricsResponse(
        container_id=container.id,
        container_name=container.name,
        container_status=container.status.value,
        process_metrics=process_metrics
    )


@router.get("/containers/{container_id}/history", response_model=List[ProcessMetricsResponse])
async def get_container_metrics_history(
    container_id: str,
    duration_minutes: int = Query(default=60, ge=1, le=1440)
):
    """Get resource metrics history for a specific container."""
    container = container_manager.get_container(container_id)
    if not container:
        raise HTTPException(status_code=404, detail="Container not found")
    
    if not container.process_id:
        return []
    
    duration = timedelta(minutes=duration_minutes)
    history = resource_monitor.get_process_history(container.process_id, duration)
    return [ProcessMetricsResponse.from_process_metrics(m) for m in history]


@router.get("/summary", response_model=ResourceSummaryResponse)
async def get_resource_summary():
    """Get comprehensive resource summary."""
    # Get system metrics
    system_metrics = resource_monitor.get_current_system_metrics()
    
    # Get container metrics
    containers = container_manager.list_containers()
    container_metrics = []
    running_containers = 0
    
    for container in containers:
        if container.status.value == "running":
            running_containers += 1
            
        process_metrics = None
        if container.process_id:
            metrics = resource_monitor.get_current_process_metrics(container.process_id)
            if metrics:
                process_metrics = ProcessMetricsResponse.from_process_metrics(metrics)
        
        container_metrics.append(ContainerMetricsResponse(
            container_id=container.id,
            container_name=container.name,
            container_status=container.status.value,
            process_metrics=process_metrics
        ))
    
    # Calculate resource utilization
    total_resources = container_manager.get_system_resources()
    available_resources = container_manager.get_available_resources()
    
    cpu_utilization = ((total_resources.cpu_cores - available_resources.cpu_cores) / 
                      total_resources.cpu_cores * 100)
    memory_utilization = ((total_resources.memory_gb - available_resources.memory_gb) / 
                         total_resources.memory_gb * 100)
    disk_utilization = ((total_resources.disk_space_gb - available_resources.disk_space_gb) / 
                       total_resources.disk_space_gb * 100)
    
    resource_utilization = {
        "cpu_percent": cpu_utilization,
        "memory_percent": memory_utilization,
        "disk_percent": disk_utilization
    }
    
    if total_resources.gpu_memory_gb and available_resources.gpu_memory_gb:
        gpu_utilization = ((total_resources.gpu_memory_gb - available_resources.gpu_memory_gb) / 
                          total_resources.gpu_memory_gb * 100)
        resource_utilization["gpu_memory_percent"] = gpu_utilization
    
    return ResourceSummaryResponse(
        system_metrics=SystemMetricsResponse.from_resource_metrics(system_metrics),
        container_metrics=container_metrics,
        total_containers=len(containers),
        running_containers=running_containers,
        resource_utilization=resource_utilization
    )


@router.post("/monitoring/start")
async def start_monitoring(interval_seconds: float = Query(default=5.0, ge=1.0, le=60.0)):
    """Start resource monitoring."""
    await resource_monitor.start_monitoring(interval_seconds)
    return {"message": "Resource monitoring started", "interval": interval_seconds}


@router.post("/monitoring/stop")
async def stop_monitoring():
    """Stop resource monitoring."""
    await resource_monitor.stop_monitoring()
    return {"message": "Resource monitoring stopped"}


@router.get("/monitoring/status")
async def get_monitoring_status():
    """Get monitoring status."""
    return {
        "monitoring": resource_monitor.monitoring,
        "history_count": len(resource_monitor.system_history),
        "tracked_processes": len(resource_monitor.process_monitor.tracked_processes)
    }


@router.get("/alerts")
async def get_resource_alerts():
    """Get resource usage alerts."""
    alerts = []
    
    # Get current system metrics
    system_metrics = resource_monitor.get_current_system_metrics()
    
    # Check for high resource usage
    if system_metrics.cpu_percent > 90:
        alerts.append({
            "type": "warning",
            "resource": "cpu",
            "message": f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
            "timestamp": system_metrics.timestamp
        })
    
    if system_metrics.memory_percent > 90:
        alerts.append({
            "type": "warning",
            "resource": "memory",
            "message": f"High memory usage: {system_metrics.memory_percent:.1f}%",
            "timestamp": system_metrics.timestamp
        })
    
    if system_metrics.disk_percent > 90:
        alerts.append({
            "type": "warning",
            "resource": "disk",
            "message": f"High disk usage: {system_metrics.disk_percent:.1f}%",
            "timestamp": system_metrics.timestamp
        })
    
    # Check GPU usage
    for i, gpu_usage in enumerate(system_metrics.gpu_usage_percent):
        if gpu_usage > 95:
            alerts.append({
                "type": "warning",
                "resource": "gpu",
                "message": f"High GPU {i} usage: {gpu_usage:.1f}%",
                "timestamp": system_metrics.timestamp
            })
    
    for i, gpu_memory in enumerate(system_metrics.gpu_memory_percent):
        if gpu_memory > 95:
            alerts.append({
                "type": "warning",
                "resource": "gpu_memory",
                "message": f"High GPU {i} memory usage: {gpu_memory:.1f}%",
                "timestamp": system_metrics.timestamp
            })
    
    # Check for resource conflicts
    available_resources = container_manager.get_available_resources()
    if available_resources.cpu_cores < 1.0:
        alerts.append({
            "type": "error",
            "resource": "cpu",
            "message": "Insufficient CPU cores available for new containers",
            "timestamp": datetime.now()
        })
    
    if available_resources.memory_gb < 2.0:
        alerts.append({
            "type": "error",
            "resource": "memory",
            "message": "Insufficient memory available for new containers",
            "timestamp": datetime.now()
        })
    
    return {"alerts": alerts}
