"""
WebSocket endpoints for real-time updates.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

from ..core.resource_monitor import resource_monitor
from ..core.container_manager import container_manager
from .resources import SystemMetricsResponse, ContainerMetricsResponse, ProcessMetricsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {
            "resources": set(),
            "containers": set(),
            "training": set()
        }
        self.container_connections: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, connection_type: str, container_id: str = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if container_id:
            if container_id not in self.container_connections:
                self.container_connections[container_id] = set()
            self.container_connections[container_id].add(websocket)
        else:
            if connection_type in self.active_connections:
                self.active_connections[connection_type].add(websocket)
        
        logger.info(f"WebSocket connected: {connection_type}" + 
                   (f" (container: {container_id})" if container_id else ""))
    
    def disconnect(self, websocket: WebSocket, connection_type: str, container_id: str = None):
        """Remove a WebSocket connection."""
        if container_id:
            if container_id in self.container_connections:
                self.container_connections[container_id].discard(websocket)
                if not self.container_connections[container_id]:
                    del self.container_connections[container_id]
        else:
            if connection_type in self.active_connections:
                self.active_connections[connection_type].discard(websocket)
        
        logger.info(f"WebSocket disconnected: {connection_type}" + 
                   (f" (container: {container_id})" if container_id else ""))
    
    async def broadcast_to_type(self, connection_type: str, message: dict):
        """Broadcast message to all connections of a specific type."""
        if connection_type not in self.active_connections:
            return
            
        connections = self.active_connections[connection_type].copy()
        if not connections:
            return
            
        message_str = json.dumps(message, default=str)
        disconnected = set()
        
        for connection in connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.active_connections[connection_type].discard(connection)
    
    async def broadcast_to_container(self, container_id: str, message: dict):
        """Broadcast message to all connections for a specific container."""
        if container_id not in self.container_connections:
            return
            
        connections = self.container_connections[container_id].copy()
        if not connections:
            return
            
        message_str = json.dumps(message, default=str)
        disconnected = set()
        
        for connection in connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to container WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.container_connections[container_id].discard(connection)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws/resources")
async def websocket_resources(websocket: WebSocket):
    """WebSocket endpoint for real-time resource updates."""
    await manager.connect(websocket, "resources")
    
    try:
        while True:
            # Wait for client message (ping/pong or configuration)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except asyncio.TimeoutError:
                # No message received, continue with regular updates
                pass
            except json.JSONDecodeError:
                # Invalid JSON, ignore
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "resources")


@router.websocket("/ws/containers")
async def websocket_containers(websocket: WebSocket):
    """WebSocket endpoint for real-time container updates."""
    await manager.connect(websocket, "containers")
    
    try:
        while True:
            # Wait for client message
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "containers")


@router.websocket("/ws/training/{container_id}")
async def websocket_training(websocket: WebSocket, container_id: str):
    """WebSocket endpoint for real-time training updates for a specific container."""
    # Verify container exists
    container = container_manager.get_container(container_id)
    if not container:
        await websocket.close(code=4004, reason="Container not found")
        return
    
    await manager.connect(websocket, "training", container_id)
    
    try:
        while True:
            # Wait for client message
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif message.get("type") == "get_status":
                    # Send current container status
                    container = container_manager.get_container(container_id)
                    if container:
                        status_message = {
                            "type": "status_update",
                            "container_id": container_id,
                            "status": container.status.value,
                            "timestamp": datetime.now().isoformat()
                        }
                        await websocket.send_text(json.dumps(status_message))
                    
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, "training", container_id)


class WebSocketBroadcaster:
    """Service for broadcasting updates to WebSocket clients."""
    
    def __init__(self):
        self.broadcasting = False
        self.broadcast_task = None
        
    async def start_broadcasting(self, interval: float = 2.0):
        """Start broadcasting updates to WebSocket clients."""
        if self.broadcasting:
            return
            
        self.broadcasting = True
        self.broadcast_task = asyncio.create_task(self._broadcast_loop(interval))
        logger.info("WebSocket broadcasting started")
        
    async def stop_broadcasting(self):
        """Stop broadcasting updates."""
        self.broadcasting = False
        if self.broadcast_task:
            self.broadcast_task.cancel()
            try:
                await self.broadcast_task
            except asyncio.CancelledError:
                pass
        logger.info("WebSocket broadcasting stopped")
        
    async def _broadcast_loop(self, interval: float):
        """Main broadcasting loop."""
        while self.broadcasting:
            try:
                # Broadcast resource updates
                await self._broadcast_resource_updates()
                
                # Broadcast container updates
                await self._broadcast_container_updates()
                
                # Broadcast training updates
                await self._broadcast_training_updates()
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(interval)
                
    async def _broadcast_resource_updates(self):
        """Broadcast system resource updates."""
        try:
            system_metrics = resource_monitor.get_current_system_metrics()
            message = {
                "type": "resource_update",
                "data": SystemMetricsResponse.from_resource_metrics(system_metrics).dict(),
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast_to_type("resources", message)
        except Exception as e:
            logger.error(f"Failed to broadcast resource updates: {e}")
            
    async def _broadcast_container_updates(self):
        """Broadcast container status updates."""
        try:
            containers = container_manager.list_containers()
            container_data = []
            
            for container in containers:
                process_metrics = None
                if container.process_id:
                    metrics = resource_monitor.get_current_process_metrics(container.process_id)
                    if metrics:
                        process_metrics = ProcessMetricsResponse.from_process_metrics(metrics).dict()
                
                container_data.append({
                    "container_id": container.id,
                    "container_name": container.name,
                    "container_status": container.status.value,
                    "process_metrics": process_metrics
                })
            
            message = {
                "type": "container_update",
                "data": container_data,
                "timestamp": datetime.now().isoformat()
            }
            await manager.broadcast_to_type("containers", message)
        except Exception as e:
            logger.error(f"Failed to broadcast container updates: {e}")
            
    async def _broadcast_training_updates(self):
        """Broadcast training progress updates for individual containers."""
        try:
            containers = container_manager.list_containers()
            
            for container in containers:
                if container.status.value == "running" and container.process_id:
                    # Get training metrics (this would be extended to read from training logs)
                    process_metrics = resource_monitor.get_current_process_metrics(container.process_id)
                    
                    training_data = {
                        "type": "training_update",
                        "container_id": container.id,
                        "status": container.status.value,
                        "process_metrics": ProcessMetricsResponse.from_process_metrics(process_metrics).dict() if process_metrics else None,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await manager.broadcast_to_container(container.id, training_data)
                    
        except Exception as e:
            logger.error(f"Failed to broadcast training updates: {e}")


# Global broadcaster instance
broadcaster = WebSocketBroadcaster()
