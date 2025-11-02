"""
Main FastAPI application for ML Container Management UI.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from .api import containers, resources, websocket, training
from .api.websocket import broadcaster
from .core.resource_monitor import resource_monitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting ML Container Management API")
    
    # Start resource monitoring
    await resource_monitor.start_monitoring(interval=5.0)
    
    # Start WebSocket broadcasting
    await broadcaster.start_broadcasting(interval=2.0)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Container Management API")
    
    # Stop WebSocket broadcasting
    await broadcaster.stop_broadcasting()
    
    # Stop resource monitoring
    await resource_monitor.stop_monitoring()
    
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="ML Container Management API",
    description="API for managing multiple ML training containers with resource monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(containers.router)
app.include_router(resources.router)
app.include_router(training.router)
app.include_router(websocket.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ml-container-management-api",
        "version": "1.0.0",
        "monitoring": resource_monitor.monitoring,
        "broadcasting": broadcaster.broadcasting
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ML Container Management API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": {
            "containers": "/api/containers",
            "resources": "/api/resources",
            "websocket": {
                "resources": "/ws/resources",
                "containers": "/ws/containers",
                "training": "/ws/training/{container_id}"
            }
        }
    }


# API information endpoint
@app.get("/api")
async def api_info():
    """API information and available endpoints."""
    return {
        "title": "ML Container Management API",
        "version": "1.0.0",
        "endpoints": {
            "containers": {
                "list": "GET /api/containers",
                "create": "POST /api/containers",
                "get": "GET /api/containers/{id}",
                "start": "POST /api/containers/{id}/start",
                "stop": "POST /api/containers/{id}/stop",
                "delete": "DELETE /api/containers/{id}",
                "metrics": "GET /api/containers/{id}/metrics",
                "logs": "GET /api/containers/{id}/logs",
                "system_resources": "GET /api/containers/system/resources"
            },
            "resources": {
                "system": "GET /api/resources/system",
                "system_history": "GET /api/resources/system/history",
                "containers": "GET /api/resources/containers",
                "container": "GET /api/resources/containers/{id}",
                "container_history": "GET /api/resources/containers/{id}/history",
                "summary": "GET /api/resources/summary",
                "alerts": "GET /api/resources/alerts",
                "monitoring": {
                    "start": "POST /api/resources/monitoring/start",
                    "stop": "POST /api/resources/monitoring/stop",
                    "status": "GET /api/resources/monitoring/status"
                }
            },
            "websockets": {
                "resources": "WS /ws/resources",
                "containers": "WS /ws/containers",
                "training": "WS /ws/training/{container_id}"
            }
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return HTTPException(status_code=404, detail="Endpoint not found")


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")


# Static files (for serving frontend if needed)
static_dir = Path(__file__).parent.parent.parent / "frontend" / "out"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/ui", response_class=HTMLResponse)
    async def serve_ui():
        """Serve the frontend UI."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text())
        return HTMLResponse(content="<h1>Frontend not built</h1>")


if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
