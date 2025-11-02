#!/usr/bin/env python3
"""
Startup script for the ML Container Management API.
"""

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Add the app directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        *([logging.FileHandler(settings.log_file)] if settings.log_file else [])
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info("Starting ML Container Management API")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Host: {settings.host}")
    logger.info(f"Port: {settings.port}")
    logger.info(f"Debug: {settings.debug}")
    logger.info(f"Reload: {settings.reload}")
    
    # Ensure required directories exist
    Path(settings.containers_base_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.ml_models_path).mkdir(parents=True, exist_ok=True)
    Path(settings.ml_videos_path).mkdir(parents=True, exist_ok=True)
    Path(settings.ml_logs_path).mkdir(parents=True, exist_ok=True)
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        access_log=True,
        use_colors=True
    )


if __name__ == "__main__":
    main()
