"""
Export Service - Unified Export Interface

Provides a single interface for exporting videos, models, and other artifacts.
Architecture supports future exporters (YouTube, Twitter, etc.) while providing
basic functionality now.

Usage:
    service = ExportService()

    # Basic operations (v1.0)
    service.copy_path_to_clipboard(video_path)
    service.open_video_location(video_path)
    service.rename_video(video_path, "My_Epic_Training.mp4")

    # Future operations (v1.1+)
    service.export_for_youtube(video_path, title="AI Plays Breakout")
    service.export_for_twitter(video_path, compression="high")
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExportService:
    """
    Unified service for exporting videos, models, and artifacts.

    Provides basic file operations now with extensible architecture
    for future export targets (YouTube, Twitter, cloud storage, etc.)
    """

    def __init__(self):
        """Initialize export service."""
        logger.info("ExportService initialized")

    # ========== Basic File Operations (v1.0) ==========

    def copy_path_to_clipboard(self, file_path: str) -> bool:
        """
        Copy file path to system clipboard.

        Args:
            file_path: Path to copy

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = str(Path(file_path).absolute())

            if sys.platform == "win32":
                # Windows - use clip command
                process = subprocess.Popen(
                    ['clip'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                process.communicate(input=file_path.encode('utf-8'))
                logger.info(f"Copied to clipboard: {file_path}")
                return True

            elif sys.platform == "darwin":
                # macOS - use pbcopy
                process = subprocess.Popen(
                    ['pbcopy'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                process.communicate(input=file_path.encode('utf-8'))
                logger.info(f"Copied to clipboard: {file_path}")
                return True

            else:
                # Linux - use xclip if available
                try:
                    process = subprocess.Popen(
                        ['xclip', '-selection', 'clipboard'],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    process.communicate(input=file_path.encode('utf-8'))
                    logger.info(f"Copied to clipboard: {file_path}")
                    return True
                except FileNotFoundError:
                    logger.warning("xclip not found on Linux system")
                    return False

        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            return False

    def open_video_location(self, video_path: str) -> bool:
        """
        Open file explorer/finder at the video's location with the file selected.

        Args:
            video_path: Path to video file

        Returns:
            True if successful, False otherwise
        """
        try:
            video_path = Path(video_path).absolute()

            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return False

            if sys.platform == "win32":
                # Windows - use explorer with /select flag
                subprocess.run(['explorer', '/select,', str(video_path)], check=True)
                logger.info(f"Opened location: {video_path}")
                return True

            elif sys.platform == "darwin":
                # macOS - use 'open' with -R flag
                subprocess.run(['open', '-R', str(video_path)], check=True)
                logger.info(f"Opened location: {video_path}")
                return True

            else:
                # Linux - open parent folder
                parent_dir = video_path.parent
                subprocess.run(['xdg-open', str(parent_dir)], check=True)
                logger.info(f"Opened parent directory: {parent_dir}")
                return True

        except Exception as e:
            logger.error(f"Failed to open location: {e}")
            return False

    def open_folder(self, folder_path: str) -> bool:
        """
        Open a folder in file explorer/finder.

        Args:
            folder_path: Path to folder

        Returns:
            True if successful, False otherwise
        """
        try:
            folder_path = Path(folder_path).absolute()

            if not folder_path.exists():
                logger.error(f"Folder not found: {folder_path}")
                return False

            if not folder_path.is_dir():
                logger.error(f"Path is not a directory: {folder_path}")
                return False

            if sys.platform == "win32":
                subprocess.run(['explorer', str(folder_path)], check=True)
            elif sys.platform == "darwin":
                subprocess.run(['open', str(folder_path)], check=True)
            else:
                subprocess.run(['xdg-open', str(folder_path)], check=True)

            logger.info(f"Opened folder: {folder_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to open folder: {e}")
            return False

    def rename_video(self, video_path: str, new_name: str) -> Optional[str]:
        """
        Rename a video file.

        Args:
            video_path: Current video path
            new_name: New filename (with or without extension)

        Returns:
            New path if successful, None otherwise
        """
        try:
            video_path = Path(video_path).absolute()

            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return None

            # Ensure new_name has extension
            if not Path(new_name).suffix:
                new_name = f"{new_name}{video_path.suffix}"

            # Create new path in same directory
            new_path = video_path.parent / new_name

            # Check if new name already exists
            if new_path.exists():
                logger.error(f"File already exists: {new_path}")
                return None

            # Rename
            video_path.rename(new_path)
            logger.info(f"Renamed video: {video_path} -> {new_path}")
            return str(new_path)

        except Exception as e:
            logger.error(f"Failed to rename video: {e}")
            return None

    def copy_video(self, video_path: str, destination: str) -> Optional[str]:
        """
        Copy video to a destination.

        Args:
            video_path: Source video path
            destination: Destination path (file or directory)

        Returns:
            Path to copied file if successful, None otherwise
        """
        try:
            video_path = Path(video_path).absolute()
            destination = Path(destination)

            if not video_path.exists():
                logger.error(f"Video file not found: {video_path}")
                return None

            # If destination is directory, use same filename
            if destination.is_dir():
                destination = destination / video_path.name

            # Copy file
            shutil.copy2(video_path, destination)
            logger.info(f"Copied video: {video_path} -> {destination}")
            return str(destination)

        except Exception as e:
            logger.error(f"Failed to copy video: {e}")
            return None

    def delete_video(self, video_path: str) -> bool:
        """
        Delete a video file.

        Args:
            video_path: Path to video to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            video_path = Path(video_path)

            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                return False

            video_path.unlink()
            logger.info(f"Deleted video: {video_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete video: {e}")
            return False

    def get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video file information.

        Args:
            video_path: Path to video

        Returns:
            Dictionary with video info or None if error
        """
        try:
            video_path = Path(video_path)

            if not video_path.exists():
                return None

            stat = video_path.stat()

            return {
                'path': str(video_path.absolute()),
                'name': video_path.name,
                'size_bytes': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'extension': video_path.suffix
            }

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None

    # ========== Future Export Methods (v1.1+) ==========

    def export_for_youtube(
        self,
        video_path: str,
        title: str,
        description: str = "",
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Prepare video for YouTube upload (future implementation).

        Args:
            video_path: Path to video
            title: Video title
            description: Video description
            tags: List of tags

        Returns:
            Dictionary with export details and instructions
        """
        logger.info("YouTube export requested (future feature)")

        return {
            'success': False,
            'message': 'YouTube export is a planned feature for v1.1',
            'instructions': [
                '1. Open your video folder using "Open Location"',
                '2. Go to https://studio.youtube.com/channel/UC/videos/upload',
                '3. Drag and drop your video',
                '4. Add title, description, and tags',
                '5. Publish!'
            ],
            'video_path': video_path,
            'suggested_title': title,
            'suggested_description': description,
            'suggested_tags': tags or []
        }

    def export_for_twitter(
        self,
        video_path: str,
        compression: str = "medium"
    ) -> Dict[str, Any]:
        """
        Prepare video for Twitter (future implementation).

        Twitter has specific requirements:
        - Max 2:20 duration for most accounts
        - Max 512MB file size
        - Recommended aspect ratio 16:9 or 1:1

        Args:
            video_path: Path to video
            compression: Compression level ('low', 'medium', 'high')

        Returns:
            Dictionary with export details
        """
        logger.info("Twitter export requested (future feature)")

        return {
            'success': False,
            'message': 'Twitter export is a planned feature for v1.1',
            'requirements': {
                'max_duration': '2:20',
                'max_file_size': '512MB',
                'recommended_aspect_ratios': ['16:9', '1:1']
            },
            'video_path': video_path
        }

    def export_model_checkpoint(
        self,
        model_path: str,
        format: str = "zip"
    ) -> Optional[str]:
        """
        Export model checkpoint for sharing (future implementation).

        Args:
            model_path: Path to model checkpoint
            format: Export format ('zip', 'tar.gz')

        Returns:
            Path to exported file if successful
        """
        logger.info("Model export requested (future feature)")
        return None


# Global singleton instance
_global_export_service = None


def get_export_service() -> ExportService:
    """
    Get the global export service singleton.

    Returns:
        ExportService instance
    """
    global _global_export_service
    if _global_export_service is None:
        _global_export_service = ExportService()
    return _global_export_service
