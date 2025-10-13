"""
Stream module for continuous evaluation and video streaming.

This module provides:
- FFmpeg integration for real-time video encoding
- Grid composition for multiple environment streams
- Continuous evaluation with checkpoint monitoring
- HUD overlay with training statistics
"""

from .ffmpeg_io import FFmpegWriter, test_ffmpeg_availability
from .stream_eval import ContinuousEvaluator
from .grid_composer import SingleScreenComposer

__all__ = [
    'FFmpegWriter',
    'test_ffmpeg_availability',
    'ContinuousEvaluator',
    'SingleScreenComposer'
]
