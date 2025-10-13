"""
Video Tools Package for YouTube 10 ML Display

This package provides post-processing tools for creating the final YouTube video
from training segments, milestone clips, and evaluation recordings.

Modules:
- build_manifest: Scan video directories and create CSV manifests
- concat_segments: Concatenate video segments using FFmpeg
- render_supercut: Create final supercut with optional music overlay
"""

from .build_manifest import VideoManifestBuilder, build_manifest_from_config
from .concat_segments import VideoSegmentConcatenator, concat_segments_from_config
from .render_supercut import SupercutRenderer, render_supercut_from_config

__all__ = [
    'VideoManifestBuilder',
    'build_manifest_from_config',
    'VideoSegmentConcatenator', 
    'concat_segments_from_config',
    'SupercutRenderer',
    'render_supercut_from_config'
]

__version__ = "1.0.0"
