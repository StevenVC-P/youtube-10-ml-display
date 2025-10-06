#!/usr/bin/env python3
"""Find the most recently created video"""

from pathlib import Path
import os

def find_latest_video():
    """Find the most recently created video file"""
    video_dirs = [
        Path("video/milestones"),
        Path("video/eval"),
        Path("video/render/parts"),
        Path("video/render")
    ]
    
    all_videos = []
    
    for video_dir in video_dirs:
        if video_dir.exists():
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                all_videos.extend(video_dir.glob(f"*{ext}"))
    
    if not all_videos:
        print("No videos found!")
        return None
    
    # Sort by modification time (most recent first)
    latest_video = max(all_videos, key=lambda p: p.stat().st_mtime)
    
    size_mb = latest_video.stat().st_size / (1024*1024)
    print(f"Latest video: {latest_video}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Created: {latest_video.stat().st_mtime}")
    
    return str(latest_video)

if __name__ == "__main__":
    latest = find_latest_video()
    if latest:
        print(f"\nTo watch: vlc \"{latest}\"")