#!/usr/bin/env python3
"""Check what videos are available to watch"""

import os
from pathlib import Path

def check_available_videos():
    """Check for existing videos"""
    print("🎬 Checking Available Videos")
    print("=" * 40)
    
    video_locations = [
        ("Milestone Videos", "video/milestones"),
        ("Evaluation Videos", "video/eval"), 
        ("Render Parts", "video/render/parts"),
        ("Final Renders", "video/render")
    ]
    
    found_videos = []
    
    for category, path in video_locations:
        print(f"\n=== {category} ===")
        video_dir = Path(path)
        
        if not video_dir.exists():
            print(f"❌ Directory {path} doesn't exist")
            continue
            
        # Look for video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        videos = []
        
        for ext in video_extensions:
            videos.extend(video_dir.glob(f"*{ext}"))
            videos.extend(video_dir.glob(f"**/*{ext}"))  # recursive
        
        if videos:
            for video in sorted(videos):
                size_mb = video.stat().st_size / (1024*1024)
                print(f"✅ {video} ({size_mb:.1f} MB)")
                found_videos.append(str(video))
        else:
            print(f"⚠️  No videos found in {path}")
    
    return found_videos

def suggest_next_steps(found_videos):
    """Suggest what to do based on available videos"""
    print("\n" + "=" * 40)
    print("🎯 RECOMMENDATIONS")
    print("=" * 40)
    
    if found_videos:
        print(f"✅ Found {len(found_videos)} video(s)!")
        print("\nYou can watch these videos with:")
        for video in found_videos[:3]:  # Show first 3
            print(f"  vlc \"{video}\"")
        
        if len(found_videos) > 3:
            print(f"  ... and {len(found_videos) - 3} more")
            
    else:
        print("❌ No videos found yet!")
        print("\nTo generate videos, you need to:")
        print("1. Run training to create milestone videos:")
        print("   python training\\train.py --config conf\\config.yaml --dryrun-seconds 300")
        print("\n2. Or run evaluation to create eval videos:")
        print("   python training\\eval.py --checkpoint models\\checkpoints\\latest.zip --episodes 2")
        print("\n3. Or run streaming to create live segments:")
        print("   python stream\\stream_eval.py --config conf\\config.yaml --save-mode segments")

if __name__ == "__main__":
    videos = check_available_videos()
    suggest_next_steps(videos)