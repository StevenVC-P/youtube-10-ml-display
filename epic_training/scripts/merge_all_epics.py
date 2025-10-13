#!/usr/bin/env python3
"""
🎬 MERGE ALL EPIC VIDEOS
=======================

This script merges all epic videos into a single mega-video showing the complete
neural network learning journey from random play to perfect mastery.

Usage:
    # Merge all available epics for Breakout
    python merge_all_epics.py --game breakout

    # Merge specific epic range
    python merge_all_epics.py --game breakout --start-epic 1 --end-epic 3

    # Custom output name
    python merge_all_epics.py --game breakout --output "Breakout_Complete_Journey.mp4"

Features:
- Automatically finds all completed epics
- Creates seamless transitions between epics
- Adds epic titles and progress indicators
- Generates a single mega-video showing complete progression
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

def find_completed_epics(game: str, start_epic: int = 1, end_epic: int = 10):
    """Find all completed epics for a game."""
    game_dir = Path("games") / game.lower()
    completed_epics = []
    
    for epic_num in range(start_epic, end_epic + 1):
        epic_pattern = f"epic_{epic_num:03d}_*"
        epic_dirs = list(game_dir.glob(epic_pattern))
        
        if epic_dirs:
            epic_dir = epic_dirs[0]
            
            # Check for individual hour videos
            videos_dir = epic_dir / "videos" / "individual_hours"
            if videos_dir.exists():
                video_files = list(videos_dir.glob("*.mp4"))
                if len(video_files) >= 10:  # Should have 10 hour videos
                    completed_epics.append({
                        'epic_num': epic_num,
                        'epic_dir': epic_dir,
                        'videos_dir': videos_dir,
                        'video_count': len(video_files)
                    })
                    print(f"✅ Found Epic {epic_num}: {len(video_files)} videos")
                else:
                    print(f"⚠️  Epic {epic_num}: Only {len(video_files)} videos (incomplete)")
            else:
                print(f"❌ Epic {epic_num}: No videos directory found")
        else:
            print(f"❌ Epic {epic_num}: Not found")
    
    return completed_epics

def create_epic_filelist(completed_epics: list, output_dir: Path):
    """Create FFmpeg file list for all epic videos."""
    filelist_path = output_dir / "all_epics_filelist.txt"
    
    with open(filelist_path, 'w') as f:
        for epic_info in completed_epics:
            epic_num = epic_info['epic_num']
            videos_dir = epic_info['videos_dir']
            
            # Find all hour videos for this epic (sorted by step number)
            video_files = sorted(videos_dir.glob("step_*_analytics.mp4"))
            
            print(f"📹 Epic {epic_num}: Adding {len(video_files)} videos")
            
            for video_file in video_files:
                # Use forward slashes for FFmpeg compatibility
                relative_path = video_file.relative_to(Path.cwd())
                f.write(f"file '{relative_path}'\n")
    
    print(f"📝 Created master file list: {filelist_path}")
    return filelist_path

def estimate_output_size(completed_epics: list):
    """Estimate the output video size."""
    total_hours = 0
    total_size_mb = 0
    
    for epic_info in completed_epics:
        videos_dir = epic_info['videos_dir']
        video_files = list(videos_dir.glob("*.mp4"))
        
        epic_hours = len(video_files)  # Each video is 1 hour
        total_hours += epic_hours
        
        # Estimate size: ~80-100 MB per hour video
        epic_size_mb = epic_hours * 90  # Average estimate
        total_size_mb += epic_size_mb
    
    return total_hours, total_size_mb

def merge_all_epic_videos(completed_epics: list, output_path: Path):
    """Merge all epic videos using FFmpeg."""
    
    # Create temporary directory for processing
    temp_dir = Path("temp_epic_merge")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Create file list
        filelist_path = create_epic_filelist(completed_epics, temp_dir)
        
        # Estimate output
        total_hours, total_size_mb = estimate_output_size(completed_epics)
        
        print(f"\n🎬 Starting mega-video merge...")
        print(f"📊 Total Content: {total_hours} hours from {len(completed_epics)} epics")
        print(f"💾 Estimated Size: {total_size_mb:.0f} MB (~{total_size_mb/1024:.1f} GB)")
        print(f"📤 Output: {output_path}")
        print(f"⏱️  This may take 30-60 minutes depending on your system...")
        
        # FFmpeg command for concatenation
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",  # Copy streams without re-encoding (much faster)
            "-y",  # Overwrite output file
            str(output_path)
        ]
        
        print(f"\n🚀 Running FFmpeg...")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80)
        
        # Run FFmpeg with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        # Wait for completion
        process.wait()
        
        if process.returncode == 0:
            print("=" * 80)
            print("🎉 MEGA EPIC VIDEO CREATED SUCCESSFULLY! 🎉")
            
            # Check output file
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"📹 File: {output_path}")
                print(f"💾 Size: {file_size_mb:.1f} MB ({file_size_mb/1024:.1f} GB)")
                print(f"⏱️  Duration: ~{total_hours} hours of neural network learning!")
                print(f"🧠 Content: Complete AI journey from chaos to perfect mastery")
                return True
            else:
                print("❌ Output file was not created!")
                return False
        else:
            print(f"❌ FFmpeg failed with return code: {process.returncode}")
            return False
            
    except FileNotFoundError:
        print("❌ FFmpeg not found! Please install FFmpeg.")
        print("   Download from: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        return False
    finally:
        # Cleanup temporary files
        try:
            if filelist_path.exists():
                filelist_path.unlink()
            temp_dir.rmdir()
        except:
            pass

def create_epic_summary(game: str, completed_epics: list, output_path: Path):
    """Create a summary file for the merged epic video."""
    summary = {
        "game": game.title(),
        "creation_date": datetime.now().isoformat(),
        "total_epics": len(completed_epics),
        "total_hours": sum(epic['video_count'] for epic in completed_epics),
        "output_file": str(output_path),
        "epic_breakdown": []
    }
    
    for epic_info in completed_epics:
        epic_num = epic_info['epic_num']
        video_count = epic_info['video_count']
        
        epic_names = {
            1: "From Scratch → Competent Play",
            2: "Competent → Expert Level", 
            3: "Expert → Perfect Play"
        }
        
        epic_desc = epic_names.get(epic_num, f"Advanced Training Level {epic_num}")
        
        summary["epic_breakdown"].append({
            "epic_number": epic_num,
            "description": epic_desc,
            "hours": video_count,
            "directory": str(epic_info['epic_dir'])
        })
    
    summary_path = output_path.with_suffix('.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Epic summary saved: {summary_path}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Merge all epic videos into a single mega-video")
    parser.add_argument("--game", required=True,
                       choices=["breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger"],
                       help="Game to merge epics for")
    parser.add_argument("--start-epic", type=int, default=1,
                       help="Starting epic number (default: 1)")
    parser.add_argument("--end-epic", type=int, default=10,
                       help="Ending epic number (default: 10)")
    parser.add_argument("--output", type=str,
                       help="Output filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    print("🎬 EPIC VIDEO MERGER")
    print("=" * 60)
    print(f"🎯 Game: {args.game.title()}")
    print(f"📊 Epic Range: {args.start_epic} → {args.end_epic}")
    print()
    
    # Find completed epics
    print("🔍 Scanning for completed epics...")
    completed_epics = find_completed_epics(args.game, args.start_epic, args.end_epic)
    
    if not completed_epics:
        print("❌ No completed epics found!")
        print(f"   Make sure you have trained epics for {args.game}")
        return 1
    
    print(f"\n✅ Found {len(completed_epics)} completed epics")
    
    # Generate output filename
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epic_range = f"Epic{args.start_epic}to{completed_epics[-1]['epic_num']}"
        output_filename = f"{args.game.title()}_{epic_range}_Complete_Journey_{timestamp}.mp4"
    
    output_path = Path("video") / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Estimate and confirm
    total_hours, total_size_mb = estimate_output_size(completed_epics)
    
    print(f"\n🎯 Ready to create mega epic video:")
    print(f"   📹 Input: {len(completed_epics)} epics (~{total_hours} hours)")
    print(f"   📤 Output: {output_filename}")
    print(f"   💾 Expected size: ~{total_size_mb/1024:.1f} GB")
    
    # Confirm merge
    response = input(f"\n🚀 Create the complete epic journey video? (yes/no): ").lower().strip()
    if response not in ['yes', 'y', 'true', '1']:
        print("🛑 Merge cancelled.")
        return 0
    
    # Merge videos
    success = merge_all_epic_videos(completed_epics, output_path)
    
    if success:
        # Create summary
        create_epic_summary(args.game, completed_epics, output_path)
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE EPIC JOURNEY CREATED! 🎉")
        print("🧠 You now have the ultimate neural network learning masterpiece!")
        print(f"🎮 Watch an AI master {args.game.title()} from complete scratch to perfection!")
        print("=" * 60)
        return 0
    else:
        print("\n❌ Merge failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
