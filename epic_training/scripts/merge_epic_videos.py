#!/usr/bin/env python3
"""
🎬 MERGE EPIC 10-HOUR NEURAL NETWORK LEARNING JOURNEY
====================================================

This script merges the 10 individual 1-hour neural network learning videos
into a single epic 10-hour masterpiece showing the complete AI learning journey.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def create_file_list():
    """Create the file list for FFmpeg concatenation."""
    video_dir = Path("video/milestones")
    
    # Expected video files in order
    video_files = [
        "step_01000000_pct_10_analytics.mp4",
        "step_02000000_pct_20_analytics.mp4", 
        "step_03000000_pct_30_analytics.mp4",
        "step_04000000_pct_40_analytics.mp4",
        "step_05000000_pct_50_analytics.mp4",
        "step_06000000_pct_60_analytics.mp4",
        "step_07000000_pct_70_analytics.mp4",
        "step_08000000_pct_80_analytics.mp4",
        "step_09000000_pct_90_analytics.mp4",
        "step_10000000_pct_100_analytics.mp4"
    ]
    
    # Check that all files exist
    missing_files = []
    existing_files = []
    
    for video_file in video_files:
        video_path = video_dir / video_file
        if video_path.exists():
            existing_files.append(video_file)
            print(f"✅ Found: {video_file} ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            missing_files.append(video_file)
            print(f"❌ Missing: {video_file}")
    
    if missing_files:
        print(f"\n⚠️  Warning: {len(missing_files)} files are missing!")
        print("Available files will be merged in order.")
    
    # Create file list for FFmpeg
    filelist_path = Path("epic_journey_filelist.txt")
    with open(filelist_path, 'w') as f:
        for video_file in existing_files:
            # Use forward slashes for FFmpeg compatibility
            video_path = f"video/milestones/{video_file}"
            f.write(f"file '{video_path}'\n")
    
    print(f"\n📝 Created file list: {filelist_path}")
    print(f"📹 Will merge {len(existing_files)} videos")
    
    return filelist_path, existing_files

def merge_videos(filelist_path, output_filename):
    """Merge videos using FFmpeg."""
    output_path = Path("video") / output_filename
    
    print(f"\n🎬 Starting video merge...")
    print(f"📤 Output: {output_path}")
    print(f"⏱️  This may take 10-30 minutes depending on your system...")
    
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
    
    try:
        print(f"\n🚀 Running FFmpeg...")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)
        
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
            print("=" * 60)
            print("🎉 EPIC 10-HOUR VIDEO CREATED SUCCESSFULLY! 🎉")
            
            # Check output file
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"📹 File: {output_path}")
                print(f"💾 Size: {file_size_mb:.1f} MB")
                print(f"⏱️  Duration: ~10 hours of neural network learning!")
                print(f"🧠 Content: Complete AI learning journey from chaos to mastery")
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

def main():
    """Main function to orchestrate the epic video merge."""
    print("🎬 EPIC 10-HOUR NEURAL NETWORK LEARNING JOURNEY MERGER")
    print("=" * 60)
    print("🎯 Merging 10 individual 1-hour videos into epic masterpiece...")
    print()
    
    # Create file list
    filelist_path, existing_files = create_file_list()
    
    if not existing_files:
        print("❌ No video files found to merge!")
        return 1
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"EPIC_10Hour_Neural_Learning_Journey_{timestamp}.mp4"
    
    print(f"\n🎯 Ready to create epic video:")
    print(f"   📹 Input: {len(existing_files)} videos (~{len(existing_files)} hours)")
    print(f"   📤 Output: {output_filename}")
    print(f"   💾 Expected size: ~800 MB")
    
    # Confirm merge
    response = input(f"\n🚀 Create the epic 10-hour neural learning journey? (yes/no): ").lower().strip()
    if response not in ['yes', 'y', 'true', '1']:
        print("🛑 Merge cancelled.")
        return 0
    
    # Merge videos
    success = merge_videos(filelist_path, output_filename)
    
    # Cleanup
    try:
        filelist_path.unlink()
        print(f"🧹 Cleaned up temporary file: {filelist_path}")
    except:
        pass
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 EPIC JOURNEY MERGE COMPLETED! 🎉")
        print("🧠 You now have the complete 10-hour neural network learning masterpiece!")
        print("🎮 Watch an AI learn to master Breakout from complete scratch!")
        print("=" * 60)
        return 0
    else:
        print("\n❌ Merge failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
