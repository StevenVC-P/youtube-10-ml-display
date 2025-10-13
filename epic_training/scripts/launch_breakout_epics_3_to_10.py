#!/usr/bin/env python3
"""
Launch Breakout Epics 3-10 in sequence
Automatically starts the next epic when the previous one completes
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def check_epic_completion(epic_num):
    """Check if an epic has completed by looking for final model"""
    epic_dir = Path(f"games/breakout/epic_{epic_num:03d}_*")
    epic_dirs = list(Path("games/breakout").glob(f"epic_{epic_num:03d}_*"))
    
    if not epic_dirs:
        return False
    
    epic_dir = epic_dirs[0]
    final_model = epic_dir / "models" / "final" / f"epic_{epic_num:03d}_final_model.zip"
    
    return final_model.exists()

def wait_for_epic_completion(epic_num, max_hours=12):
    """Wait for an epic to complete, checking every 5 minutes"""
    print(f"⏳ Waiting for Epic {epic_num} to complete...")
    start_time = time.time()
    max_seconds = max_hours * 3600
    
    while time.time() - start_time < max_seconds:
        if check_epic_completion(epic_num):
            print(f"✅ Epic {epic_num} completed!")
            return True
        
        # Check every 5 minutes
        time.sleep(300)
        elapsed_hours = (time.time() - start_time) / 3600
        print(f"⏳ Epic {epic_num} still running... ({elapsed_hours:.1f}h elapsed)")
    
    print(f"⚠️ Epic {epic_num} did not complete within {max_hours} hours")
    return False

def launch_epic(epic_num):
    """Launch a single epic"""
    print(f"\n🚀 Launching Epic {epic_num}")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Launch the epic training script
        cmd = [
            sys.executable, 
            "epic_training/scripts/train_epic_continuous.py",
            "--game", "breakout",
            "--epic", str(epic_num),
            "--hours", "10"
        ]
        
        print(f"🔧 Command: {' '.join(cmd)}")
        
        # Start the process in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(10)
        
        # Check if process is still running
        if process.poll() is None:
            print(f"✅ Epic {epic_num} launched successfully (PID: {process.pid})")
            return True
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Epic {epic_num} failed to start:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error launching Epic {epic_num}: {e}")
        return False

def main():
    """Main function to launch epics 3-10 in sequence"""
    print("🎮 BREAKOUT EPIC TRAINING SEQUENCE")
    print("=" * 50)
    print("📋 Plan: Launch Epics 3-10 in sequence")
    print("⏱️  Each epic: 10 hours of training")
    print("📈 Total time: ~80 hours of continuous training")
    print("🎯 Goal: Complete neural network learning journey")
    print("=" * 50)
    
    # Epic range to launch
    start_epic = 3
    end_epic = 10
    
    # Check if Epic 3 is already running
    if not check_epic_completion(3):
        print("ℹ️  Epic 3 appears to be running already")
        print("⏳ Waiting for Epic 3 to complete before starting Epic 4...")
        
        if not wait_for_epic_completion(3):
            print("❌ Epic 3 did not complete. Aborting sequence.")
            return
    
    # Launch epics 4-10 in sequence
    for epic_num in range(4, end_epic + 1):
        print(f"\n{'='*20} EPIC {epic_num} {'='*20}")
        
        # Check if this epic is already completed
        if check_epic_completion(epic_num):
            print(f"✅ Epic {epic_num} already completed, skipping...")
            continue
        
        # Launch the epic
        if not launch_epic(epic_num):
            print(f"❌ Failed to launch Epic {epic_num}. Stopping sequence.")
            break
        
        # Wait for completion before launching next epic
        if epic_num < end_epic:  # Don't wait after the last epic
            if not wait_for_epic_completion(epic_num):
                print(f"❌ Epic {epic_num} did not complete. Stopping sequence.")
                break
    
    print("\n🎉 EPIC SEQUENCE COMPLETE!")
    print("📊 Summary:")
    for epic_num in range(start_epic, end_epic + 1):
        status = "✅ Complete" if check_epic_completion(epic_num) else "❌ Incomplete"
        print(f"   Epic {epic_num}: {status}")

if __name__ == "__main__":
    main()
