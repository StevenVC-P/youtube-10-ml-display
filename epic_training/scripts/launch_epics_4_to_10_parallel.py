#!/usr/bin/env python3
"""
Launch Breakout Epics 4-10 in parallel (if system can handle it)
OR launch them in sequence with minimal delay
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def launch_epic_background(epic_num):
    """Launch a single epic in the background"""
    print(f"🚀 Launching Epic {epic_num} at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        cmd = [
            sys.executable, 
            "epic_training/scripts/train_epic_continuous.py",
            "--game", "breakout",
            "--epic", str(epic_num),
            "--hours", "10"
        ]
        
        # Start the process in the background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # Suppress output to avoid conflicts
            stderr=subprocess.DEVNULL,
            text=True
        )
        
        print(f"✅ Epic {epic_num} launched (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Error launching Epic {epic_num}: {e}")
        return None

def launch_epic_sequential(epic_num, delay_minutes=1):
    """Launch epic and wait a bit before returning"""
    process = launch_epic_background(epic_num)
    if process:
        print(f"⏳ Waiting {delay_minutes} minute(s) before next launch...")
        time.sleep(delay_minutes * 60)
    return process

def main():
    """Main function"""
    print("🎮 BREAKOUT EPIC BATCH LAUNCHER")
    print("=" * 40)
    print("📋 Target: Launch Epics 4-10")
    print("⚠️  Note: Epic 3 should already be running")
    print("=" * 40)
    
    # Ask user for launch mode
    print("\nChoose launch mode:")
    print("1. Sequential (recommended) - Launch one every 2 minutes")
    print("2. Parallel (risky) - Launch all at once")
    print("3. Manual - Launch one specific epic")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        # Sequential launch
        print("\n🔄 Sequential launch mode selected")
        processes = []
        
        for epic_num in range(4, 11):
            print(f"\n--- Launching Epic {epic_num} ---")
            process = launch_epic_sequential(epic_num, delay_minutes=2)
            if process:
                processes.append((epic_num, process))
            else:
                print(f"❌ Failed to launch Epic {epic_num}")
                break
        
        print(f"\n✅ Launched {len(processes)} epics successfully!")
        for epic_num, process in processes:
            print(f"   Epic {epic_num}: PID {process.pid}")
    
    elif choice == "2":
        # Parallel launch (risky)
        print("\n⚡ Parallel launch mode selected")
        print("⚠️  WARNING: This may cause memory issues!")
        
        confirm = input("Are you sure? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("❌ Cancelled")
            return
        
        processes = []
        for epic_num in range(4, 11):
            process = launch_epic_background(epic_num)
            if process:
                processes.append((epic_num, process))
            time.sleep(5)  # Small delay between launches
        
        print(f"\n✅ Launched {len(processes)} epics in parallel!")
        for epic_num, process in processes:
            print(f"   Epic {epic_num}: PID {process.pid}")
    
    elif choice == "3":
        # Manual single epic
        epic_num = input("\nEnter epic number (4-10): ").strip()
        try:
            epic_num = int(epic_num)
            if 4 <= epic_num <= 10:
                launch_epic_background(epic_num)
            else:
                print("❌ Epic number must be between 4 and 10")
        except ValueError:
            print("❌ Invalid epic number")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
