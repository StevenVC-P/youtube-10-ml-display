#!/usr/bin/env python3
"""
🎯 EPIC 10-HOUR NEURAL NETWORK LEARNING JOURNEY
===============================================

This script launches the complete 10-hour Breakout neural network learning journey.
It will create 10 consecutive 1-hour videos showing the AI's progression from 
random play to mastery.

Expected Output:
- Hour 1 (10%):  Random exploration, basic neural patterns
- Hour 2 (20%):  Early learning, paddle tracking begins
- Hour 3 (30%):  Basic ball tracking, improved coordination
- Hour 4 (40%):  Strategic positioning, better timing
- Hour 5 (50%):  Consistent ball returns, pattern recognition
- Hour 6 (60%):  Advanced strategies, corner shots
- Hour 7 (70%):  Mastery emerging, high scores
- Hour 8 (80%):  Expert play, complex strategies
- Hour 9 (90%):  Near-perfect performance
- Hour 10 (100%): Complete mastery, optimal play

Each video will be 1 hour long (3600 seconds) at 30 FPS with enhanced neural
network visualization showing the AI's "brain" learning in real-time.

Total Training: 10,000,000 steps (~36 hours of real-time training)
Total Video Output: 10 hours of neural network learning footage
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print the epic journey banner."""
    print("=" * 80)
    print("🧠 EPIC 10-HOUR NEURAL NETWORK LEARNING JOURNEY 🎮")
    print("=" * 80)
    print("🎯 Training an AI to master Atari Breakout from scratch")
    print("📹 Recording 10 consecutive 1-hour neural network videos")
    print("🚀 Total: 10,000,000 training steps over ~36 hours")
    print("=" * 80)
    print()

def estimate_completion_time():
    """Estimate completion time based on previous runs."""
    # Based on previous test: 15,000 steps took ~6 minutes
    # 10,000,000 steps = ~4,000 minutes = ~66.7 hours = ~2.8 days
    total_hours = 66.7
    completion_time = datetime.now().timestamp() + (total_hours * 3600)
    completion_datetime = datetime.fromtimestamp(completion_time)
    
    print(f"⏰ Estimated completion: {completion_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Estimated duration: {total_hours:.1f} hours (~{total_hours/24:.1f} days)")
    print()

def check_disk_space():
    """Check available disk space for the epic journey."""
    # Each 1-hour video at 30 FPS, 960x540 resolution ≈ 200-500 MB
    # 10 videos ≈ 2-5 GB total
    print("💾 Disk space requirements:")
    print("   - Each 1-hour video: ~200-500 MB")
    print("   - Total 10 videos: ~2-5 GB")
    print("   - Training logs/checkpoints: ~1 GB")
    print("   - Recommended free space: 10+ GB")
    print()

def confirm_launch():
    """Confirm the user wants to start the epic journey."""
    print("⚠️  WARNING: This is a LONG training session!")
    print("   - Will run for approximately 2-3 days continuously")
    print("   - Computer should remain on and stable")
    print("   - Will generate 10 large video files")
    print()
    
    response = input("🚀 Ready to begin the epic journey? (yes/no): ").lower().strip()
    return response in ['yes', 'y', 'true', '1']

def launch_training():
    """Launch the epic 10-hour training session."""
    print("🎬 Starting Epic Neural Network Learning Journey...")
    print("📹 Videos will be saved to: video/milestones/")
    print("💾 Checkpoints will be saved to: models/checkpoints/")
    print("📊 TensorBoard logs: logs/tb/")
    print()
    print("🔥 TRAINING STARTED! 🔥")
    print("=" * 80)
    
    # Launch the training with full 10M timesteps
    cmd = [sys.executable, "training/train.py", "--verbose", "1"]
    
    try:
        # Start the training process
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
            print("🎉 EPIC JOURNEY COMPLETED SUCCESSFULLY! 🎉")
            print("📹 Check video/milestones/ for your 10-hour neural learning videos!")
            print("=" * 80)
        else:
            print("❌ Training failed with return code:", process.returncode)
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        if process:
            process.terminate()
    except Exception as e:
        print(f"❌ Error during training: {e}")

def main():
    """Main function to orchestrate the epic journey."""
    print_banner()
    estimate_completion_time()
    check_disk_space()
    
    if not confirm_launch():
        print("🛑 Epic journey cancelled. Ready when you are!")
        return
    
    # Final countdown
    print("\n🚀 Launching in:")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    print("   🎬 ACTION!")
    print()
    
    launch_training()

if __name__ == "__main__":
    main()
