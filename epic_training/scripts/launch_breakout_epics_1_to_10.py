#!/usr/bin/env python3
"""
Launch Complete Breakout Epic Series: Epics 1-10
Each epic is a 10-hour GPU-accelerated training session with analytics videos

Epic Journey:
1. Epic 1: "From Scratch" - Learn basic Breakout from zero
2. Epic 2: "Advanced Mastery" - Build on Epic 1 skills  
3. Epic 3: "Perfect Play" - Achieve consistent performance
4. Epic 4: "Speed Demon" - Optimize for fast completion
5. Epic 5: "Precision Master" - Perfect ball control
6. Epic 6: "Strategic Genius" - Advanced game strategies
7. Epic 7: "Endurance Champion" - Long-term consistency
8. Epic 8: "Adaptive Expert" - Handle game variations
9. Epic 9: "Ultimate Mastery" - Peak performance
10. Epic 10: "Legendary Status" - Transcendent gameplay

Total Training Time: 100 hours of GPU-accelerated learning
Total Videos: 100+ hour-by-hour analytics videos + 10 epic summaries
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Epic definitions with thematic names
EPIC_DEFINITIONS = [
    {"num": 1, "name": "from_scratch", "title": "From Scratch", "description": "Learn basic Breakout from zero"},
    {"num": 2, "name": "advanced_mastery", "title": "Advanced Mastery", "description": "Build on Epic 1 skills"},
    {"num": 3, "name": "perfect_play", "title": "Perfect Play", "description": "Achieve consistent performance"},
    {"num": 4, "name": "speed_demon", "title": "Speed Demon", "description": "Optimize for fast completion"},
    {"num": 5, "name": "precision_master", "title": "Precision Master", "description": "Perfect ball control"},
    {"num": 6, "name": "strategic_genius", "title": "Strategic Genius", "description": "Advanced game strategies"},
    {"num": 7, "name": "endurance_champion", "title": "Endurance Champion", "description": "Long-term consistency"},
    {"num": 8, "name": "adaptive_expert", "title": "Adaptive Expert", "description": "Handle game variations"},
    {"num": 9, "name": "ultimate_mastery", "title": "Ultimate Mastery", "description": "Peak performance"},
    {"num": 10, "name": "legendary_status", "title": "Legendary Status", "description": "Transcendent gameplay"}
]

def check_epic_completion(epic_num):
    """Check if an epic has completed by looking for final model"""
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

def launch_epic(epic_def):
    """Launch a single epic"""
    epic_num = epic_def["num"]
    epic_name = epic_def["name"]
    epic_title = epic_def["title"]
    
    print(f"\n🚀 Launching Epic {epic_num}: {epic_title}")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Description: {epic_def['description']}")
    
    # Launch the epic training
    cmd = [
        sys.executable, 
        "epic_training/scripts/train_epic_continuous.py",
        "--game", "breakout",
        "--epic", str(epic_num),
        "--hours", "10"
    ]
    
    print(f"🔧 Command: {' '.join(cmd)}")
    
    try:
        # Launch in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        print(f"🎯 Epic {epic_num} launched with PID: {process.pid}")
        return process
        
    except Exception as e:
        print(f"❌ Failed to launch Epic {epic_num}: {e}")
        return None

def save_progress_log(completed_epics, current_epic, start_time):
    """Save progress to a log file"""
    log_data = {
        "series_start_time": start_time.isoformat(),
        "current_time": datetime.now().isoformat(),
        "completed_epics": completed_epics,
        "current_epic": current_epic,
        "total_epics": 10,
        "progress_percentage": len(completed_epics) / 10 * 100
    }
    
    log_file = Path("games/breakout/epic_series_progress.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

def main():
    """Launch the complete epic series"""
    print("🎮 BREAKOUT EPIC SERIES LAUNCHER")
    print("=" * 50)
    print("🎯 Goal: 10 Epic Training Sessions (100 hours total)")
    print("⚡ GPU-Accelerated with RTX 3070 Ti")
    print("📹 Enhanced Analytics Videos with Legend")
    print("=" * 50)
    
    start_time = datetime.now()
    completed_epics = []
    
    # Check if any epics are already completed
    for epic_def in EPIC_DEFINITIONS:
        if check_epic_completion(epic_def["num"]):
            completed_epics.append(epic_def["num"])
            print(f"✅ Epic {epic_def['num']} already completed")
    
    if completed_epics:
        print(f"📊 Found {len(completed_epics)} already completed epics: {completed_epics}")
        start_epic = max(completed_epics) + 1
        if start_epic > 10:
            print("🎉 All epics already completed!")
            return
    else:
        start_epic = 1
    
    print(f"🚀 Starting from Epic {start_epic}")
    
    # Launch epics sequentially
    for epic_def in EPIC_DEFINITIONS[start_epic-1:]:
        epic_num = epic_def["num"]
        
        # Launch the epic
        process = launch_epic(epic_def)
        if not process:
            print(f"❌ Failed to launch Epic {epic_num}, stopping series")
            break
        
        # Wait for completion
        if wait_for_epic_completion(epic_num, max_hours=12):
            completed_epics.append(epic_num)
            save_progress_log(completed_epics, epic_num, start_time)
            
            # Calculate progress
            progress = len(completed_epics) / 10 * 100
            elapsed = datetime.now() - start_time
            estimated_total = elapsed * (10 / len(completed_epics))
            remaining = estimated_total - elapsed
            
            print(f"📊 Progress: {progress:.1f}% ({len(completed_epics)}/10 epics)")
            print(f"⏱️ Elapsed: {elapsed}")
            print(f"⏱️ Estimated remaining: {remaining}")
            
        else:
            print(f"❌ Epic {epic_num} failed to complete, stopping series")
            break
    
    # Final summary
    total_time = datetime.now() - start_time
    print("\n" + "=" * 50)
    print("🎉 BREAKOUT EPIC SERIES COMPLETE!")
    print(f"✅ Completed epics: {len(completed_epics)}/10")
    print(f"⏱️ Total time: {total_time}")
    print(f"📹 Videos location: games/breakout/epic_XXX_*/videos/")
    print("=" * 50)

if __name__ == "__main__":
    main()
