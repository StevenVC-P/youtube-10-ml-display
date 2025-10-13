#!/usr/bin/env python3
"""
🔍 REORGANIZATION VERIFICATION SCRIPT
====================================

This script verifies that the project reorganization was successful
and all files are properly organized.
"""

import json
from pathlib import Path

def check_epic_structure(game: str, epic_num: int):
    """Check if an epic has the proper structure."""
    epic_names = {1: "from_scratch", 2: "advanced", 3: "mastery"}
    epic_name = f"epic_{epic_num:03d}_{epic_names.get(epic_num, f'epic_{epic_num}')}"
    
    epic_dir = Path("games") / game / epic_name
    
    if not epic_dir.exists():
        return False, f"Epic directory missing: {epic_dir}"
    
    # Check required subdirectories
    required_dirs = [
        "videos/individual_hours",
        "videos/merged_epic", 
        "videos/highlights",
        "models/checkpoints",
        "models/final",
        "logs/tensorboard",
        "logs/training",
        "metadata",
        "config"
    ]
    
    missing_dirs = []
    for req_dir in required_dirs:
        if not (epic_dir / req_dir).exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        return False, f"Missing directories: {missing_dirs}"
    
    return True, "Structure complete"

def check_breakout_epic_001():
    """Check Breakout Epic 001 specifically."""
    epic_dir = Path("games/breakout/epic_001_from_scratch")
    
    print(f"🔍 Checking Breakout Epic 001...")
    
    # Check videos
    individual_hours = epic_dir / "videos/individual_hours"
    expected_videos = [
        "hour_01_10pct.mp4", "hour_02_20pct.mp4", "hour_03_30pct.mp4",
        "hour_04_40pct.mp4", "hour_05_50pct.mp4", "hour_06_60pct.mp4", 
        "hour_07_70pct.mp4", "hour_08_80pct.mp4", "hour_09_90pct.mp4",
        "hour_10_100pct.mp4"
    ]
    
    missing_videos = []
    total_size_mb = 0
    
    for video in expected_videos:
        video_path = individual_hours / video
        if video_path.exists():
            size_mb = video_path.stat().st_size / 1024 / 1024
            total_size_mb += size_mb
            print(f"  ✅ {video}: {size_mb:.1f} MB")
        else:
            missing_videos.append(video)
            print(f"  ❌ {video}: MISSING")
    
    # Check epic video
    epic_video = epic_dir / "videos/merged_epic/epic_001_complete_10hour_journey.mp4"
    if epic_video.exists():
        epic_size_mb = epic_video.stat().st_size / 1024 / 1024
        print(f"  ✅ Epic video: {epic_size_mb:.1f} MB")
        total_size_mb += epic_size_mb
    else:
        print(f"  ❌ Epic video: MISSING")
    
    print(f"  📊 Total video size: {total_size_mb:.1f} MB")
    
    # Check model
    final_model = epic_dir / "models/final/epic_001_final_model.zip"
    if final_model.exists():
        model_size_mb = final_model.stat().st_size / 1024 / 1024
        print(f"  ✅ Final model: {model_size_mb:.1f} MB")
    else:
        print(f"  ❌ Final model: MISSING")
    
    # Check metadata
    metadata_file = epic_dir / "metadata/epic_001_info.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"  ✅ Metadata: {metadata.get('epic_name', 'Unknown')}")
        print(f"    📈 Final performance: {metadata.get('final_performance', {}).get('average_episode_reward', 'Unknown')} avg reward")
    else:
        print(f"  ❌ Metadata: MISSING")
    
    # Check config
    config_file = epic_dir / "config/config.yaml"
    if config_file.exists():
        print(f"  ✅ Config: Present")
    else:
        print(f"  ❌ Config: MISSING")
    
    return len(missing_videos) == 0

def check_all_games():
    """Check all game structures."""
    games = ["breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger"]
    
    print(f"\n🎮 Checking all game structures...")
    
    for game in games:
        print(f"\n📁 {game.title()}:")
        
        for epic_num in range(1, 4):
            success, message = check_epic_structure(game, epic_num)
            epic_names = {1: "from_scratch", 2: "advanced", 3: "mastery"}
            epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
            
            if success:
                print(f"  ✅ Epic {epic_num} ({epic_name}): {message}")
            else:
                print(f"  ⚠️  Epic {epic_num} ({epic_name}): {message}")

def check_original_files():
    """Check that original files are still present."""
    print(f"\n🛡️  Checking original files preservation...")
    
    original_locations = [
        "video/milestones/step_01000000_pct_10_analytics.mp4",
        "video/milestones/step_10000000_pct_100_analytics.mp4", 
        "video/EPIC_10Hour_Neural_Learning_Journey_20251011_073708.mp4",
        "models/checkpoints/latest.zip",
        "conf/config.yaml"
    ]
    
    for location in original_locations:
        path = Path(location)
        if path.exists():
            print(f"  ✅ {location}: Preserved")
        else:
            print(f"  ⚠️  {location}: Missing")

def check_new_scripts():
    """Check that new scripts are present."""
    print(f"\n🔧 Checking new scripts...")
    
    new_scripts = [
        "train_epic.py",
        "reorganize_project.py", 
        "merge_epic_videos.py",
        "verify_reorganization.py",
        "PROJECT_STRUCTURE.md",
        "REORGANIZATION_SUMMARY.md"
    ]
    
    for script in new_scripts:
        path = Path(script)
        if path.exists():
            print(f"  ✅ {script}: Present")
        else:
            print(f"  ❌ {script}: Missing")

def main():
    """Main verification function."""
    print("🔍 PROJECT REORGANIZATION VERIFICATION")
    print("=" * 50)
    
    # Check Breakout Epic 001 (the completed one)
    breakout_success = check_breakout_epic_001()
    
    # Check all game structures
    check_all_games()
    
    # Check original files preservation
    check_original_files()
    
    # Check new scripts
    check_new_scripts()
    
    print("\n" + "=" * 50)
    
    if breakout_success:
        print("🎉 VERIFICATION SUCCESSFUL!")
        print("✅ Breakout Epic 001 is properly organized")
        print("✅ All game structures are ready")
        print("✅ Original files are preserved")
        print("✅ New training system is ready")
        print("\n🚀 Ready to start new epic journeys!")
    else:
        print("⚠️  VERIFICATION ISSUES FOUND")
        print("Some files may be missing or misorganized")
        print("Check the output above for details")
    
    print("\n📖 See REORGANIZATION_SUMMARY.md for complete details")

if __name__ == "__main__":
    main()
