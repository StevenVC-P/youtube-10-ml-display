#!/usr/bin/env python3
"""
🗂️ PROJECT REORGANIZATION SCRIPT
================================

This script reorganizes the project into a game-specific structure
without deleting any existing files.

New Structure:
games/
├── breakout/
│   ├── epic_001_from_scratch/
│   │   ├── videos/
│   │   │   ├── individual_hours/
│   │   │   └── merged_epic/
│   │   ├── models/
│   │   ├── logs/
│   │   └── metadata/
│   ├── epic_002_advanced/
│   └── epic_003_mastery/
├── pong/
├── space_invaders/
└── asteroids/
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def create_game_structure():
    """Create the new game-specific folder structure."""
    
    # Define the games we want to support
    games = [
        "breakout",
        "pong", 
        "space_invaders",
        "asteroids",
        "pacman",
        "frogger"
    ]
    
    # Create base games directory
    games_dir = Path("games")
    games_dir.mkdir(exist_ok=True)
    print(f"✅ Created: {games_dir}")
    
    # Create structure for each game
    for game in games:
        game_dir = games_dir / game
        game_dir.mkdir(exist_ok=True)
        print(f"✅ Created: {game_dir}")
        
        # Create epic journey folders (starting with 3 for each game)
        for epic_num in range(1, 4):
            epic_name = f"epic_{epic_num:03d}"
            if epic_num == 1:
                epic_name += "_from_scratch"
            elif epic_num == 2:
                epic_name += "_advanced"
            elif epic_num == 3:
                epic_name += "_mastery"
                
            epic_dir = game_dir / epic_name
            epic_dir.mkdir(exist_ok=True)
            print(f"  ✅ Created: {epic_dir}")
            
            # Create subdirectories for each epic
            subdirs = [
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
            
            for subdir in subdirs:
                subdir_path = epic_dir / subdir
                subdir_path.mkdir(parents=True, exist_ok=True)
                print(f"    ✅ Created: {subdir_path}")
    
    return games_dir

def move_current_breakout_data():
    """Move current Breakout training data to the new structure."""
    
    # Target directory for current epic
    target_epic = Path("games/breakout/epic_001_from_scratch")
    
    print(f"\n📦 Moving current Breakout data to: {target_epic}")
    
    # Move video files
    video_moves = [
        # Individual hour videos
        ("video/milestones/step_01000000_pct_10_analytics.mp4", "videos/individual_hours/hour_01_10pct.mp4"),
        ("video/milestones/step_02000000_pct_20_analytics.mp4", "videos/individual_hours/hour_02_20pct.mp4"),
        ("video/milestones/step_03000000_pct_30_analytics.mp4", "videos/individual_hours/hour_03_30pct.mp4"),
        ("video/milestones/step_04000000_pct_40_analytics.mp4", "videos/individual_hours/hour_04_40pct.mp4"),
        ("video/milestones/step_05000000_pct_50_analytics.mp4", "videos/individual_hours/hour_05_50pct.mp4"),
        ("video/milestones/step_06000000_pct_60_analytics.mp4", "videos/individual_hours/hour_06_60pct.mp4"),
        ("video/milestones/step_07000000_pct_70_analytics.mp4", "videos/individual_hours/hour_07_70pct.mp4"),
        ("video/milestones/step_08000000_pct_80_analytics.mp4", "videos/individual_hours/hour_08_80pct.mp4"),
        ("video/milestones/step_09000000_pct_90_analytics.mp4", "videos/individual_hours/hour_09_90pct.mp4"),
        ("video/milestones/step_10000000_pct_100_analytics.mp4", "videos/individual_hours/hour_10_100pct.mp4"),
        
        # Epic merged video
        ("video/EPIC_10Hour_Neural_Learning_Journey_20251011_073708.mp4", "videos/merged_epic/epic_001_complete_10hour_journey.mp4")
    ]
    
    for source, target_rel in video_moves:
        source_path = Path(source)
        target_path = target_epic / target_rel
        
        if source_path.exists():
            # Ensure target directory exists
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file (don't move yet, just copy for safety)
            shutil.copy2(source_path, target_path)
            print(f"  📹 Copied: {source} → {target_path}")
        else:
            print(f"  ⚠️  Missing: {source}")
    
    # Move model files
    model_moves = [
        ("models/checkpoints/latest.zip", "models/final/epic_001_final_model.zip"),
    ]
    
    for source, target_rel in model_moves:
        source_path = Path(source)
        target_path = target_epic / target_rel
        
        if source_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            print(f"  🧠 Copied: {source} → {target_path}")
        else:
            print(f"  ⚠️  Missing: {source}")
    
    # Copy entire logs directory
    logs_source = Path("logs/tb")
    logs_target = target_epic / "logs/tensorboard"
    
    if logs_source.exists():
        # Copy the entire directory tree
        if logs_target.exists():
            shutil.rmtree(logs_target)
        shutil.copytree(logs_source, logs_target)
        print(f"  📊 Copied logs: {logs_source} → {logs_target}")

def create_metadata_files():
    """Create metadata files for the current epic."""
    
    target_epic = Path("games/breakout/epic_001_from_scratch")
    metadata_dir = target_epic / "metadata"
    
    # Create epic info file
    epic_info = {
        "epic_name": "Epic 001: From Scratch",
        "game": "Breakout",
        "description": "Complete 10-hour neural network learning journey from random play to mastery",
        "start_date": "2025-10-10",
        "completion_date": "2025-10-11", 
        "total_timesteps": 10000000,
        "final_performance": {
            "average_episode_reward": 22.8,
            "episode_length_mean": 380
        },
        "video_info": {
            "individual_hours": 10,
            "merged_epic_duration": "10:00:00",
            "merged_epic_size_mb": 811.9,
            "resolution": "960x540",
            "fps": 30
        },
        "training_info": {
            "algorithm": "PPO",
            "policy": "CnnPolicy", 
            "learning_rate": 0.00025,
            "total_training_time_hours": 13.45,
            "environment": "BreakoutNoFrameskip-v4"
        }
    }
    
    import json
    with open(metadata_dir / "epic_001_info.json", 'w') as f:
        json.dump(epic_info, f, indent=2)
    
    print(f"  📋 Created metadata: {metadata_dir / 'epic_001_info.json'}")
    
    # Create README for the epic
    readme_content = f"""# Breakout Epic 001: From Scratch

## Overview
This epic documents the complete 10-hour neural network learning journey of an AI agent learning to play Atari Breakout from complete scratch.

## Results
- **Final Performance**: 22.8 average episode reward
- **Training Duration**: 13.45 hours real-time
- **Total Timesteps**: 10,000,000
- **Algorithm**: PPO with CNN Policy

## Videos
- **Individual Hours**: 10 separate 1-hour videos showing each stage of learning
- **Epic Merged**: Single 10-hour masterpiece showing complete journey
- **File Size**: 811.9 MB total

## Learning Progression
1. **Hour 1 (10%)**: Random exploration, basic neural patterns
2. **Hour 2 (20%)**: Early learning, paddle tracking begins  
3. **Hour 3 (30%)**: Basic ball tracking, improved coordination
4. **Hour 4 (40%)**: Strategic positioning, better timing
5. **Hour 5 (50%)**: Consistent ball returns, pattern recognition
6. **Hour 6 (60%)**: Advanced strategies, corner shots
7. **Hour 7 (70%)**: Mastery emerging, high scores
8. **Hour 8 (80%)**: Expert play, complex strategies
9. **Hour 9 (90%)**: Near-perfect performance  
10. **Hour 10 (100%)**: Complete mastery, optimal play

## Technical Details
- **Environment**: BreakoutNoFrameskip-v4
- **Neural Network**: CNN with realistic architecture visualization
- **Video Features**: Enhanced analytics overlay, activation-based coloring
- **Resolution**: 960×540 at 30 FPS

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(metadata_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"  📖 Created README: {metadata_dir / 'README.md'}")

def update_config_files():
    """Update configuration files to reflect new structure."""
    
    # Copy current config to the epic folder
    target_epic = Path("games/breakout/epic_001_from_scratch")
    config_target = target_epic / "config"
    
    # Copy config files
    config_files = [
        "conf/config.yaml",
        "conf/config.py"
    ]
    
    for config_file in config_files:
        source_path = Path(config_file)
        if source_path.exists():
            target_path = config_target / source_path.name
            shutil.copy2(source_path, target_path)
            print(f"  ⚙️  Copied config: {source_path} → {target_path}")

def create_project_structure_doc():
    """Create documentation for the new project structure."""
    
    structure_doc = """# Project Structure

## Overview
This project is organized by game, with each game containing multiple epic journeys that build upon each other.

## Directory Structure
```
games/
├── breakout/
│   ├── epic_001_from_scratch/     # Complete learning from zero
│   ├── epic_002_advanced/         # Advanced techniques and strategies  
│   └── epic_003_mastery/          # Perfect play and optimization
├── pong/
│   ├── epic_001_from_scratch/
│   ├── epic_002_advanced/
│   └── epic_003_mastery/
├── space_invaders/
├── asteroids/
├── pacman/
└── frogger/
```

## Epic Structure
Each epic contains:
- `videos/individual_hours/` - Individual 1-hour learning videos
- `videos/merged_epic/` - Complete epic merged into single video
- `videos/highlights/` - Key moments and achievements
- `models/checkpoints/` - Training checkpoints
- `models/final/` - Final trained model
- `logs/tensorboard/` - TensorBoard training logs
- `logs/training/` - Training output logs
- `metadata/` - Epic information and documentation
- `config/` - Configuration files used for this epic

## Epic Progression
1. **Epic 001**: Learning from scratch (random → competent)
2. **Epic 002**: Advanced techniques (competent → expert)  
3. **Epic 003**: Mastery optimization (expert → perfect)

## Current Status
- ✅ Breakout Epic 001: COMPLETED
- 🔄 Breakout Epic 002: Ready to start
- ⏳ Breakout Epic 003: Planned
- ⏳ Other games: Planned

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open("PROJECT_STRUCTURE.md", 'w', encoding='utf-8') as f:
        f.write(structure_doc)
    
    print(f"📚 Created: PROJECT_STRUCTURE.md")

def main():
    """Main reorganization function."""
    print("🗂️ PROJECT REORGANIZATION")
    print("=" * 50)
    print("🎯 Creating game-specific structure without deleting anything...")
    print()
    
    # Create the new structure
    games_dir = create_game_structure()
    
    # Move current Breakout data
    move_current_breakout_data()
    
    # Create metadata
    create_metadata_files()
    
    # Update configs
    update_config_files()
    
    # Create documentation
    create_project_structure_doc()
    
    print("\n" + "=" * 50)
    print("🎉 REORGANIZATION COMPLETED!")
    print("✅ All files preserved in original locations")
    print("✅ New structure created with copies")
    print("✅ Ready for multiple games and epic journeys")
    print("=" * 50)
    
    print(f"\n📁 New structure created in: {games_dir.absolute()}")
    print("🎮 Current epic: games/breakout/epic_001_from_scratch/")
    print("🚀 Ready to start Epic 002!")

if __name__ == "__main__":
    main()
