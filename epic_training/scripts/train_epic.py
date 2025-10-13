#!/usr/bin/env python3
"""
🎮 EPIC TRAINING LAUNCHER
========================

This script launches epic training sessions for different games and epic levels.
It automatically organizes outputs into the new game-specific structure.

Usage:
    python train_epic.py --game breakout --epic 2
    python train_epic.py --game pong --epic 1 --hours 10
    python train_epic.py --game space_invaders --epic 1 --test
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def get_epic_info(game: str, epic_num: int):
    """Get information about an epic."""
    epic_names = {
        1: "from_scratch",
        2: "advanced", 
        3: "mastery"
    }
    
    epic_descriptions = {
        1: "Complete learning from zero - random play to competent performance",
        2: "Advanced techniques and strategies - competent to expert level",
        3: "Mastery optimization - expert to perfect play"
    }
    
    epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
    epic_desc = epic_descriptions.get(epic_num, f"Epic {epic_num} training")
    
    return f"epic_{epic_num:03d}_{epic_name}", epic_desc

def get_game_env_id(game: str):
    """Get the environment ID for a game."""
    env_mapping = {
        "breakout": "BreakoutNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4", 
        "asteroids": "AsteroidsNoFrameskip-v4",
        "pacman": "MsPacmanNoFrameskip-v4",
        "frogger": "FroggerNoFrameskip-v4"
    }
    
    return env_mapping.get(game.lower(), f"{game.title()}NoFrameskip-v4")

def setup_epic_directories(game: str, epic_num: int):
    """Set up directories for the epic and return paths."""
    epic_name, _ = get_epic_info(game, epic_num)
    
    # Create epic directory structure
    epic_dir = Path("games") / game.lower() / epic_name
    epic_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all subdirectories exist
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
        (epic_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return {
        'epic_dir': epic_dir,
        'videos_dir': epic_dir / "videos",
        'models_dir': epic_dir / "models",
        'logs_dir': epic_dir / "logs",
        'config_dir': epic_dir / "config",
        'metadata_dir': epic_dir / "metadata"
    }

def create_epic_config(game: str, epic_num: int, paths: dict, hours: int = 10):
    """Create configuration for the epic."""
    epic_name, epic_desc = get_epic_info(game, epic_num)
    env_id = get_game_env_id(game)
    
    # Load base config
    base_config_path = Path("conf/config.yaml")
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            config_content = f.read()
        
        # Update paths for this epic
        config_content = config_content.replace(
            'videos_milestones: "video/milestones"',
            f'videos_milestones: "{paths["videos_dir"]}/individual_hours"'
        )
        config_content = config_content.replace(
            'logs_tb: "logs/tb"',
            f'logs_tb: "{paths["logs_dir"]}/tensorboard"'
        )
        config_content = config_content.replace(
            'models: "models/checkpoints"',
            f'models: "{paths["models_dir"]}/checkpoints"'
        )
        config_content = config_content.replace(
            'env_id: "BreakoutNoFrameskip-v4"',
            f'env_id: "{env_id}"'
        )
        
        # Save epic-specific config
        epic_config_path = paths['config_dir'] / "config.yaml"
        with open(epic_config_path, 'w') as f:
            f.write(config_content)
        
        print(f"⚙️  Created epic config: {epic_config_path}")
        return epic_config_path
    else:
        print("⚠️  Base config not found, using defaults")
        return None

def create_epic_metadata(game: str, epic_num: int, paths: dict, hours: int):
    """Create metadata for the epic."""
    epic_name, epic_desc = get_epic_info(game, epic_num)
    
    metadata = {
        "epic_name": f"Epic {epic_num:03d}: {epic_desc.split(' - ')[0]}",
        "game": game.title(),
        "description": epic_desc,
        "start_date": datetime.now().strftime("%Y-%m-%d"),
        "planned_hours": hours,
        "status": "in_progress",
        "environment": get_game_env_id(game),
        "epic_number": epic_num
    }
    
    import json
    metadata_path = paths['metadata_dir'] / f"epic_{epic_num:03d}_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Created metadata: {metadata_path}")

def launch_training(game: str, epic_num: int, hours: int = 10, test_mode: bool = False):
    """Launch the epic training session."""
    print(f"🎮 LAUNCHING EPIC TRAINING")
    print(f"=" * 50)
    
    epic_name, epic_desc = get_epic_info(game, epic_num)
    env_id = get_game_env_id(game)
    
    print(f"🎯 Game: {game.title()}")
    print(f"🚀 Epic: {epic_name}")
    print(f"📖 Description: {epic_desc}")
    print(f"🎮 Environment: {env_id}")
    print(f"⏱️  Duration: {hours} hours")
    print(f"🧪 Test Mode: {test_mode}")
    print()
    
    # Set up directories
    paths = setup_epic_directories(game, epic_num)
    print(f"📁 Epic directory: {paths['epic_dir']}")
    
    # Create configuration
    config_path = create_epic_config(game, epic_num, paths, hours)
    
    # Create metadata
    create_epic_metadata(game, epic_num, paths, hours)
    
    # Prepare training command
    if test_mode:
        # Short test run
        cmd = [
            sys.executable, "training/train.py",
            "--dryrun-seconds", "60",  # 1 minute test
            "--verbose", "1"
        ]
        print(f"🧪 Running 1-minute test...")
    else:
        # Full epic training
        cmd = [
            sys.executable, "training/train.py", 
            "--verbose", "1"
        ]
        print(f"🚀 Starting {hours}-hour epic training...")
    
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Set environment variables for the training script
    env = os.environ.copy()
    env['EPIC_GAME'] = game
    env['EPIC_NUMBER'] = str(epic_num)
    env['EPIC_DIR'] = str(paths['epic_dir'])
    
    # Launch training
    import subprocess
    try:
        process = subprocess.run(cmd, env=env, cwd=Path.cwd())
        
        if process.returncode == 0:
            print("🎉 EPIC TRAINING COMPLETED!")
            
            # Update metadata
            metadata_path = paths['metadata_dir'] / f"epic_{epic_num:03d}_info.json"
            if metadata_path.exists():
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = 'completed'
                metadata['completion_date'] = datetime.now().strftime("%Y-%m-%d")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        else:
            print(f"❌ Training failed with return code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Launch epic training sessions")
    parser.add_argument("--game", required=True, 
                       choices=["breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger"],
                       help="Game to train on")
    parser.add_argument("--epic", type=int, required=True, choices=[1, 2, 3],
                       help="Epic number (1=from_scratch, 2=advanced, 3=mastery)")
    parser.add_argument("--hours", type=int, default=10,
                       help="Training duration in hours (default: 10)")
    parser.add_argument("--test", action="store_true",
                       help="Run a short test instead of full training")
    
    args = parser.parse_args()
    
    # Validate epic progression
    if args.epic > 1:
        prev_epic_dir = Path("games") / args.game / f"epic_{args.epic-1:03d}_*"
        prev_epics = list(Path("games").glob(f"{args.game}/epic_{args.epic-1:03d}_*"))
        if not prev_epics:
            print(f"⚠️  Warning: Epic {args.epic-1} not found for {args.game}")
            print(f"   Consider running Epic {args.epic-1} first for proper progression")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                return
    
    launch_training(args.game, args.epic, args.hours, args.test)

if __name__ == "__main__":
    main()
