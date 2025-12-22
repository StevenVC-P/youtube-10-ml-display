#!/usr/bin/env python3
"""
🎮 EPIC CONTINUOUS TRAINING LAUNCHER
===================================

This script implements proper epic continuity where each epic builds upon the previous one:
- Epic 1: From scratch (0% → 100% competency)
- Epic 2: Advanced training (100% competency → expert level)
- Epic 3: Mastery training (expert → perfect play)

Each epic loads the final model from the previous epic and continues training,
creating a true progression narrative in the videos.

Usage:
    python train_epic_continuous.py --game breakout --epic 2
    python train_epic_continuous.py --game breakout --epic 3 --hours 10
    python train_epic_continuous.py --game breakout --epic 2 --test
    python train_epic_continuous.py --game breakout --epic 1 --fast  (Train fast, render later)
"""

import os
import sys
import argparse
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def get_epic_info(game: str, epic_num: int):
    """Get information about an epic with proper progression descriptions."""
    epic_names = {
        1: "from_scratch",
        2: "advanced_mastery",
        3: "perfect_play",
        4: "speed_demon",
        5: "precision_master",
        6: "strategic_genius",
        7: "endurance_champion",
        8: "adaptive_expert",
        9: "ultimate_mastery",
        10: "legendary_status"
    }

    epic_descriptions = {
        1: "Complete learning from zero - random play to competent performance",
        2: "Advanced mastery - building on competent play to reach expert level",
        3: "Perfect play optimization - expert level to flawless execution",
        4: "Speed demon training - optimize for fast completion and efficiency",
        5: "Precision master - perfect ball control and strategic positioning",
        6: "Strategic genius - advanced game strategies and pattern recognition",
        7: "Endurance champion - long-term consistency and sustained performance",
        8: "Adaptive expert - handle game variations and edge cases",
        9: "Ultimate mastery - peak performance across all game scenarios",
        10: "Legendary status - transcendent gameplay beyond human capability"
    }
    
    epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
    epic_desc = epic_descriptions.get(epic_num, f"Epic {epic_num} training")
    
    return f"epic_{epic_num:03d}_{epic_name}", epic_desc

def get_game_env_id(game: str):
    """Get the environment ID for a game."""
    env_mapping = {
        # Atari Games (ALE)
        "breakout": "BreakoutNoFrameskip-v4",
        "pong": "PongNoFrameskip-v4",
        "space_invaders": "SpaceInvadersNoFrameskip-v4",
        "asteroids": "AsteroidsNoFrameskip-v4",
        "pacman": "MsPacmanNoFrameskip-v4",
        "frogger": "FroggerNoFrameskip-v4",
        
        # Gameboy Games (tetris-gymnasium - coded remake)
        "tetris": "tetris_gymnasium/Tetris",
        
        # Gameboy Games (PyBoy - authentic emulation)
        "tetris_authentic": "tetris_gb_authentic",
        "tetris_gb_authentic": "tetris_gb_authentic",
        "mario_land_authentic": "mario_land_authentic",
        "super_mario_land_authentic": "super_mario_land_authentic",
        "kirby_authentic": "kirby_authentic",
        "kirbys_dream_land_authentic": "kirbys_dream_land_authentic",
        
        # Gameboy Games (stable-retro - if available)
        "tetris_gb": "Tetris-GameBoy",
        "super_mario_land": "SuperMarioLand-GameBoy",
        "kirbys_dream_land": "KirbysDreamLand-GameBoy",
        "mario_land": "SuperMarioLand-GameBoy",
        "kirby": "KirbysDreamLand-GameBoy"
    }
    
    return env_mapping.get(game.lower(), f"{game.title()}NoFrameskip-v4")

def find_previous_epic_model(game: str, epic_num: int):
    """Find the final model from the previous epic."""
    if epic_num <= 1:
        return None
    
    prev_epic_num = epic_num - 1
    prev_epic_pattern = f"epic_{prev_epic_num:03d}_*"
    
    # Find previous epic directory
    game_dir = Path("games") / game.lower()
    prev_epic_dirs = list(game_dir.glob(prev_epic_pattern))
    
    if not prev_epic_dirs:
        print(f"❌ Previous epic {prev_epic_num} not found for {game}")
        return None
    
    prev_epic_dir = prev_epic_dirs[0]
    
    # Look for final model
    final_model_path = prev_epic_dir / "models" / "final" / f"epic_{prev_epic_num:03d}_final_model.zip"
    
    if final_model_path.exists():
        print(f"✅ Found previous epic model: {final_model_path}")
        return str(final_model_path)
    
    # Fallback: look for latest checkpoint
    checkpoint_dir = prev_epic_dir / "models" / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.zip"
    
    if latest_checkpoint.exists():
        print(f"✅ Found previous epic checkpoint: {latest_checkpoint}")
        return str(latest_checkpoint)
    
    print(f"❌ No model found in previous epic {prev_epic_num}")
    return None

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

def create_epic_config(game: str, epic_num: int, paths: dict, hours: int = 10, previous_model: str = None, fast_mode: bool = False):
    """Create configuration for the epic with optional model loading."""
    epic_name, epic_desc = get_epic_info(game, epic_num)
    env_id = get_game_env_id(game)
    
    # Load base config
    base_config_path = Path("conf/config.yaml")
    if base_config_path.exists():
        with open(base_config_path, 'r') as f:
            config_content = f.read()
        
        # Update paths for this epic (convert to forward slashes for YAML compatibility)
        config_content = config_content.replace(
            'videos_milestones: "video/milestones"',
            f'videos_milestones: "{str(paths["videos_dir"]).replace(chr(92), "/")}/individual_hours"'
        )
        config_content = config_content.replace(
            'logs_tb: "logs/tb"',
            f'logs_tb: "{str(paths["logs_dir"]).replace(chr(92), "/")}/tensorboard"'
        )
        config_content = config_content.replace(
            'models: "models/checkpoints"',
            f'models: "{str(paths["models_dir"]).replace(chr(92), "/")}/checkpoints"'
        )
        config_content = config_content.replace(
            'env_id: "BreakoutNoFrameskip-v4"',
            f'env_id: "{env_id}"'
        )
        
        # Add previous model path if available
        if previous_model:
            # Add a new section for model loading (use forward slashes for YAML compatibility)
            previous_model_yaml = previous_model.replace('\\', '/')
            config_content += f'\\n\\n# Epic Continuity\\nepic:\\n  load_previous_model: "{previous_model_yaml}"\\n  epic_number: {epic_num}\\n'
            
        # Fast Mode Overrides
        if fast_mode:
            print("🚀 Fast Mode: Disabling live video recording to maximize training speed")
            config_content += '\\n\\n# Fast Mode Overrides\\nrecording:\\n  milestone_clip_seconds: 0  # Disable partial clips\\n\\ntraining_video:\\n  enabled: false  # Disable smart recording\\n'
        
        # Save epic-specific config
        epic_config_path = paths['config_dir'] / "config.yaml"
        with open(epic_config_path, 'w') as f:
            f.write(config_content)
        
        print(f"⚙️  Created epic config: {epic_config_path}")
        if previous_model:
            print(f"🔗 Will load previous model: {previous_model}")
        return epic_config_path
    else:
        print("⚠️  Base config not found, using defaults")
        return None

def create_epic_metadata(game: str, epic_num: int, paths: dict, hours: int, previous_model: str = None):
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
        "epic_number": epic_num,
        "continues_from_epic": epic_num - 1 if epic_num > 1 else None,
        "previous_model": previous_model,
        "training_type": "continuation" if previous_model else "from_scratch"
    }
    
    metadata_path = paths['metadata_dir'] / f"epic_{epic_num:03d}_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Created metadata: {metadata_path}")

def validate_epic_progression(game: str, epic_num: int):
    """Validate that previous epics exist for proper progression."""
    if epic_num <= 1:
        return True
    
    # Check if previous epic exists
    prev_epic_num = epic_num - 1
    prev_epic_pattern = f"epic_{prev_epic_num:03d}_*"
    game_dir = Path("games") / game.lower()
    prev_epics = list(game_dir.glob(prev_epic_pattern))
    
    if not prev_epics:
        print(f"❌ Epic {prev_epic_num} not found for {game}!")
        print(f"   Epic {epic_num} requires Epic {prev_epic_num} to be completed first.")
        print(f"   Please run Epic {prev_epic_num} before attempting Epic {epic_num}.")
        return False
    
    # Check if previous epic has a final model
    prev_epic_dir = prev_epics[0]
    final_model_path = prev_epic_dir / "models" / "final" / f"epic_{prev_epic_num:03d}_final_model.zip"
    latest_checkpoint = prev_epic_dir / "models" / "checkpoints" / "latest.zip"
    
    if not final_model_path.exists() and not latest_checkpoint.exists():
        print(f"⚠️  Epic {prev_epic_num} exists but has no trained model!")
        print(f"   Epic {epic_num} needs a trained model from Epic {prev_epic_num} to continue.")
        response = input(f"Continue anyway (will start from scratch)? (y/n): ")
        return response.lower() == 'y'
    
    return True

def run_post_training_render(game: str, epic_num: int, paths: dict, config_path: Path, hours: int = 10):
    """Run post-training video generation."""
    print(f"\\n🎥 STARTING POST-TRAINING VIDEO RENDER")
    print(f"=" * 60)
    
    # Find post-training generator script
    # It should be in the training directory relative to project root
    script_path = Path("training") / "post_training_video_generator.py"
    
    if not script_path.exists():
        print(f"❌ Does not exist: {script_path}")
        # Try finding it relative to this script
        script_path = Path(__file__).parent.parent.parent / "training" / "post_training_video_generator.py"
        if not script_path.exists():
             print(f"❌ Cannot find post-training generator script at {script_path}")
             return False

    checkpoint_dir = paths['models_dir'] / "checkpoints"
    
    # Determine full output video path (Video/Merged Epic)
    output_dir = paths['videos_dir'] / "merged_epic"
    
    total_seconds = hours * 3600
    
    cmd = [
        sys.executable, str(script_path),
        "--model-dir", str(checkpoint_dir),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--total-seconds", str(total_seconds)  # Generate single continuous video
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(cmd, cwd=Path.cwd())
        
        if process.returncode == 0:
            print("✅ Post-training render completed successfully!")
            return True
        else:
            print(f"❌ Post-training render failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error during post-training render: {e}")
        return False

def launch_continuous_training(game: str, epic_num: int, hours: int = 10, test_mode: bool = False, fast_mode: bool = False, render_only: bool = False):
    """Launch the epic training session with proper continuity."""
    print(f"🎮 LAUNCHING CONTINUOUS EPIC TRAINING")
    print(f"=" * 60)
    
    epic_name, epic_desc = get_epic_info(game, epic_num)
    env_id = get_game_env_id(game)
    
    print(f"🎯 Game: {game.title()}")
    print(f"🚀 Epic: {epic_name}")
    print(f"📖 Description: {epic_desc}")
    print(f"🎮 Environment: {env_id}")
    if not render_only:
        print(f"⏱️  Duration: {hours} hours")
    print(f"🧪 Test Mode: {test_mode}")
    print(f"⚡ Fast Mode: {fast_mode}")
    print(f"🎥 Render Only: {render_only}")
    
    # Set up directories
    paths = setup_epic_directories(game, epic_num)
    
    # Validate epic progression (skip if render only)
    if not render_only and not validate_epic_progression(game, epic_num):
        return False
        
    # Find previous model if this is a continuation epic
    previous_model = find_previous_epic_model(game, epic_num) if not render_only else None
    
    # Create configuration
    config_path = create_epic_config(game, epic_num, paths, hours, previous_model, fast_mode)
    if not config_path:
        return False
        
    if render_only:
        # Just run the post-training render
        return run_post_training_render(game, epic_num, paths, config_path, hours)
    
    # Normal training flow
    if epic_num > 1:
        if previous_model:
            print(f"🔗 Continuation Mode: Will load model from Epic {epic_num-1}")
            print(f"📈 Expected progression: Expert → Master level performance")
        else:
            print(f"⚠️  No previous model found - starting from scratch")
            print(f"📈 Will show: Beginner → Expert level (not true continuation)")
    else:
        print(f"🆕 Fresh Start: Training from random initialization")
        print(f"📈 Expected progression: Random → Competent performance")
        
    print()
    print(f"📁 Epic directory: {paths['epic_dir']}")
    
    # Create metadata
    create_epic_metadata(game, epic_num, paths, hours, previous_model)
    
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
    print("=" * 60)
    
    # Set environment variables for the training script
    env = os.environ.copy()
    env['EPIC_GAME'] = game
    env['EPIC_NUMBER'] = str(epic_num)
    env['EPIC_DIR'] = str(paths['epic_dir'])
    if previous_model:
        env['EPIC_PREVIOUS_MODEL'] = previous_model
        
    # Launch training
    try:
        process = subprocess.run(cmd, env=env, cwd=Path.cwd())
        
        if process.returncode == 0:
            print("🎉 EPIC TRAINING COMPLETED!")
            
            # Save final model
            final_model_path = paths['models_dir'] / "final" / f"epic_{epic_num:03d}_final_model.zip"
            latest_checkpoint = paths['models_dir'] / "checkpoints" / "latest.zip"
            
            if latest_checkpoint.exists():
                # Copy latest checkpoint as final model
                final_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(latest_checkpoint), str(final_model_path))
                print(f"💾 Final model saved: {final_model_path}")
            
            # Update metadata
            metadata_path = paths['metadata_dir'] / f"epic_{epic_num:03d}_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                metadata['status'] = 'completed'
                metadata['completion_date'] = datetime.now().strftime("%Y-%m-%d")
                metadata['final_model_path'] = str(final_model_path)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            print(f"📋 Epic {epic_num} ready for next epic continuation!")
            
            # If Fast Mode was used, trigger post-training render
            if fast_mode:
                print("\\n🚀 Fast Mode: Triggering automatic post-training render...")
                run_post_training_render(game, epic_num, paths, config_path, hours)
            
            return True
        else:
            print(f"❌ Training failed with return code: {process.returncode}")
            return False
            
    except KeyboardInterrupt:
        print("\\n⏹️  Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Launch continuous epic training sessions")
    parser.add_argument("--game", required=True, 
                        choices=["breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger", 
                                "tetris", "tetris_authentic", "tetris_gb_authentic",
                                "mario_land_authentic", "super_mario_land_authentic",
                                "kirby_authentic", "kirbys_dream_land_authentic",
                                "tetris_gb", "super_mario_land", "kirbys_dream_land", "mario_land", "kirby"],
                        help="Game to train on")
    parser.add_argument("--epic", type=int, required=True, choices=list(range(1, 11)),
                        help="Epic number (1-10: from_scratch to legendary_status)")
    parser.add_argument("--hours", type=int, default=10,
                        help="Training duration in hours (default: 10)")
    parser.add_argument("--test", action="store_true",
                        help="Run a short test instead of full training")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: disable live videos, train faster, render after")
    parser.add_argument("--render-only", action="store_true",
                        help="Skip training and just run the video generator for an existing epic")
    
    args = parser.parse_args()
    
    success = launch_continuous_training(
        args.game, 
        args.epic, 
        args.hours, 
        args.test,
        args.fast,
        args.render_only
    )
    
    if success:
        if args.render_only:
             print(f"\\n✅ Epic {args.epic} video render completed successfully!")
        else:
            print(f"\\n✅ Epic {args.epic} completed successfully!")
            if args.epic < 3:
                print(f"🚀 Ready to run Epic {args.epic + 1} for continued progression!")
                print(f"   Command: python train_epic_continuous.py --game {args.game} --epic {args.epic + 1}")
    else:
        print(f"\\n❌ Epic {args.epic} failed or was cancelled.")

if __name__ == "__main__":
    main()
