#!/usr/bin/env python3
"""
🎮 CUSTOM TRAINING COMMANDS
==========================

This script provides flexible, parameterized commands for training different games
and generating videos of various lengths across multiple gaming systems.

Usage Examples:
    # Train any game for custom duration
    python scripts/custom_training_commands.py train --game tetris --hours 5 --system gameboy
    python scripts/custom_training_commands.py train --game breakout --hours 2 --system atari
    
    # Generate videos of specific lengths
    python scripts/custom_training_commands.py video --game space_invaders --length 30min --quality high
    python scripts/custom_training_commands.py video --game tetris --length 2h --epic 1
    
    # Batch operations
    python scripts/custom_training_commands.py batch --games "breakout,tetris,pong" --hours 1 --test
    
    # System-specific training
    python scripts/custom_training_commands.py system-train --system atari --epic 1 --hours 8
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Game system mappings
GAME_SYSTEMS = {
    # Atari Games
    "breakout": "atari",
    "pong": "atari",
    "space_invaders": "atari",
    "asteroids": "atari",
    "pacman": "atari",
    "frogger": "atari",

    # Gameboy Games (Coded Remakes)
    "tetris": "gameboy",

    # Gameboy Games (Authentic Emulation)
    "tetris_authentic": "gameboy_authentic",
    "mario_land_authentic": "gameboy_authentic",
    "super_mario_land_authentic": "gameboy_authentic",
    "kirby_authentic": "gameboy_authentic",
    "kirbys_dream_land_authentic": "gameboy_authentic",
}

SYSTEM_CONFIGS = {
    "atari": {
        "base_config": "conf/config.yaml",
        "epic_script": "epic_training/scripts/train_epic_continuous.py",
        "supported_games": ["breakout", "pong", "space_invaders", "asteroids", "pacman", "frogger"]
    },
    "gameboy": {
        "base_config": "conf/config.yaml",
        "epic_script": "epic_training/scripts/train_epic_continuous.py",
        "supported_games": ["tetris"]
    },
    "gameboy_authentic": {
        "base_config": "conf/config.yaml",
        "epic_script": "epic_training/scripts/train_epic_continuous.py",
        "supported_games": ["tetris_authentic", "mario_land_authentic", "super_mario_land_authentic", "kirby_authentic", "kirbys_dream_land_authentic"]
    }
}

def parse_duration(duration_str: str) -> float:
    """Parse duration string to hours. Examples: '30sec', '30min', '2h', '90m', '1.5h'"""
    duration_str = duration_str.lower().strip()

    if duration_str.endswith('sec') or duration_str.endswith('s'):
        seconds = float(duration_str.rstrip('sec').rstrip('s'))
        return seconds / 3600.0  # Convert seconds to hours
    elif duration_str.endswith('min') or duration_str.endswith('m'):
        minutes = float(duration_str.rstrip('min').rstrip('m'))
        return minutes / 60.0
    elif duration_str.endswith('h') or duration_str.endswith('hour') or duration_str.endswith('hours'):
        return float(duration_str.rstrip('hours').rstrip('hour').rstrip('h'))
    else:
        # Assume hours if no unit
        return float(duration_str)

def validate_game_system(game: str, system: Optional[str] = None) -> str:
    """Validate game and return its system."""
    if game not in GAME_SYSTEMS:
        available_games = list(GAME_SYSTEMS.keys())
        raise ValueError(f"Game '{game}' not supported. Available: {available_games}")
    
    detected_system = GAME_SYSTEMS[game]
    
    if system and system != detected_system:
        raise ValueError(f"Game '{game}' belongs to '{detected_system}' system, not '{system}'")
    
    return detected_system

def run_training_command(game: str, hours: float, epic: int = 1, test: bool = False, 
                        system: Optional[str] = None, verbose: bool = True) -> bool:
    """Run training command for specified game and duration."""
    
    # Validate game and system
    actual_system = validate_game_system(game, system)
    system_config = SYSTEM_CONFIGS[actual_system]
    
    if game not in system_config["supported_games"]:
        raise ValueError(f"Game '{game}' not yet implemented for {actual_system} system")
    
    # Build command
    cmd = [
        sys.executable,
        system_config["epic_script"],
        "--game", game,
        "--epic", str(epic),
        "--hours", str(int(hours))
    ]
    
    if test:
        cmd.append("--test")
    
    if verbose:
        print(f"🎮 Training {game} ({actual_system}) for {hours}h (Epic {epic})")
        print(f"Command: {' '.join(cmd)}")
    
    # Execute command
    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def run_video_generation(game: str, length: str, epic: int = 1, quality: str = "medium") -> bool:
    """Generate video of specific length for a game."""

    hours = parse_duration(length)
    system = validate_game_system(game)

    print(f"📹 Generating {length} video for {game} ({system}) - Epic {epic}")

    # For now, use evaluation to generate videos
    # In the future, this could use dedicated video generation tools

    model_path = f"games/{game}/epic_{epic:03d}_from_scratch/models/final/epic_{epic:03d}_final_model.zip"
    config_path = f"games/{game}/epic_{epic:03d}_from_scratch/config/config.yaml"

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("💡 Train the model first using the train command")
        return False

    if not Path(config_path).exists():
        print(f"❌ Config not found: {config_path}")
        print("💡 Train the model first to generate the config")
        return False

    # Calculate episodes needed for desired duration (rough estimate)
    episodes = max(1, int(hours * 10))  # Rough estimate: 6min per episode average

    # Use the evaluation script to generate video
    cmd = [
        sys.executable,
        "training/eval.py",
        "--checkpoint", model_path,
        "--config", config_path,
        "--episodes", str(episodes),
        "--seconds", str(int(hours * 3600)),
        "--deterministic"
    ]

    print(f"🎬 Running evaluation command:")
    print(f"   {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, cwd=Path.cwd())
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Video generation failed: {e}")
        return False

def run_batch_training(games: List[str], hours: float, epic: int = 1, test: bool = False) -> Dict[str, bool]:
    """Run training for multiple games."""
    
    results = {}
    
    for game in games:
        print(f"\n{'='*50}")
        print(f"🎯 Starting batch training: {game}")
        print(f"{'='*50}")
        
        try:
            success = run_training_command(game, hours, epic, test, verbose=True)
            results[game] = success
            
            if success:
                print(f"✅ {game} training completed successfully")
            else:
                print(f"❌ {game} training failed")
                
        except Exception as e:
            print(f"❌ {game} training error: {e}")
            results[game] = False
    
    return results

def run_system_training(system: str, epic: int = 1, hours: float = 10, test: bool = False) -> Dict[str, bool]:
    """Train all games in a specific system."""
    
    if system not in SYSTEM_CONFIGS:
        available_systems = list(SYSTEM_CONFIGS.keys())
        raise ValueError(f"System '{system}' not supported. Available: {available_systems}")
    
    games = SYSTEM_CONFIGS[system]["supported_games"]
    
    print(f"🎮 Training all {system.upper()} games (Epic {epic}, {hours}h each)")
    print(f"Games: {', '.join(games)}")
    
    return run_batch_training(games, hours, epic, test)

def main():
    """Main command dispatcher."""
    parser = argparse.ArgumentParser(
        description="Custom training commands for multi-game ML system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a specific game")
    train_parser.add_argument("--game", required=True, choices=list(GAME_SYSTEMS.keys()),
                             help="Game to train")
    train_parser.add_argument("--hours", type=float, default=10, help="Training duration in hours")
    train_parser.add_argument("--epic", type=int, default=1, choices=[1, 2, 3], help="Epic number")
    train_parser.add_argument("--system", choices=list(SYSTEM_CONFIGS.keys()), 
                             help="Gaming system (auto-detected if not specified)")
    train_parser.add_argument("--test", action="store_true", help="Run test mode")
    
    # Video command
    video_parser = subparsers.add_parser("video", help="Generate video of specific length")
    video_parser.add_argument("--game", required=True, choices=list(GAME_SYSTEMS.keys()),
                             help="Game for video generation")
    video_parser.add_argument("--length", required=True, help="Video length (e.g., '30min', '2h', '90m')")
    video_parser.add_argument("--epic", type=int, default=1, choices=[1, 2, 3], help="Epic number")
    video_parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium",
                             help="Video quality")
    
    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Train multiple games")
    batch_parser.add_argument("--games", required=True, help="Comma-separated list of games")
    batch_parser.add_argument("--hours", type=float, default=10, help="Training duration per game")
    batch_parser.add_argument("--epic", type=int, default=1, choices=[1, 2, 3], help="Epic number")
    batch_parser.add_argument("--test", action="store_true", help="Run test mode")
    
    # System training command
    system_parser = subparsers.add_parser("system-train", help="Train all games in a system")
    system_parser.add_argument("--system", required=True, choices=list(SYSTEM_CONFIGS.keys()),
                              help="Gaming system to train")
    system_parser.add_argument("--hours", type=float, default=10, help="Training duration per game")
    system_parser.add_argument("--epic", type=int, default=1, choices=[1, 2, 3], help="Epic number")
    system_parser.add_argument("--test", action="store_true", help="Run test mode")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "train":
            success = run_training_command(
                args.game, args.hours, args.epic, args.test, args.system
            )
            sys.exit(0 if success else 1)
            
        elif args.command == "video":
            success = run_video_generation(
                args.game, args.length, args.epic, args.quality
            )
            sys.exit(0 if success else 1)
            
        elif args.command == "batch":
            games = [g.strip() for g in args.games.split(",")]
            results = run_batch_training(games, args.hours, args.epic, args.test)
            
            print(f"\n{'='*50}")
            print("📊 BATCH TRAINING RESULTS")
            print(f"{'='*50}")
            
            for game, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"{game:20} {status}")
            
            failed_count = sum(1 for success in results.values() if not success)
            sys.exit(failed_count)
            
        elif args.command == "system-train":
            results = run_system_training(args.system, args.epic, args.hours, args.test)
            
            print(f"\n{'='*50}")
            print(f"📊 {args.system.upper()} SYSTEM TRAINING RESULTS")
            print(f"{'='*50}")
            
            for game, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"{game:20} {status}")
            
            failed_count = sum(1 for success in results.values() if not success)
            sys.exit(failed_count)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
