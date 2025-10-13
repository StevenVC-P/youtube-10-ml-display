#!/usr/bin/env python3
"""
Epic Training Launcher
======================

Clean interface for launching epic training sessions.

Usage:
    python epic_launcher.py --game breakout --epic 2
    python epic_launcher.py --game breakout --epic 2-5 --batch
    python epic_launcher.py --list-games
"""

import sys
import subprocess
import argparse
from pathlib import Path

def list_available_games():
    """List available games for epic training."""
    games_dir = Path("games")
    if games_dir.exists():
        games = [d.name for d in games_dir.iterdir() if d.is_dir()]
        print("Available games:")
        for game in sorted(games):
            print(f"  - {game}")
    else:
        print("No games directory found")

def launch_single_epic(game, epic_num):
    """Launch a single epic training session."""
    script_path = Path("epic_training/scripts/train_epic_continuous.py")

    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--game", game,
        "--epic", str(epic_num),
        "--hours", "10"
    ]

    print(f"Launching Epic {epic_num} for {game.title()}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error launching epic: {e}")
        return False

def launch_batch_epics(game, start_epic, end_epic):
    """Launch multiple epic training sessions."""
    script_path = Path("epic_training/scripts/run_multiple_epics.py")

    if not script_path.exists():
        print(f"Error: {script_path} not found")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        "--game", game,
        "--start-epic", str(start_epic),
        "--end-epic", str(end_epic)
    ]

    print(f"Launching Epics {start_epic}-{end_epic} for {game.title()}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error launching batch epics: {e}")
        return False

def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(description="Epic Training Launcher")
    parser.add_argument("--game", help="Game to train (e.g., breakout)")
    parser.add_argument("--epic", help="Epic number or range (e.g., 2 or 2-5)")
    parser.add_argument("--batch", action="store_true", help="Run multiple epics")
    parser.add_argument("--list-games", action="store_true", help="List available games")

    args = parser.parse_args()

    if args.list_games:
        list_available_games()
        return

    if not args.game or not args.epic:
        print("Usage: python epic_launcher.py --game breakout --epic 2")
        print("       python epic_launcher.py --list-games")
        return

    # Parse epic range
    if "-" in args.epic:
        start_epic, end_epic = map(int, args.epic.split("-"))
        launch_batch_epics(args.game, start_epic, end_epic)
    else:
        epic_num = int(args.epic)
        launch_single_epic(args.game, epic_num)

if __name__ == "__main__":
    main()