#!/usr/bin/env python3
"""
Stop all currently running training processes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tools.retro_ml_desktop.process_manager import ProcessManager
from tools.retro_ml_desktop.ml_database import MetricsDatabase
from tools.retro_ml_desktop.config_manager import ConfigManager

def stop_all_runs():
    """Stop all active training runs."""
    print("Stopping all training runs...")

    # Initialize process manager with correct database path
    process_manager = ProcessManager(str(project_root))
    config_manager = ConfigManager(project_root)
    db_path = str(config_manager.get_database_path())
    ml_database = MetricsDatabase(db_path)
    process_manager.set_database(ml_database)

    # Get all processes
    processes = process_manager.get_processes()

    active_processes = [p for p in processes if p.status in ["running", "paused"]]

    if not active_processes:
        print("No active processes found.")
        return

    print(f"Found {len(active_processes)} active processes:")
    for process in active_processes:
        print(f"  - {process.name} (ID: {process.id}, Status: {process.status})")

    # Stop each process
    for process in active_processes:
        print(f"\nStopping {process.name}...")
        success = process_manager.stop_process(process.id)
        if success:
            print(f"   Stopped successfully")
        else:
            print(f"   Failed to stop (may have already finished)")

    print("\nAll processes stopped!")

if __name__ == "__main__":
    stop_all_runs()
