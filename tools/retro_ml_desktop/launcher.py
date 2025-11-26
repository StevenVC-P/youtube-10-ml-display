"""
Launcher for Retro ML Trainer

Handles first-run detection and setup wizard before launching main application.
"""

import sys
from pathlib import Path

# Add project root to path - handle both frozen and normal execution
if getattr(sys, 'frozen', False):
    # Running as frozen executable
    project_root = Path(sys.executable).parent
else:
    # Running as normal Python script
    project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.retro_ml_desktop.config_manager import ConfigManager
from tools.retro_ml_desktop.setup_wizard import run_setup_wizard


def main():
    """Main launcher function."""
    # Initialize configuration manager
    config = ConfigManager()

    print(f"Config file: {config.config_file}")
    print(f"First run completed: {config.get('app.first_run_completed', False)}")

    # Check if this is first run
    if config.is_first_run():
        print("First run detected. Launching setup wizard...")

        # Run setup wizard
        def on_setup_complete():
            """Callback when setup is complete."""
            print("Setup wizard complete! Launching application...")
            print(f"First run now marked as: {config.get('app.first_run_completed', False)}")
            launch_application(config)

        run_setup_wizard(config, on_complete=on_setup_complete)

    else:
        # Not first run - launch application directly
        print("Launching Retro ML Trainer...")
        launch_application(config)


def launch_application(config: ConfigManager):
    """
    Launch the main application.
    
    Args:
        config: ConfigManager instance with loaded configuration
    """
    # Import and run main application
    from tools.retro_ml_desktop.main_simple import RetroMLSimple
    
    # Pass config to application
    app = RetroMLSimple(config=config)
    app.run()


if __name__ == "__main__":
    main()

