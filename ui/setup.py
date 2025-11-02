#!/usr/bin/env python3
"""
Setup script for ML Container Management UI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_requirements():
    """Check if required tools are installed."""
    print("Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        sys.exit(1)
    
    # Check if Node.js is installed
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        node_version = result.stdout.strip()
        print(f"Node.js version: {node_version}")
    except FileNotFoundError:
        print("Error: Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/")
        sys.exit(1)
    
    # Check if npm is installed
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        npm_version = result.stdout.strip()
        print(f"npm version: {npm_version}")
    except FileNotFoundError:
        print("Error: npm is not installed")
        sys.exit(1)
    
    print("✓ All requirements satisfied")

def setup_backend():
    """Set up the backend environment."""
    print("\n" + "="*50)
    print("Setting up backend...")
    print("="*50)
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Create virtual environment if it doesn't exist
    venv_dir = backend_dir / "venv"
    if not venv_dir.exists():
        print("Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", "venv"], cwd=backend_dir)
    
    # Determine the correct python executable
    if os.name == 'nt':  # Windows
        python_exe = venv_dir / "Scripts" / "python.exe"
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        python_exe = venv_dir / "bin" / "python"
        pip_exe = venv_dir / "bin" / "pip"
    
    # Install dependencies
    print("Installing Python dependencies...")
    run_command([str(pip_exe), "install", "--upgrade", "pip"], cwd=backend_dir)
    run_command([str(pip_exe), "install", "-r", "requirements.txt"], cwd=backend_dir)
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"
    if not env_file.exists() and env_example.exists():
        print("Creating .env file...")
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from .env.example")
        print("  Please review and modify the .env file as needed")
    
    print("✓ Backend setup complete")

def setup_frontend():
    """Set up the frontend environment."""
    print("\n" + "="*50)
    print("Setting up frontend...")
    print("="*50)
    
    frontend_dir = Path(__file__).parent / "frontend"
    
    if not frontend_dir.exists():
        print("Frontend directory not found, skipping frontend setup")
        return
    
    # Install dependencies
    print("Installing Node.js dependencies...")
    run_command(["npm", "install"], cwd=frontend_dir)
    
    print("✓ Frontend setup complete")

def create_directories():
    """Create necessary directories."""
    print("\n" + "="*50)
    print("Creating directories...")
    print("="*50)
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "models",
        "video",
        "logs",
        "ui/backend/containers",
        "ui/backend/logs"
    ]
    
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {directory}")

def test_setup():
    """Test the setup."""
    print("\n" + "="*50)
    print("Testing setup...")
    print("="*50)
    
    backend_dir = Path(__file__).parent / "backend"
    
    # Determine the correct python executable
    if os.name == 'nt':  # Windows
        python_exe = backend_dir / "venv" / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        python_exe = backend_dir / "venv" / "bin" / "python"
    
    # Test backend
    print("Testing backend...")
    result = run_command([str(python_exe), "test_api.py"], cwd=backend_dir, check=False)
    if result.returncode == 0:
        print("✓ Backend test passed")
    else:
        print("⚠ Backend test failed, but setup is complete")
    
    print("✓ Setup testing complete")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("SETUP COMPLETE! 🎉")
    print("="*60)
    print()
    print("Next steps:")
    print()
    print("1. Start the backend server:")
    if os.name == 'nt':  # Windows
        print("   cd ui/backend")
        print("   venv\\Scripts\\activate")
        print("   python start.py")
    else:  # Unix/Linux/macOS
        print("   cd ui/backend")
        print("   source venv/bin/activate")
        print("   python start.py")
    print()
    print("2. In a new terminal, start the frontend (if available):")
    print("   cd ui/frontend")
    print("   npm run dev")
    print()
    print("3. Open your browser and go to:")
    print("   Backend API: http://localhost:8000")
    print("   Frontend UI: http://localhost:3000 (if available)")
    print("   API Docs: http://localhost:8000/docs")
    print()
    print("4. Create your first ML training container!")
    print()
    print("For more information, see:")
    print("   - ui/README.md")
    print("   - docs/UI_ARCHITECTURE_PLAN.md")
    print("   - docs/CONTAINER_MANAGEMENT_SPEC.md")
    print()

def main():
    """Main setup function."""
    print("ML Container Management UI Setup")
    print("="*50)
    
    try:
        check_requirements()
        create_directories()
        setup_backend()
        setup_frontend()
        test_setup()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
