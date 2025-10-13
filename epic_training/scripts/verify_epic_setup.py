#!/usr/bin/env python3
"""
🔍 EPIC JOURNEY SETUP VERIFICATION
==================================

This script verifies that everything is properly configured for the 
10-hour epic neural network learning journey.
"""

import yaml
from pathlib import Path
import sys
import os

def check_directories():
    """Check that all required directories exist and are clean."""
    print("📁 Checking directories...")
    
    dirs_to_check = [
        "video/milestones",
        "models/checkpoints", 
        "logs/tb",
        "training"
    ]
    
    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            file_count = len(list(path.glob("*")))
            print(f"   ✅ {dir_path} exists ({file_count} files)")
        else:
            print(f"   ❌ {dir_path} missing!")
            return False
    
    return True

def check_config():
    """Check the configuration file."""
    print("\n⚙️  Checking configuration...")
    
    config_path = Path("conf/config.yaml")
    if not config_path.exists():
        print("   ❌ conf/config.yaml not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        total_timesteps = config.get('total_timesteps', 0)
        env_id = config.get('env_id', '')
        
        print(f"   ✅ Config loaded successfully")
        print(f"   📊 Total timesteps: {total_timesteps:,}")
        print(f"   🎮 Environment: {env_id}")
        
        if total_timesteps >= 10_000_000:
            print(f"   ✅ Timesteps sufficient for 10-hour journey")
        else:
            print(f"   ⚠️  Timesteps may be too low for full journey")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading config: {e}")
        return False

def check_training_script():
    """Check the training script configuration."""
    print("\n🎯 Checking training script...")
    
    train_path = Path("training/train.py")
    if not train_path.exists():
        print("   ❌ training/train.py not found!")
        return False
    
    try:
        with open(train_path, 'r') as f:
            content = f.read()
        
        # Check for epic journey configuration
        if "milestones_pct=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]" in content:
            print("   ✅ Epic journey milestones configured")
        else:
            print("   ❌ Epic journey milestones not found!")
            return False
        
        if "clip_seconds=3600" in content:
            print("   ✅ 1-hour video duration configured")
        else:
            print("   ❌ 1-hour video duration not configured!")
            return False
        
        if "MLAnalyticsVideoCallback" in content:
            print("   ✅ Enhanced neural network visualization enabled")
        else:
            print("   ❌ Neural network visualization not found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading training script: {e}")
        return False

def check_dependencies():
    """Check that all required dependencies are available."""
    print("\n📦 Checking dependencies...")
    
    required_modules = [
        'stable_baselines3',
        'gymnasium', 
        'torch',
        'numpy',
        'cv2',
        'yaml',
        'ale_py'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - MISSING!")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def estimate_resources():
    """Estimate resource requirements."""
    print("\n💻 Resource estimates for epic journey:")
    print("   ⏰ Duration: ~36-48 hours (2-3 days)")
    print("   💾 Disk space: ~5-10 GB")
    print("   🧠 RAM usage: ~2-4 GB")
    print("   🔥 CPU usage: High (training intensive)")
    print("   📹 Video output: 10 files × 1 hour each")
    print("   📊 Expected milestones:")
    
    milestones = [
        "Hour 1 (10%): Random exploration",
        "Hour 2 (20%): Basic learning begins", 
        "Hour 3 (30%): Paddle tracking improves",
        "Hour 4 (40%): Ball prediction develops",
        "Hour 5 (50%): Consistent returns",
        "Hour 6 (60%): Strategic positioning",
        "Hour 7 (70%): Advanced techniques",
        "Hour 8 (80%): Expert-level play",
        "Hour 9 (90%): Near-perfect performance",
        "Hour 10 (100%): Complete mastery"
    ]
    
    for milestone in milestones:
        print(f"      🎯 {milestone}")

def main():
    """Main verification function."""
    print("🔍 EPIC JOURNEY SETUP VERIFICATION")
    print("=" * 50)
    
    checks = [
        check_directories(),
        check_config(),
        check_training_script(),
        check_dependencies()
    ]
    
    print("\n" + "=" * 50)
    
    if all(checks):
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Ready for epic 10-hour neural network learning journey!")
        print("\n🚀 To start the journey, run:")
        print("   python launch_epic_journey.py")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("🔧 Please fix the issues above before starting the journey.")
        return 1
    
    estimate_resources()
    
    print("\n" + "=" * 50)
    print("🧠 May the neural networks be with you! 🎮")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
