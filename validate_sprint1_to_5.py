#!/usr/bin/env python3
"""Validate Sprint 1-5 completion"""

import sys
import os
from pathlib import Path

def test_config_loading():
    """Test configuration loading"""
    print("=== Testing Config Loading ===")
    try:
        from conf.config import load_config
        config = load_config('conf/config.yaml')
        print(f"✅ Config loaded successfully")
        print(f"✅ Environment ID: {config['game']['env_id']}")
        print(f"✅ Frame stack: {config['game']['frame_stack']}")
        return True, config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False, None

def test_environment_creation(config):
    """Test environment creation"""
    print("\n=== Testing Environment Creation ===")
    try:
        from envs.make_env import make_eval_env
        print("✅ make_eval_env imported")
        
        # Create environment
        env = make_eval_env(config, seed=42)
        print("✅ Environment created")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Environment reset successful")
        print(f"✅ Observation shape: {obs.shape}")
        print(f"✅ Observation dtype: {obs.dtype}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✅ Environment step successful, reward: {reward}")
        
        env.close()
        print("✅ Environment closed")
        return True
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test file structure exists"""
    print("\n=== Testing File Structure ===")
    
    required_files = [
        "conf/config.yaml",
        "envs/make_env.py",
        "envs/atari_wrappers.py"
    ]
    
    optional_files = [
        "training/train.py",
        "training/eval.py", 
        "training/callbacks.py",
        "agents/algo_factory.py",
        "stream/ffmpeg_io.py",
        "stream/stream_eval.py"
    ]
    
    all_good = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (REQUIRED)")
            all_good = False
    
    print("\nOptional Sprint 2-5 files:")
    for file_path in optional_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} (not implemented yet)")
    
    return all_good

def test_imports():
    """Test key imports work"""
    print("\n=== Testing Key Imports ===")
    
    imports_to_test = [
        ("conf.config", "load_config"),
        ("envs.make_env", "make_eval_env"),
        ("envs.atari_wrappers", "apply_atari_wrappers")
    ]
    
    all_good = True
    
    for module, function in imports_to_test:
        try:
            exec(f"from {module} import {function}")
            print(f"✅ {module}.{function}")
        except Exception as e:
            print(f"❌ {module}.{function}: {e}")
            all_good = False
    
    return all_good

def main():
    """Main validation"""
    print("🚀 Sprint 1-5 Validation")
    print("=" * 50)
    
    results = []
    
    # Test 1: Config loading
    config_ok, config = test_config_loading()
    results.append(("Config Loading", config_ok))
    
    # Test 2: File structure
    structure_ok = test_file_structure()
    results.append(("File Structure", structure_ok))
    
    # Test 3: Imports
    imports_ok = test_imports()
    results.append(("Key Imports", imports_ok))
    
    # Test 4: Environment creation (only if config loaded)
    if config_ok and config:
        env_ok = test_environment_creation(config)
        results.append(("Environment Creation", env_ok))
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 VALIDATION SUMMARY")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    core_passed = all(success for test_name, success in results if test_name in ["Config Loading", "Environment Creation"])
    
    if core_passed:
        print("\n🎉 SUCCESS: Core Sprint 1 functionality working!")
        print("✅ Environment factory operational")
        print("✅ Ready to test/implement remaining sprints")
    else:
        print("\n❌ FAILED: Core functionality not working")
        print("Please fix the errors above before proceeding")
    
    return core_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)