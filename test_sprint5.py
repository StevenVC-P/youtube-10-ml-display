#!/usr/bin/env python3
"""
Sprint 5 Test Runner - Comprehensive testing for continuous streaming functionality.
Tests all Sprint 5 components before merging to main.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if "passed" in result.stdout:
                # Extract test summary
                lines = result.stdout.split('\n')
                for line in lines:
                    if "passed" in line and ("failed" in line or "skipped" in line or "warning" in line):
                        print(f"   📊 {line.strip()}")
                        break
        else:
            print(f"❌ {description} - FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False
    
    return True


def main():
    """Run comprehensive Sprint 5 tests."""
    print("🚀 Sprint 5 Testing Suite - Continuous Evaluation Streamer")
    print("Testing all components before merging to main branch")
    
    # Test commands in order of complexity
    test_commands = [
        # 1. Basic configuration and environment tests
        ("python -m pytest tests/test_config_schema.py -v", 
         "Configuration Schema Validation"),
        
        ("python -m pytest tests/test_env_factory.py -v", 
         "Environment Factory Tests"),
        
        # 2. Sprint 5 specific component tests
        ("python -m pytest tests/test_stream_components.py -v", 
         "Streaming Components Unit Tests"),
        
        # 3. Sprint 5 integration tests
        ("python -m pytest tests/test_stream_integration.py -v", 
         "Streaming Integration Tests"),
        
        # 4. Sprint 5 acceptance criteria tests
        ("python -m pytest tests/test_stream_integration.py::TestAcceptanceCriteria -v", 
         "Sprint 5 Acceptance Criteria"),
        
        # 5. Full test suite
        ("python -m pytest tests/ -v --tb=short", 
         "Complete Test Suite"),
    ]
    
    passed_tests = 0
    total_tests = len(test_commands)
    
    for cmd, description in test_commands:
        if run_command(cmd, description):
            passed_tests += 1
        else:
            print(f"\n💔 Test failed: {description}")
            print("Stopping test execution due to failure.")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 SPRINT 5 TEST SUMMARY")
    print(f"{'='*60}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Sprint 5 is ready for merge to main.")
        print("\n✅ Verified Components:")
        print("   • FFmpeg I/O wrapper with mock mode")
        print("   • Grid composition (1x1, 2x2, 3x3 layouts)")
        print("   • HUD overlay system")
        print("   • Continuous evaluation streamer")
        print("   • Checkpoint polling and model reloading")
        print("   • Segmented output mode")
        print("   • Performance at target FPS")
        print("   • Error handling and recovery")
        
        print("\n🎯 Sprint 5 Acceptance Criteria Met:")
        print("   ✓ Grid layouts working correctly")
        print("   ✓ Real-time streaming at target FPS")
        print("   ✓ HUD displays all required information")
        print("   ✓ Segmented output mode functional")
        print("   ✓ Mock mode enables testing without FFmpeg")
        
        print("\n🚀 Ready to merge feature/sprint-5-continuous-streamer → main")
        return True
    else:
        print(f"❌ {passed_tests}/{total_tests} test suites passed.")
        print("🔧 Please fix failing tests before merging to main.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
