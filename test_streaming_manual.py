#!/usr/bin/env python3
"""
Manual test for Sprint 5 streaming components.
Demonstrates the streaming functionality working end-to-end.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from conf.config import load_config
from stream.ffmpeg_io import FFmpegWriter, test_ffmpeg_availability
from stream.stream_eval import GridComposer


def test_ffmpeg_writer():
    """Test FFmpeg writer in mock mode."""
    print("🎬 Testing FFmpeg Writer (Mock Mode)")
    
    writer = FFmpegWriter(
        output_path="test_output.mp4",
        width=480,
        height=360,
        fps=30,
        mock_mode=True,
        verbose=True
    )
    
    print(f"   FFmpeg available: {test_ffmpeg_availability()}")
    print(f"   Mock mode: {writer.mock_mode}")
    
    # Start writer
    success = writer.start()
    print(f"   Writer started: {success}")
    
    # Write some test frames
    for i in range(10):
        frame = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)
        success = writer.write_frame(frame)
        if not success:
            print(f"   ❌ Failed to write frame {i}")
            return False
    
    print(f"   Frames written: {writer.frames_written}")
    
    # Stop writer
    writer.stop()
    print(f"   Writer stopped: {not writer.is_running}")
    
    return True


def test_grid_composer():
    """Test grid composition with different layouts."""
    print("\n🎨 Testing Grid Composer")
    
    # Test different grid sizes
    grid_configs = [
        (1, (480, 360), "Single pane"),
        (4, (240, 180), "2x2 grid"),
        (9, (160, 120), "3x3 grid"),
    ]
    
    for grid_size, pane_size, description in grid_configs:
        print(f"   Testing {description}...")
        
        composer = GridComposer(
            grid_size=grid_size,
            pane_size=pane_size,
            overlay_hud=True
        )
        
        # Create test frames
        frames = []
        for i in range(grid_size):
            # Create colorful test frame
            frame = np.zeros((84, 84, 3), dtype=np.uint8)
            frame[:, :, i % 3] = 255  # Different color per frame
            frames.append(frame)
        
        # Test composition with HUD
        hud_data = {
            'global_step': 123456,
            'checkpoint_time': '2025-01-01 12:00:00',
            'fps': 29.5,
            'rewards': [float(i * 10) for i in range(grid_size)]
        }
        
        composed = composer.compose_frame(frames, hud_info=hud_data)
        
        expected_height = pane_size[1] * composer.grid_rows + 80  # +80 for HUD
        expected_width = pane_size[0] * composer.grid_cols
        
        print(f"     Output shape: {composed.shape}")
        print(f"     Expected: ({expected_height}, {expected_width}, 3)")
        
        if composed.shape != (expected_height, expected_width, 3):
            print(f"     ❌ Shape mismatch!")
            return False
        
        print(f"     ✅ {description} working correctly")
    
    return True


def test_integration():
    """Test integration of writer and composer."""
    print("\n🔗 Testing Integration (Writer + Composer)")
    
    # Create writer
    writer = FFmpegWriter(
        output_path="integration_test.mp4",
        width=480,
        height=440,  # 360 + 80 for HUD
        fps=30,
        mock_mode=True,
        verbose=False
    )
    
    # Create composer
    composer = GridComposer(
        grid_size=4,
        pane_size=(240, 180),
        overlay_hud=True
    )
    
    # Start writer
    writer.start()
    
    # Generate and write frames
    print("   Generating 30 frames...")
    start_time = time.time()
    
    for frame_idx in range(30):
        # Create test frames
        frames = []
        for i in range(4):
            frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Compose with HUD
        hud_data = {
            'global_step': frame_idx * 1000,
            'checkpoint_time': '2025-01-01 12:00:00',
            'fps': 30.0,
            'rewards': [np.random.uniform(-10, 20) for _ in range(4)]
        }
        
        composed_frame = composer.compose_frame(frames, hud_info=hud_data)
        
        # Write frame
        success = writer.write_frame(composed_frame)
        if not success:
            print(f"   ❌ Failed to write frame {frame_idx}")
            return False
    
    elapsed = time.time() - start_time
    fps = 30 / elapsed
    
    print(f"   Frames written: {writer.frames_written}")
    print(f"   Time elapsed: {elapsed:.2f}s")
    print(f"   Effective FPS: {fps:.1f}")
    
    # Stop writer
    writer.stop()
    
    if fps < 30:
        print(f"   ⚠️ Performance below target (30 FPS)")
    else:
        print(f"   ✅ Performance above target")
    
    return True


def test_config_loading():
    """Test configuration loading."""
    print("\n⚙️ Testing Configuration Loading")
    
    try:
        config = load_config("conf/config.yaml")
        
        # Check streaming config
        stream_config = config.get('stream', {})
        required_keys = ['enabled', 'grid', 'pane_size', 'fps', 'overlay_hud']
        
        for key in required_keys:
            if key not in stream_config:
                print(f"   ❌ Missing stream config key: {key}")
                return False
        
        print(f"   Grid size: {stream_config['grid']}")
        print(f"   Pane size: {stream_config['pane_size']}")
        print(f"   FPS: {stream_config['fps']}")
        print(f"   HUD overlay: {stream_config['overlay_hud']}")
        print("   ✅ Configuration loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False


def main():
    """Run manual tests."""
    print("🧪 Sprint 5 Manual Testing Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("FFmpeg Writer", test_ffmpeg_writer),
        ("Grid Composer", test_grid_composer),
        ("Integration Test", test_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"💥 {test_name} - ERROR: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Manual Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All manual tests passed!")
        print("🚀 Sprint 5 streaming components are working correctly!")
        return True
    else:
        print("🔧 Some manual tests failed. Please investigate.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
