#!/usr/bin/env python3
"""
Test Sprint 5 streaming with real FFmpeg output.
Creates actual video files to verify the streaming functionality.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from stream.ffmpeg_io import FFmpegWriter, test_ffmpeg_availability
from stream.stream_eval import GridComposer


def test_real_ffmpeg_simple():
    """Test creating a simple video with real FFmpeg."""
    print("🎬 Testing Real FFmpeg - Simple Video")
    
    # Set FFmpeg path explicitly
    ffmpeg_path = r"C:\Users\steve\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    
    # Test availability
    available = test_ffmpeg_availability(ffmpeg_path)
    print(f"   FFmpeg available: {available}")
    
    if not available:
        print("   ❌ FFmpeg not available, skipping real test")
        return False
    
    # Create output directory
    output_dir = Path("videos_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create writer
    output_path = output_dir / "test_simple.mp4"
    writer = FFmpegWriter(
        output_path=str(output_path),
        width=320,
        height=240,
        fps=10,  # Lower FPS for faster testing
        mock_mode=False,  # Real FFmpeg!
        ffmpeg_path=ffmpeg_path,
        verbose=True
    )
    
    print(f"   Output: {output_path}")
    
    # Start writer
    success = writer.start()
    if not success:
        print("   ❌ Failed to start FFmpeg writer")
        return False
    
    print("   ✅ FFmpeg writer started")
    
    # Generate colorful test frames
    print("   Generating 30 frames...")
    for i in range(30):
        # Create a colorful frame that changes over time
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Create moving colored rectangles
        color_r = int(255 * (i / 30))
        color_g = int(255 * ((30 - i) / 30))
        color_b = 128
        
        # Draw rectangles
        frame[50:150, 50:150] = [color_r, 0, 0]      # Red square
        frame[50:150, 170:270] = [0, color_g, 0]    # Green square
        frame[90:190, 110:210] = [0, 0, color_b]    # Blue square
        
        # Add frame number text (simple)
        frame[10:30, 10:100] = [255, 255, 255]  # White rectangle for text
        
        success = writer.write_frame(frame)
        if not success:
            print(f"   ❌ Failed to write frame {i}")
            writer.stop()
            return False
    
    print(f"   ✅ Wrote {writer.frames_written} frames")
    
    # Stop writer
    writer.stop()
    
    # Check if file was created
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"   ✅ Video created: {file_size} bytes")
        return True
    else:
        print("   ❌ Video file not created")
        return False


def test_real_ffmpeg_grid():
    """Test creating a grid layout video with real FFmpeg."""
    print("\n🎨 Testing Real FFmpeg - Grid Layout")
    
    # Set FFmpeg path explicitly
    ffmpeg_path = r"C:\Users\steve\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    
    # Create output directory
    output_dir = Path("videos_test")
    output_dir.mkdir(exist_ok=True)
    
    # Create grid composer
    composer = GridComposer(
        grid_size=4,  # 2x2 grid
        pane_size=(240, 180),
        overlay_hud=True
    )
    
    # Create writer for grid output
    output_path = output_dir / "test_grid.mp4"
    writer = FFmpegWriter(
        output_path=str(output_path),
        width=480,  # 2 * 240
        height=440,  # 2 * 180 + 80 for HUD
        fps=10,
        mock_mode=False,
        ffmpeg_path=ffmpeg_path,
        verbose=True
    )
    
    print(f"   Output: {output_path}")
    
    # Start writer
    success = writer.start()
    if not success:
        print("   ❌ Failed to start FFmpeg writer")
        return False
    
    print("   ✅ FFmpeg writer started")
    
    # Generate grid frames
    print("   Generating 50 frames with 2x2 grid...")
    for frame_idx in range(50):
        # Create 4 different colored frames
        frames = []
        colors = [
            [255, 100, 100],  # Red
            [100, 255, 100],  # Green
            [100, 100, 255],  # Blue
            [255, 255, 100],  # Yellow
        ]
        
        for i in range(4):
            frame = np.full((84, 84, 3), colors[i], dtype=np.uint8)
            
            # Add some animation
            brightness = int(128 + 127 * np.sin(frame_idx * 0.2 + i))
            frame = (frame * brightness / 255).astype(np.uint8)
            
            frames.append(frame)
        
        # Create HUD data
        hud_data = {
            'global_step': frame_idx * 1000,
            'checkpoint_time': '2025-01-06 15:30:00',
            'fps': 10.0,
            'rewards': [
                np.sin(frame_idx * 0.1) * 10,
                np.cos(frame_idx * 0.1) * 15,
                np.sin(frame_idx * 0.15) * 8,
                np.cos(frame_idx * 0.12) * 12,
            ]
        }
        
        # Compose grid with HUD
        composed_frame = composer.compose_frame(frames, hud_info=hud_data)
        
        # Write frame
        success = writer.write_frame(composed_frame)
        if not success:
            print(f"   ❌ Failed to write frame {frame_idx}")
            writer.stop()
            return False
    
    print(f"   ✅ Wrote {writer.frames_written} frames")
    
    # Stop writer
    writer.stop()
    
    # Check if file was created
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"   ✅ Grid video created: {file_size} bytes")
        return True
    else:
        print("   ❌ Grid video file not created")
        return False


def main():
    """Run real FFmpeg tests."""
    print("🚀 Sprint 5 Real FFmpeg Testing")
    print("=" * 40)
    
    tests = [
        ("Simple Video", test_real_ffmpeg_simple),
        ("Grid Layout Video", test_real_ffmpeg_grid),
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
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 40)
    print(f"🏁 Real FFmpeg Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All real FFmpeg tests passed!")
        print("🎬 Sprint 5 can create actual video files!")
        
        # List created files
        output_dir = Path("videos_test")
        if output_dir.exists():
            print("\n📁 Created video files:")
            for video_file in output_dir.glob("*.mp4"):
                size = video_file.stat().st_size
                print(f"   • {video_file.name}: {size:,} bytes")
        
        return True
    else:
        print("🔧 Some real FFmpeg tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
