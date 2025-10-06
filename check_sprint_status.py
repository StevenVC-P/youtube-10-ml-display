#!/usr/bin/env python3
"""Check which sprints are implemented"""

from pathlib import Path

def check_sprint_implementation():
    """Check what's been implemented"""
    print("🚀 Sprint Implementation Status")
    print("=" * 40)
    
    sprints = [
        ("Sprint 1", "Environment Factory", ["envs/make_env.py", "envs/atari_wrappers.py"]),
        ("Sprint 2", "Training Pipeline", ["training/train.py", "agents/algo_factory.py"]),
        ("Sprint 3", "Milestone Videos", ["training/callbacks.py"]),
        ("Sprint 4", "Evaluation Videos", ["training/eval.py"]),
        ("Sprint 5", "Continuous Streaming", ["stream/stream_eval.py", "stream/ffmpeg_io.py"]),
        ("Sprint 6", "Video Post-processing", ["video_tools/build_manifest.py", "video_tools/render_supercut.py"])
    ]
    
    for sprint_name, description, files in sprints:
        print(f"\n{sprint_name}: {description}")
        all_exist = True
        for file_path in files:
            if Path(file_path).exists():
                print(f"  ✅ {file_path}")
            else:
                print(f"  ❌ {file_path}")
                all_exist = False
        
        status = "✅ COMPLETE" if all_exist else "⚠️  INCOMPLETE"
        print(f"  Status: {status}")
    
    print(f"\n{'='*40}")
    print("🎯 TO GET 10-HOUR VIDEOS:")
    print("1. Sprint 5: Run continuous streaming")
    print("2. Sprint 6: Build manifest & render supercut")

if __name__ == "__main__":
    check_sprint_implementation()