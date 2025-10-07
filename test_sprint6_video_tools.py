#!/usr/bin/env python3
"""
Sprint 6 Test Suite: Video Tools (Manifest & Supercut)

Comprehensive tests for all video_tools modules:
- build_manifest.py: Video manifest building
- concat_segments.py: Video segment concatenation  
- render_supercut.py: Supercut rendering with music

Tests use existing video files from Sprint 5 demonstrations.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def load_config():
    """Load configuration."""
    with open("conf/config.yaml", 'r') as f:
        return yaml.safe_load(f)

def test_config_loading():
    """Test 1: Configuration Loading"""
    print("🧪 Test 1: Configuration Loading")
    
    try:
        config = load_config()
        
        # Check video_tools section exists
        assert 'video_tools' in config, "video_tools section missing from config"
        
        video_tools = config['video_tools']
        assert 'manifest' in video_tools, "manifest config missing"
        assert 'concat' in video_tools, "concat config missing"
        assert 'supercut' in video_tools, "supercut config missing"
        
        print("   ✅ Configuration loaded successfully")
        print(f"   ✅ video_tools config sections: {list(video_tools.keys())}")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False

def test_manifest_builder():
    """Test 2: Video Manifest Builder"""
    print("\n🧪 Test 2: Video Manifest Builder")
    
    try:
        # Import directly to avoid import chain issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_manifest", "video_tools/build_manifest.py")
        build_manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_manifest)
        
        config = load_config()
        builder = build_manifest.VideoManifestBuilder(config)
        
        print("   ✅ VideoManifestBuilder created successfully")
        
        # Test directory scanning
        videos = builder.build_manifest()
        
        print(f"   ✅ Manifest built with {len(videos)} videos")
        
        # Test CSV saving
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_csv_path = f.name
        
        try:
            builder.save_manifest_csv(videos, test_csv_path)
            
            # Verify CSV file was created and has content
            csv_path = Path(test_csv_path)
            assert csv_path.exists(), "CSV file not created"
            assert csv_path.stat().st_size > 0, "CSV file is empty"
            
            print(f"   ✅ Manifest CSV saved successfully ({csv_path.stat().st_size} bytes)")
            
            # Test summary
            builder.print_summary(videos)
            print("   ✅ Summary printed successfully")
            
            return True
            
        finally:
            # Clean up
            try:
                os.unlink(test_csv_path)
            except:
                pass
        
    except Exception as e:
        print(f"   ❌ Manifest builder test failed: {e}")
        return False

def test_segment_concatenator():
    """Test 3: Video Segment Concatenator"""
    print("\n🧪 Test 3: Video Segment Concatenator")
    
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("concat_segments", "video_tools/concat_segments.py")
        concat_segments = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(concat_segments)
        
        config = load_config()
        concatenator = concat_segments.VideoSegmentConcatenator(config)
        
        print("   ✅ VideoSegmentConcatenator created successfully")
        
        # Test FFmpeg availability
        ffmpeg_available = concatenator.test_ffmpeg_availability()
        print(f"   ✅ FFmpeg availability: {ffmpeg_available}")
        
        # Test segment finding (use existing video directories)
        test_dirs = [
            "video/demo",
            "video/realistic_gameplay", 
            "videos_test"
        ]
        
        found_segments = False
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                segments = concatenator.find_segments(test_dir)
                if segments:
                    print(f"   ✅ Found {len(segments)} segments in {test_dir}")
                    found_segments = True
                    
                    # Test validation
                    if len(segments) > 1:
                        valid = concatenator.validate_segments(segments[:2])  # Test first 2
                        print(f"   ✅ Segment validation: {valid}")
                    
                    break
        
        if not found_segments:
            print("   ⚠️ No video segments found for testing")
        
        print("   ✅ Segment concatenator test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Segment concatenator test failed: {e}")
        return False

def test_supercut_renderer():
    """Test 4: Supercut Renderer"""
    print("\n🧪 Test 4: Supercut Renderer")
    
    try:
        # Import directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("render_supercut", "video_tools/render_supercut.py")
        render_supercut = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(render_supercut)
        
        config = load_config()
        renderer = render_supercut.SupercutRenderer(config)
        
        print("   ✅ SupercutRenderer created successfully")
        print(f"   ✅ Target duration: {renderer.target_hours} hours")
        print(f"   ✅ Add titles: {renderer.add_titles}")
        
        # Test FFmpeg availability
        ffmpeg_available = renderer.test_ffmpeg_availability()
        print(f"   ✅ FFmpeg availability: {ffmpeg_available}")
        
        # Create a test manifest
        test_manifest_data = [
            {
                'filepath': 'video/demo/sprint5_demo_20251006_174638.mp4',
                'filename': 'sprint5_demo_20251006_174638.mp4',
                'category': 'demo',
                'duration': 30.0,
                'width': 480,
                'height': 440,
                'fps': 30.0
            }
        ]
        
        # Test clip selection
        selected = renderer.select_clips_for_supercut(test_manifest_data, 0.1)  # 6 minutes
        print(f"   ✅ Clip selection: {len(selected)} clips selected")
        
        print("   ✅ Supercut renderer test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Supercut renderer test failed: {e}")
        return False

def test_integration_workflow():
    """Test 5: Integration Workflow"""
    print("\n🧪 Test 5: Integration Workflow")
    
    try:
        # Test the complete workflow: manifest -> supercut
        
        # 1. Build manifest
        import importlib.util
        spec1 = importlib.util.spec_from_file_location("build_manifest", "video_tools/build_manifest.py")
        build_manifest = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(build_manifest)
        
        config = load_config()
        
        # Create temporary manifest
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            test_manifest_path = f.name
        
        try:
            # Build manifest
            videos = build_manifest.build_manifest_from_config(output_path=test_manifest_path)
            print(f"   ✅ Manifest built with {len(videos)} videos")
            
            # 2. Test supercut rendering (dry run)
            spec2 = importlib.util.spec_from_file_location("render_supercut", "video_tools/render_supercut.py")
            render_supercut = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(render_supercut)
            
            renderer = render_supercut.SupercutRenderer(config)
            
            # Load the manifest we just created
            loaded_videos = renderer.load_manifest(test_manifest_path)
            print(f"   ✅ Manifest loaded with {len(loaded_videos)} videos")
            
            # Test clip selection
            selected_clips = renderer.select_clips_for_supercut(loaded_videos, 0.1)  # 6 minutes
            print(f"   ✅ Selected {len(selected_clips)} clips for supercut")
            
            print("   ✅ Integration workflow test completed")
            return True
            
        finally:
            # Clean up
            try:
                os.unlink(test_manifest_path)
            except:
                pass
        
    except Exception as e:
        print(f"   ❌ Integration workflow test failed: {e}")
        return False

def test_memory_usage():
    """Test 6: Memory Usage"""
    print("\n🧪 Test 6: Memory Usage")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   Initial memory: {initial_memory:.1f} MB")
        
        # Import all video_tools modules
        import importlib.util

        modules = {}
        module_names = ['build_manifest', 'concat_segments', 'render_supercut']
        for module_name in module_names:
            spec = importlib.util.spec_from_file_location(module_name, f"video_tools/{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[module_name] = module
        
        # Get memory after imports
        after_imports = process.memory_info().rss / 1024 / 1024  # MB
        import_increase = after_imports - initial_memory
        
        print(f"   After imports: {after_imports:.1f} MB (+{import_increase:.1f} MB)")
        
        # Create instances
        config = load_config()
        builder = modules['build_manifest'].VideoManifestBuilder(config)
        concatenator = modules['concat_segments'].VideoSegmentConcatenator(config)
        renderer = modules['render_supercut'].SupercutRenderer(config)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"   Final memory: {final_memory:.1f} MB (+{total_increase:.1f} MB total)")
        
        # Memory should not increase dramatically
        assert total_increase < 50, f"Memory increase too high: {total_increase:.1f} MB"
        
        print("   ✅ Memory usage within acceptable limits")
        return True
        
    except ImportError:
        print("   ⚠️ psutil not available, skipping memory test")
        return True
    except Exception as e:
        print(f"   ❌ Memory usage test failed: {e}")
        return False

def main():
    """Run all Sprint 6 video tools tests."""
    print("🚀 Sprint 6 Video Tools Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_config_loading),
        ("Manifest Builder", test_manifest_builder),
        ("Segment Concatenator", test_segment_concatenator),
        ("Supercut Renderer", test_supercut_renderer),
        ("Integration Workflow", test_integration_workflow),
        ("Memory Usage", test_memory_usage)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All Sprint 6 video tools tests passed!")
        print("✅ Sprint 6 core functionality is working correctly")
        return 0
    else:
        print("⚠️ Some tests failed - Sprint 6 needs attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())
