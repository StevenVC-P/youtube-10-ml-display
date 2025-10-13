#!/usr/bin/env python3
"""
Video Segment Concatenator for Sprint 6

Concatenates video segments using FFmpeg with proper handling of different formats and codecs.
Supports both file list concatenation and filter complex concatenation methods.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class VideoSegmentConcatenator:
    """Concatenates video segments using FFmpeg."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.paths = config.get('paths', {})
        self.ffmpeg_path = self.paths.get('ffmpeg_path', 'ffmpeg')
        
        print(f"🔗 Video Segment Concatenator initialized")
        print(f"   FFmpeg path: {self.ffmpeg_path}")
    
    def test_ffmpeg_availability(self) -> bool:
        """Test if FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_video_info(self, video_path: Path) -> Optional[Dict]:
        """Get video information using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            format_info = data.get('format', {})
            
            return {
                'duration': float(format_info.get('duration', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'pixel_format': video_stream.get('pix_fmt', 'unknown')
            }
            
        except Exception as e:
            print(f"⚠️ Error getting video info for {video_path}: {e}")
            return None
    
    def find_segments(self, input_dir: str, pattern: str = "*.mp4") -> List[Path]:
        """Find video segments in directory."""
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"❌ Input directory not found: {input_dir}")
            return []
        
        # Find all video files matching pattern
        segments = list(input_path.glob(pattern))
        
        # Sort segments (assuming numbered segments)
        def sort_key(path):
            # Extract number from filename
            import re
            match = re.search(r'(\d+)', path.stem)
            return int(match.group(1)) if match else 0
        
        segments.sort(key=sort_key)
        
        print(f"🔍 Found {len(segments)} segments in {input_dir}")
        for i, segment in enumerate(segments[:5]):  # Show first 5
            print(f"   {i+1}: {segment.name}")
        if len(segments) > 5:
            print(f"   ... and {len(segments) - 5} more")
        
        return segments
    
    def validate_segments(self, segments: List[Path]) -> bool:
        """Validate that segments are compatible for concatenation."""
        if not segments:
            print("❌ No segments to validate")
            return False
        
        print("🔍 Validating segment compatibility...")
        
        # Get info for first segment as reference
        first_info = self.get_video_info(segments[0])
        if not first_info:
            print(f"❌ Cannot get info for first segment: {segments[0]}")
            return False
        
        reference = {
            'width': first_info['width'],
            'height': first_info['height'],
            'fps': first_info['fps'],
            'codec': first_info['codec']
        }
        
        print(f"   Reference: {reference['width']}x{reference['height']} @ {reference['fps']:.1f}fps, {reference['codec']}")
        
        # Check compatibility with other segments
        incompatible = []
        for i, segment in enumerate(segments[1:], 1):
            info = self.get_video_info(segment)
            if not info:
                print(f"⚠️ Cannot get info for segment {i}: {segment}")
                continue
            
            if (info['width'] != reference['width'] or 
                info['height'] != reference['height'] or
                abs(info['fps'] - reference['fps']) > 0.1):
                incompatible.append((i, segment, info))
        
        if incompatible:
            print(f"⚠️ Found {len(incompatible)} incompatible segments:")
            for i, segment, info in incompatible[:3]:  # Show first 3
                print(f"   Segment {i}: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
            return False
        
        print(f"✅ All {len(segments)} segments are compatible")
        return True
    
    def concat_with_file_list(self, segments: List[Path], output_path: str) -> bool:
        """Concatenate segments using FFmpeg file list method."""
        print("🔗 Concatenating using file list method...")
        
        # Create temporary file list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            file_list_path = f.name
            for segment in segments:
                # Use absolute paths and escape special characters
                abs_path = segment.resolve()
                f.write(f"file '{abs_path}'\n")
        
        try:
            # FFmpeg concat command
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-c', 'copy',  # Copy streams without re-encoding
                '-y',  # Overwrite output
                output_path
            ]
            
            print(f"   Command: {' '.join(cmd[:6])} ... {output_path}")
            
            # Run FFmpeg
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                output_size = Path(output_path).stat().st_size / 1024 / 1024
                print(f"✅ Concatenation successful!")
                print(f"   Duration: {elapsed:.1f} seconds")
                print(f"   Output size: {output_size:.1f} MB")
                return True
            else:
                print(f"❌ FFmpeg failed with return code {result.returncode}")
                print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timeout (1 hour)")
            return False
        except Exception as e:
            print(f"❌ Concatenation error: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(file_list_path)
            except:
                pass
    
    def concat_with_filter_complex(self, segments: List[Path], output_path: str) -> bool:
        """Concatenate segments using FFmpeg filter complex method (re-encodes)."""
        print("🔗 Concatenating using filter complex method...")
        
        if len(segments) > 50:
            print("⚠️ Too many segments for filter complex method, using file list instead")
            return self.concat_with_file_list(segments, output_path)
        
        # Build FFmpeg command with filter complex
        cmd = [self.ffmpeg_path]
        
        # Add input files
        for segment in segments:
            cmd.extend(['-i', str(segment)])
        
        # Build filter complex
        filter_parts = []
        for i in range(len(segments)):
            filter_parts.append(f"[{i}:v][{i}:a]")
        
        filter_complex = f"{''.join(filter_parts)}concat=n={len(segments)}:v=1:a=1[outv][outa]"
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'fast',
            '-crf', '23',
            '-y',
            output_path
        ])
        
        print(f"   Command: {' '.join(cmd[:6])} ... (filter complex) ... {output_path}")
        
        try:
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout for re-encoding
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                output_size = Path(output_path).stat().st_size / 1024 / 1024
                print(f"✅ Concatenation successful!")
                print(f"   Duration: {elapsed:.1f} seconds")
                print(f"   Output size: {output_size:.1f} MB")
                return True
            else:
                print(f"❌ FFmpeg failed with return code {result.returncode}")
                print(f"   Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ FFmpeg timeout (2 hours)")
            return False
        except Exception as e:
            print(f"❌ Concatenation error: {e}")
            return False
    
    def concatenate_segments(self, input_dir: str, output_path: str, 
                           pattern: str = "*.mp4", method: str = "auto") -> bool:
        """Main concatenation method."""
        print(f"🎬 Starting segment concatenation")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_path}")
        print(f"   Pattern: {pattern}")
        print(f"   Method: {method}")
        
        # Check FFmpeg availability
        if not self.test_ffmpeg_availability():
            print("❌ FFmpeg not available")
            return False
        
        # Find segments
        segments = self.find_segments(input_dir, pattern)
        if not segments:
            print("❌ No segments found")
            return False
        
        # Validate segments
        if not self.validate_segments(segments):
            if method == "auto":
                print("⚠️ Segments incompatible, switching to filter complex method")
                method = "filter_complex"
            elif method == "file_list":
                print("❌ Segments incompatible for file list method")
                return False
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Choose concatenation method
        if method == "auto" or method == "file_list":
            return self.concat_with_file_list(segments, output_path)
        elif method == "filter_complex":
            return self.concat_with_filter_complex(segments, output_path)
        else:
            print(f"❌ Unknown method: {method}")
            return False


def load_config():
    """Load configuration from config.yaml."""
    import yaml
    with open("conf/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def concat_segments_from_config(input_dir: str = None, output_path: str = None, 
                               config_path: str = "conf/config.yaml") -> bool:
    """Concatenate segments using configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Use defaults from config if not specified
    if input_dir is None:
        input_dir = config.get('paths', {}).get('videos_parts', 'video/render/parts')
    
    if output_path is None:
        output_path = "video/render/youtube_10h.mp4"
    
    concatenator = VideoSegmentConcatenator(config)
    return concatenator.concatenate_segments(input_dir, output_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Concatenate video segments for YouTube 10 ML Display")
    parser.add_argument("--input", required=True, help="Input directory containing segments")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--pattern", default="*.mp4", help="File pattern to match")
    parser.add_argument("--method", choices=["auto", "file_list", "filter_complex"], 
                       default="auto", help="Concatenation method")
    parser.add_argument("--config", default="conf/config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    print("🔗 Video Segment Concatenator - Sprint 6")
    print("=" * 50)
    
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        concatenator = VideoSegmentConcatenator(config)
        success = concatenator.concatenate_segments(
            args.input, 
            args.output, 
            args.pattern, 
            args.method
        )
        
        if success:
            print("\n✅ Segment concatenation complete!")
            return 0
        else:
            print("\n❌ Segment concatenation failed!")
            return 1
        
    except Exception as e:
        print(f"❌ Concatenation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
