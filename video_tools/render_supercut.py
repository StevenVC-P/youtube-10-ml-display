#!/usr/bin/env python3
"""
Supercut Renderer for Sprint 6

Creates a single long MP4 from manifest with optional music overlay and target duration.
Supports intelligent clip selection, transitions, and audio mixing.
"""

import os
import sys
import csv
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import math

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class SupercutRenderer:
    """Renders supercut videos from manifests with optional music."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.paths = config.get('paths', {})
        self.ffmpeg_path = self.paths.get('ffmpeg_path', 'ffmpeg')
        
        # Supercut settings
        self.render_config = config.get('render', {})
        self.target_hours = self.render_config.get('target_hours', 10)
        self.add_titles = self.render_config.get('add_titles', True)
        
        print(f"🎬 Supercut Renderer initialized")
        print(f"   Target duration: {self.target_hours} hours")
        print(f"   Add titles: {self.add_titles}")
    
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
    
    def load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load video manifest from CSV file."""
        videos = []
        
        if not Path(manifest_path).exists():
            print(f"❌ Manifest file not found: {manifest_path}")
            return videos
        
        print(f"📋 Loading manifest: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Convert numeric fields
                for field in ['duration', 'file_size_bytes', 'width', 'height', 'fps', 'step', 'percentage', 'episode', 'segment_index']:
                    if row.get(field) and row[field] != '':
                        try:
                            if field == 'duration' or field == 'fps':
                                row[field] = float(row[field])
                            else:
                                row[field] = int(row[field])
                        except ValueError:
                            row[field] = 0
                
                # Convert timestamp
                if row.get('timestamp'):
                    try:
                        row['timestamp'] = datetime.fromisoformat(row['timestamp'])
                    except ValueError:
                        row['timestamp'] = None
                
                videos.append(row)
        
        print(f"   Loaded {len(videos)} video entries")
        return videos
    
    def select_clips_for_supercut(self, videos: List[Dict], target_duration_hours: float) -> List[Dict]:
        """Intelligently select clips to create a supercut of target duration."""
        target_seconds = target_duration_hours * 3600
        
        print(f"🎯 Selecting clips for {target_duration_hours:.1f} hour supercut")
        
        # Categorize videos
        milestones = [v for v in videos if v['category'] == 'milestones']
        eval_clips = [v for v in videos if v['category'] == 'eval']
        segments = [v for v in videos if v['category'] == 'segments']
        demos = [v for v in videos if v['category'] in ['demo', 'realistic_gameplay']]
        
        print(f"   Available: {len(milestones)} milestones, {len(eval_clips)} eval, {len(segments)} segments, {len(demos)} demos")
        
        selected_clips = []
        total_duration = 0
        
        # Strategy: Prioritize variety and learning progression
        
        # 1. Add key milestones (start, middle, end)
        if milestones:
            milestones.sort(key=lambda x: x.get('percentage', 0))
            key_percentages = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            
            for pct in key_percentages:
                # Find milestone closest to this percentage
                def distance_func(x):
                    x_pct = x.get('percentage', 0)
                    if x_pct is None or x_pct == '':
                        x_pct = 0
                    try:
                        return abs(int(x_pct) - pct)
                    except (ValueError, TypeError):
                        return float('inf')

                closest = min(milestones, key=distance_func, default=None)
                if closest and closest not in selected_clips:
                    selected_clips.append(closest)
                    total_duration += closest.get('duration', 0)
                    print(f"   Added milestone: {pct}% ({closest.get('duration', 0):.1f}s)")
        
        # 2. Add evaluation clips showing improvement
        if eval_clips:
            eval_clips.sort(key=lambda x: x.get('step', 0))
            # Take every Nth eval clip to show progression
            step_size = max(1, len(eval_clips) // 20)  # Up to 20 eval clips
            for i in range(0, len(eval_clips), step_size):
                clip = eval_clips[i]
                selected_clips.append(clip)
                total_duration += clip.get('duration', 0)
                print(f"   Added eval: step {clip.get('step', 0)} ({clip.get('duration', 0):.1f}s)")
        
        # 3. Add demo/realistic gameplay clips
        for demo in demos:
            if total_duration < target_seconds * 0.8:  # Don't exceed 80% with demos
                selected_clips.append(demo)
                total_duration += demo.get('duration', 0)
                print(f"   Added demo: {demo['filename']} ({demo.get('duration', 0):.1f}s)")
        
        # 4. Fill remaining time with segments if available
        if segments and total_duration < target_seconds:
            segments.sort(key=lambda x: x.get('segment_index', 0))
            remaining_time = target_seconds - total_duration
            
            for segment in segments:
                if total_duration >= target_seconds:
                    break
                selected_clips.append(segment)
                total_duration += segment.get('duration', 0)
                print(f"   Added segment: {segment.get('segment_index', 0)} ({segment.get('duration', 0):.1f}s)")
        
        print(f"📊 Selected {len(selected_clips)} clips, total duration: {total_duration/3600:.2f} hours")
        return selected_clips
    
    def create_clip_list_file(self, clips: List[Dict]) -> str:
        """Create temporary file list for FFmpeg concatenation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            file_list_path = f.name
            
            for clip in clips:
                filepath = clip['filepath']
                if not Path(filepath).exists():
                    print(f"⚠️ Clip not found: {filepath}")
                    continue
                
                # Use absolute path
                abs_path = Path(filepath).resolve()
                f.write(f"file '{abs_path}'\n")
        
        return file_list_path
    
    def render_supercut_basic(self, clips: List[Dict], output_path: str, music_path: Optional[str] = None) -> bool:
        """Render basic supercut by concatenating clips."""
        print("🎬 Rendering basic supercut...")
        
        if not clips:
            print("❌ No clips to render")
            return False
        
        # Create clip list file
        file_list_path = self.create_clip_list_file(clips)
        
        try:
            # Build FFmpeg command
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path
            ]
            
            # Add music if provided
            if music_path and Path(music_path).exists():
                print(f"🎵 Adding background music: {music_path}")
                cmd.extend(['-i', music_path])
                
                # Mix audio: video audio + background music
                cmd.extend([
                    '-filter_complex', 
                    '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]',
                    '-map', '0:v',
                    '-map', '[aout]'
                ])
            else:
                cmd.extend(['-c', 'copy'])
            
            cmd.extend([
                '-y',  # Overwrite output
                output_path
            ])
            
            print(f"   Command: {' '.join(cmd[:8])} ... {output_path}")
            
            # Run FFmpeg
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                output_size = Path(output_path).stat().st_size / 1024 / 1024
                print(f"✅ Supercut rendering successful!")
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
            print(f"❌ Rendering error: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(file_list_path)
            except:
                pass
    
    def render_supercut_with_titles(self, clips: List[Dict], output_path: str, music_path: Optional[str] = None) -> bool:
        """Render supercut with title cards between sections."""
        print("🎬 Rendering supercut with titles...")
        
        # For now, fall back to basic rendering
        # TODO: Implement title card generation
        print("⚠️ Title card generation not yet implemented, using basic rendering")
        return self.render_supercut_basic(clips, output_path, music_path)
    
    def render_supercut(self, manifest_path: str, output_path: str, 
                       target_hours: Optional[float] = None, music_path: Optional[str] = None) -> bool:
        """Main supercut rendering method."""
        print(f"🎬 Starting supercut rendering")
        print(f"   Manifest: {manifest_path}")
        print(f"   Output: {output_path}")
        
        # Use target hours from parameter or config
        if target_hours is None:
            target_hours = self.target_hours
        
        print(f"   Target duration: {target_hours} hours")
        
        # Check FFmpeg availability
        if not self.test_ffmpeg_availability():
            print("❌ FFmpeg not available")
            return False
        
        # Load manifest
        videos = self.load_manifest(manifest_path)
        if not videos:
            print("❌ No videos in manifest")
            return False
        
        # Select clips for supercut
        selected_clips = self.select_clips_for_supercut(videos, target_hours)
        if not selected_clips:
            print("❌ No clips selected")
            return False
        
        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Render supercut
        if self.add_titles:
            return self.render_supercut_with_titles(selected_clips, output_path, music_path)
        else:
            return self.render_supercut_basic(selected_clips, output_path, music_path)


def load_config():
    """Load configuration from config.yaml."""
    import yaml
    with open("conf/config.yaml", 'r') as f:
        return yaml.safe_load(f)


def render_supercut_from_config(manifest_path: str = "manifest.csv", 
                               output_path: str = "video/render/youtube_supercut.mp4",
                               config_path: str = "conf/config.yaml") -> bool:
    """Render supercut using configuration file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    renderer = SupercutRenderer(config)
    
    # Get music path from config
    music_path = config.get('render', {}).get('music_path')
    
    return renderer.render_supercut(manifest_path, output_path, music_path=music_path)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Render supercut video for YouTube 10 ML Display")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV file")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--target-hours", type=float, help="Target duration in hours")
    parser.add_argument("--music", help="Background music file path")
    parser.add_argument("--config", default="conf/config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    print("🎬 Supercut Renderer - Sprint 6")
    print("=" * 50)
    
    try:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        renderer = SupercutRenderer(config)
        success = renderer.render_supercut(
            args.manifest,
            args.output,
            target_hours=args.target_hours,
            music_path=args.music
        )
        
        if success:
            print("\n✅ Supercut rendering complete!")
            return 0
        else:
            print("\n❌ Supercut rendering failed!")
            return 1
        
    except Exception as e:
        print(f"❌ Rendering failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
