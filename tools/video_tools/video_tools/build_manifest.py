#!/usr/bin/env python3
"""
Video Manifest Builder for Sprint 6

Scans video directories and builds a CSV manifest of all video clips with metadata.
Supports milestone videos, evaluation clips, and streaming segments.
"""

import os
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import subprocess
import json
from fractions import Fraction

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class VideoManifestBuilder:
    """Builds manifests of video files with metadata."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.paths = config.get('paths', {})
        
        # Video directories to scan
        self.video_dirs = {
            'milestones': self.paths.get('videos_milestones', 'video/milestones'),
            'eval': self.paths.get('videos_eval', 'video/eval'),
            'segments': self.paths.get('videos_parts', 'video/render/parts'),
            'demo': 'video/demo',
            'realistic_gameplay': 'video/realistic_gameplay'
        }
        
        print(f"[Manifest] Video Manifest Builder initialized")
        print(f"   Scanning directories: {list(self.video_dirs.keys())}")
    
    def get_video_metadata(self, video_path: Path) -> Optional[Dict]:
        """Extract metadata from video file using ffprobe."""
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
                print(f"Warning ffprobe failed for {video_path}: {result.stderr}")
                return None
            
            data = json.loads(result.stdout)
            format_info = data.get('format', {})
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                print(f"Warning No video stream found in {video_path}")
                return None
            
            fps_raw = video_stream.get('r_frame_rate', '0/1')
            try:
                fps_value = float(Fraction(fps_raw))
            except (ValueError, ZeroDivisionError):
                fps_value = 0.0

            return {
                'duration': float(format_info.get('duration', 0)),
                'size_bytes': int(format_info.get('size', 0)),
                'bitrate': int(format_info.get('bit_rate', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': fps_value,
                'codec': video_stream.get('codec_name', 'unknown')
            }
            
        except Exception as e:
            print(f"ERROR Error getting metadata for {video_path}: {e}")
            return None
    
    def parse_filename_metadata(self, filename: str, category: str) -> Dict:
        """Extract metadata from filename patterns."""
        metadata = {
            'category': category,
            'timestamp': None,
            'step': None,
            'episode': None,
            'percentage': None,
            'segment_index': None
        }
        
        # Parse different filename patterns
        if category == 'milestones':
            # Pattern: step_100000_pct_5.mp4
            if 'step_' in filename and '_pct_' in filename:
                parts = filename.replace('.mp4', '').split('_')
                try:
                    step_idx = parts.index('step') + 1
                    pct_idx = parts.index('pct') + 1
                    metadata['step'] = int(parts[step_idx])
                    metadata['percentage'] = int(parts[pct_idx])
                except (ValueError, IndexError):
                    pass
        
        elif category == 'eval':
            # Pattern: eval_step_100000_ep_1.mp4
            if 'eval_step_' in filename and '_ep_' in filename:
                parts = filename.replace('.mp4', '').split('_')
                try:
                    step_idx = parts.index('step') + 1
                    ep_idx = parts.index('ep') + 1
                    metadata['step'] = int(parts[step_idx])
                    metadata['episode'] = int(parts[ep_idx])
                except (ValueError, IndexError):
                    pass
        
        elif category == 'segments':
            # Pattern: youtube_continuous_001.mp4 or segment_001.mp4
            if 'segment_' in filename or 'continuous_' in filename:
                # Extract segment number
                import re
                match = re.search(r'(\d+)\.mp4$', filename)
                if match:
                    metadata['segment_index'] = int(match.group(1))
        
        elif category in ['demo', 'realistic_gameplay']:
            # Pattern: realistic_breakout_60min_20251006_191607.mp4
            if '_20' in filename:  # Contains timestamp
                import re
                match = re.search(r'(\d{8}_\d{6})', filename)
                if match:
                    timestamp_str = match.group(1)
                    try:
                        metadata['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except ValueError:
                        pass
        
        return metadata
    
    def scan_directory(self, dir_path: str, category: str) -> List[Dict]:
        """Scan a directory for video files and collect metadata."""
        videos = []
        path = Path(dir_path)
        
        if not path.exists():
            print(f"Warning Directory not found: {dir_path}")
            return videos
        
        print(f"[Scan] Scanning {category}: {dir_path}")
        
        # Supported video extensions
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                print(f"   Found: {file_path.name}")
                
                # Get file stats
                stat = file_path.stat()
                
                # Parse filename metadata
                filename_meta = self.parse_filename_metadata(file_path.name, category)
                
                # Get video metadata
                video_meta = self.get_video_metadata(file_path)
                
                # Handle relative path safely
                try:
                    relative_path = str(file_path.relative_to(Path.cwd()))
                except ValueError:
                    # If file is not in current working directory, use absolute path
                    relative_path = str(file_path)

                video_info = {
                    'filepath': str(file_path),
                    'filename': file_path.name,
                    'category': category,
                    'file_size_bytes': stat.st_size,
                    'file_mtime': datetime.fromtimestamp(stat.st_mtime),
                    'relative_path': relative_path,
                }
                
                # Add filename metadata
                video_info.update(filename_meta)
                
                # Add video metadata if available
                if video_meta:
                    video_info.update(video_meta)
                else:
                    # Fallback values
                    video_info.update({
                        'duration': 0,
                        'size_bytes': stat.st_size,
                        'bitrate': 0,
                        'width': 0,
                        'height': 0,
                        'fps': 0,
                        'codec': 'unknown'
                    })
                
                videos.append(video_info)
        
        print(f"   Found {len(videos)} video files")
        return videos
    
    def build_manifest(self) -> List[Dict]:
        """Build complete manifest of all video files."""
        print("[Video] Building video manifest...")
        
        all_videos = []
        
        for category, dir_path in self.video_dirs.items():
            videos = self.scan_directory(dir_path, category)
            all_videos.extend(videos)
        
        # Sort by category, then by step/timestamp/segment_index
        def sort_key(video):
            category_order = {'milestones': 0, 'eval': 1, 'segments': 2, 'demo': 3, 'realistic_gameplay': 4}
            return (
                category_order.get(video['category'], 999),
                video.get('step', 0),
                video.get('segment_index', 0),
                video.get('timestamp', datetime.min),
                video['filename']
            )
        
        all_videos.sort(key=sort_key)
        
        print(f"[Stats] Manifest complete: {len(all_videos)} total videos")
        return all_videos
    
    def save_manifest_csv(self, videos: List[Dict], output_path: str = "manifest.csv"):
        """Save manifest to CSV file."""
        if not videos:
            print("Warning No videos to save in manifest")
            return
        
        print(f"[Save] Saving manifest to: {output_path}")
        
        # Define CSV columns
        columns = [
            'filepath', 'filename', 'category', 'relative_path',
            'duration', 'file_size_bytes', 'width', 'height', 'fps', 'codec',
            'step', 'percentage', 'episode', 'segment_index', 'timestamp', 'file_mtime'
        ]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for video in videos:
                # Prepare row data
                row = {}
                for col in columns:
                    value = video.get(col)
                    if isinstance(value, datetime):
                        value = value.isoformat()
                    row[col] = value
                
                writer.writerow(row)
        
        print(f"OK Manifest saved with {len(videos)} entries")
    
    def print_summary(self, videos: List[Dict]):
        """Print summary statistics of the manifest."""
        if not videos:
            print("[Stats] No videos found")
            return
        
        print("\n[Stats] Manifest Summary")
        print("=" * 50)
        
        # Count by category
        by_category = {}
        total_duration = 0
        total_size = 0
        
        for video in videos:
            category = video['category']
            by_category[category] = by_category.get(category, 0) + 1
            total_duration += video.get('duration', 0)
            total_size += video.get('file_size_bytes', 0)
        
        for category, count in by_category.items():
            print(f"   {category}: {count} videos")
        
        print(f"\n[Totals] Totals:")
        print(f"   Videos: {len(videos)}")
        print(f"   Duration: {total_duration/3600:.1f} hours")
        print(f"   Size: {total_size/1024/1024:.1f} MB")


def load_config(config_path: str = "conf/config.yaml") -> Dict:
    """Load configuration from config.yaml."""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_manifest_from_config(config_path: str = "conf/config.yaml", output_path: str = "manifest.csv"):
    """Build manifest using configuration file."""
    config = load_config(config_path)
    builder = VideoManifestBuilder(config)
    videos = builder.build_manifest()
    builder.save_manifest_csv(videos, output_path)
    builder.print_summary(videos)
    return builder, videos


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Build video manifest for YouTube 10 ML Display")
    parser.add_argument("--config", default="conf/config.yaml", help="Configuration file path")
    parser.add_argument("--output", default="manifest.csv", help="Output CSV file path")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    
    args = parser.parse_args()
    
    print("Video Manifest Builder - Sprint 6")
    print("=" * 50)
    
    try:
        builder, videos = build_manifest_from_config(args.config, args.output)
        
        if args.summary:
            builder.print_summary(videos)
        
        print("\nManifest building complete!")
        return 0
        
    except Exception as e:
        print(f"Manifest building failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
