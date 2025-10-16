#!/usr/bin/env python3
"""
Organize Epic Training Videos into Clear, Well-Labeled Structure

Creates a comprehensive video organization system:
- Individual hour videos by epic
- Merged epic videos
- Highlight reels
- Master collection with clear naming
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def organize_epic_videos():
    """Organize all epic videos into a clear structure"""
    
    print("📹 EPIC VIDEO ORGANIZER")
    print("=" * 40)
    
    # Create master video organization structure
    base_video_dir = Path("video")
    epic_video_dir = base_video_dir / "epic_series_breakout"
    epic_video_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        "individual_hours": epic_video_dir / "individual_hours",
        "merged_epics": epic_video_dir / "merged_epics", 
        "highlights": epic_video_dir / "highlights",
        "analytics_showcase": epic_video_dir / "analytics_showcase"
    }
    
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)
    
    # Epic definitions for proper naming
    epic_names = {
        1: "from_scratch",
        2: "advanced_mastery", 
        3: "perfect_play",
        4: "speed_demon",
        5: "precision_master",
        6: "strategic_genius",
        7: "endurance_champion",
        8: "adaptive_expert",
        9: "ultimate_mastery",
        10: "legendary_status"
    }
    
    epic_titles = {
        1: "From Scratch",
        2: "Advanced Mastery",
        3: "Perfect Play", 
        4: "Speed Demon",
        5: "Precision Master",
        6: "Strategic Genius",
        7: "Endurance Champion",
        8: "Adaptive Expert",
        9: "Ultimate Mastery",
        10: "Legendary Status"
    }
    
    organized_count = 0
    
    # Process each epic
    for epic_num in range(1, 11):
        epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
        epic_title = epic_titles.get(epic_num, f"Epic {epic_num}")
        
        # Find epic directory
        epic_dirs = list(Path("games/breakout").glob(f"epic_{epic_num:03d}_*"))
        if not epic_dirs:
            print(f"⚠️ Epic {epic_num} directory not found")
            continue
            
        epic_dir = epic_dirs[0]
        videos_dir = epic_dir / "videos"
        
        if not videos_dir.exists():
            print(f"⚠️ Epic {epic_num} videos directory not found")
            continue
        
        print(f"📁 Processing Epic {epic_num}: {epic_title}")
        
        # Organize individual hour videos
        individual_hours_dir = videos_dir / "individual_hours"
        if individual_hours_dir.exists():
            epic_hour_dir = subdirs["individual_hours"] / f"epic_{epic_num:02d}_{epic_name}"
            epic_hour_dir.mkdir(exist_ok=True)
            
            hour_count = 0
            for hour_video in individual_hours_dir.glob("*.mp4"):
                # Create descriptive filename
                new_name = f"Epic_{epic_num:02d}_{epic_title.replace(' ', '_')}_{hour_video.name}"
                dest_path = epic_hour_dir / new_name
                
                if not dest_path.exists():
                    shutil.copy2(hour_video, dest_path)
                    hour_count += 1
            
            print(f"  ✅ Organized {hour_count} hour videos")
            organized_count += hour_count
        
        # Organize merged epic videos
        merged_dir = videos_dir / "merged_epic"
        if merged_dir.exists():
            for merged_video in merged_dir.glob("*.mp4"):
                new_name = f"Epic_{epic_num:02d}_{epic_title.replace(' ', '_')}_COMPLETE_{merged_video.name}"
                dest_path = subdirs["merged_epics"] / new_name
                
                if not dest_path.exists():
                    shutil.copy2(merged_video, dest_path)
                    print(f"  ✅ Organized merged epic video")
                    organized_count += 1
        
        # Organize highlights
        highlights_dir = videos_dir / "highlights"
        if highlights_dir.exists():
            epic_highlights_dir = subdirs["highlights"] / f"epic_{epic_num:02d}_{epic_name}"
            epic_highlights_dir.mkdir(exist_ok=True)
            
            highlight_count = 0
            for highlight_video in highlights_dir.glob("*.mp4"):
                new_name = f"Epic_{epic_num:02d}_{epic_title.replace(' ', '_')}_HIGHLIGHT_{highlight_video.name}"
                dest_path = epic_highlights_dir / new_name
                
                if not dest_path.exists():
                    shutil.copy2(highlight_video, dest_path)
                    highlight_count += 1
            
            if highlight_count > 0:
                print(f"  ✅ Organized {highlight_count} highlight videos")
                organized_count += highlight_count
    
    # Create master index file
    create_video_index(epic_video_dir, epic_names, epic_titles)
    
    print("\n" + "=" * 40)
    print(f"🎉 VIDEO ORGANIZATION COMPLETE!")
    print(f"✅ Organized {organized_count} videos")
    print(f"📁 Master location: {epic_video_dir.absolute()}")
    print("=" * 40)
    
    return epic_video_dir

def create_video_index(epic_video_dir, epic_names, epic_titles):
    """Create a comprehensive index of all organized videos"""
    
    index_data = {
        "created": datetime.now().isoformat(),
        "total_epics": 10,
        "series_title": "Breakout Epic Training Series",
        "description": "100-hour GPU-accelerated ML training journey with analytics",
        "epics": {}
    }
    
    # Scan organized videos
    for epic_num in range(1, 11):
        epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
        epic_title = epic_titles.get(epic_num, f"Epic {epic_num}")
        
        epic_info = {
            "number": epic_num,
            "name": epic_name,
            "title": epic_title,
            "individual_hours": [],
            "merged_epic": [],
            "highlights": []
        }
        
        # Count individual hours
        hour_dir = epic_video_dir / "individual_hours" / f"epic_{epic_num:02d}_{epic_name}"
        if hour_dir.exists():
            epic_info["individual_hours"] = [f.name for f in hour_dir.glob("*.mp4")]
        
        # Count merged epics
        for merged_file in (epic_video_dir / "merged_epics").glob(f"Epic_{epic_num:02d}_*.mp4"):
            epic_info["merged_epic"].append(merged_file.name)
        
        # Count highlights
        highlight_dir = epic_video_dir / "highlights" / f"epic_{epic_num:02d}_{epic_name}"
        if highlight_dir.exists():
            epic_info["highlights"] = [f.name for f in highlight_dir.glob("*.mp4")]
        
        index_data["epics"][epic_num] = epic_info
    
    # Save index
    index_file = epic_video_dir / "video_index.json"
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    # Create README
    readme_content = f"""# Breakout Epic Training Series Videos

## Overview
Complete 100-hour GPU-accelerated ML training journey across 10 epic sessions.
Each epic represents 10 hours of neural network learning with real-time analytics.

## Video Organization

### 📁 individual_hours/
Hour-by-hour training videos showing gradual learning progression.
- Each epic has 10 individual hour videos
- Shows real-time ML analytics with legend
- ~90MB per hour video

### 📁 merged_epics/
Complete epic videos (10 hours condensed).
- Full epic training sessions
- Comprehensive learning journey
- Larger file sizes

### 📁 highlights/
Best moments and breakthrough learning instances.
- Key learning milestones
- Dramatic improvement moments
- Curated content

### 📁 analytics_showcase/
Demonstration videos of the ML analytics system.
- Neural network visualizations
- Training metrics displays
- Educational content

## Epic Series

"""
    
    for epic_num in range(1, 11):
        epic_title = epic_titles.get(epic_num, f"Epic {epic_num}")
        epic_name = epic_names.get(epic_num, f"epic_{epic_num}")
        
        readme_content += f"**Epic {epic_num}: {epic_title}**\n"
        readme_content += f"- Directory: `epic_{epic_num:02d}_{epic_name}/`\n"
        readme_content += f"- Training: 10 hours GPU-accelerated\n"
        readme_content += f"- Videos: Individual hours + merged epic\n\n"
    
    readme_content += f"""
## Technical Details
- **GPU**: NVIDIA RTX 3070 Ti
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Environment**: Atari Breakout
- **Analytics**: Real-time ML metrics with legend
- **Total Training**: 100 hours
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_file = epic_video_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    print(f"📄 Created video index: {index_file}")
    print(f"📄 Created README: {readme_file}")

if __name__ == "__main__":
    organize_epic_videos()
