#!/usr/bin/env python3
"""
10-Hour Training Day Video Generator

Creates realistic 10-hour training videos showing ML agent learning progression.
Each video acts as a "day" in the learning journey.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from create_realistic_gameplay_video import RealisticGameplayStreamer


class TrainingDayGenerator:
    """Generator for 10-hour training day videos"""
    
    def __init__(self, game_name: str = "Breakout", env_id: str = "ALE/Breakout-v5"):
        self.game_name = game_name
        self.env_id = env_id
        self.duration_hours = 10
        self.duration_minutes = 600  # 10 hours
        
    def generate_day_video(self, day_number: int, output_dir: str = "video/days") -> bool:
        """Generate a complete 10-hour training day video"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"day{day_number:02d}_{self.game_name.lower()}_{self.duration_hours}h_{timestamp}.mp4"
        output_file = output_path / filename
        
        print(f"🎬 Generating Training Day {day_number}")
        print(f"   Game: {self.game_name} ({self.env_id})")
        print(f"   Duration: {self.duration_hours} hours ({self.duration_minutes} minutes)")
        print(f"   Output: {output_file}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Create the realistic gameplay streamer
            streamer = RealisticGameplayStreamer(
                duration_minutes=self.duration_minutes,
                grid_size=4  # 4-pane grid for multiple simultaneous games
            )
            
            # Override output path
            streamer.output_path = str(output_file)
            
            print(f"🚀 Starting {self.duration_hours}-hour video generation...")
            print(f"   This will take approximately 1-2 hours to generate")
            print(f"   Progress will be shown every 30 minutes of video content")
            
            # Generate the video
            success = streamer.create_realistic_gameplay_video()
            
            if success:
                elapsed_time = time.time() - start_time
                file_size = output_file.stat().st_size / (1024 * 1024)  # MB
                
                print(f"\n✅ Training Day {day_number} Generation Complete!")
                print(f"   Output: {output_file}")
                print(f"   Duration: {self.duration_hours} hours")
                print(f"   File Size: {file_size:.1f} MB")
                print(f"   Generation Time: {elapsed_time/60:.1f} minutes")
                print(f"   Ready for YouTube upload!")
                
                return True
            else:
                print(f"\n❌ Training Day {day_number} Generation Failed!")
                return False
                
        except Exception as e:
            print(f"\n❌ Error generating Training Day {day_number}: {e}")
            return False


def create_multi_day_series(days: int = 3, start_day: int = 1) -> bool:
    """Create a series of training day videos"""
    
    # Game configurations for different days
    games = [
        ("Breakout", "ALE/Breakout-v5"),
        ("Pong", "ALE/Pong-v5"), 
        ("SpaceInvaders", "ALE/SpaceInvaders-v5"),
        ("MsPacman", "ALE/MsPacman-v5"),
        ("Asterix", "ALE/Asterix-v5"),
        ("Seaquest", "ALE/Seaquest-v5")
    ]
    
    print(f"🎬 Creating {days}-Day Training Series")
    print(f"   Starting from Day {start_day}")
    print("=" * 60)
    
    success_count = 0
    
    for i in range(days):
        day_number = start_day + i
        game_name, env_id = games[i % len(games)]  # Cycle through games
        
        print(f"\n📅 Day {day_number}: {game_name}")
        
        generator = TrainingDayGenerator(game_name, env_id)
        
        if generator.generate_day_video(day_number):
            success_count += 1
            print(f"✅ Day {day_number} completed successfully")
        else:
            print(f"❌ Day {day_number} failed")
            
        print("-" * 60)
    
    print(f"\n🏁 Series Generation Complete!")
    print(f"   Successful Days: {success_count}/{days}")
    print(f"   Total Content: {success_count * 10} hours")
    
    if success_count == days:
        print("✅ All training days generated successfully!")
        return True
    else:
        print(f"⚠️ {days - success_count} days failed to generate")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate 10-hour training day videos")
    parser.add_argument("--day", type=int, default=1, help="Day number to generate")
    parser.add_argument("--game", choices=["breakout", "pong", "spaceinvaders", "mspacman"], 
                       default="breakout", help="Game to train on")
    parser.add_argument("--series", type=int, help="Generate series of N days")
    parser.add_argument("--output-dir", default="video/days", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎯 10-Hour Training Day Video Generator")
    print("=" * 60)
    
    # Game mapping
    game_configs = {
        "breakout": ("Breakout", "ALE/Breakout-v5"),
        "pong": ("Pong", "ALE/Pong-v5"),
        "spaceinvaders": ("SpaceInvaders", "ALE/SpaceInvaders-v5"),
        "mspacman": ("MsPacman", "ALE/MsPacman-v5")
    }
    
    if args.series:
        # Generate series of days
        success = create_multi_day_series(args.series, args.day)
        sys.exit(0 if success else 1)
    else:
        # Generate single day
        game_name, env_id = game_configs[args.game]
        generator = TrainingDayGenerator(game_name, env_id)
        
        success = generator.generate_day_video(args.day, args.output_dir)
        
        if success:
            print(f"\n🎉 Day {args.day} video ready for upload!")
            sys.exit(0)
        else:
            print(f"\n💥 Day {args.day} generation failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
