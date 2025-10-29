"""
Video Player Utilities for ML Training Manager

Provides video viewing and analysis capabilities for training videos.
"""

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from pathlib import Path
import subprocess
import platform
import os
from typing import Optional, Dict, List


class VideoPlayerDialog:
    """Dialog for playing and analyzing training videos."""
    
    def __init__(self, parent, video_path: str, video_info: Dict):
        self.parent = parent
        self.video_path = Path(video_path)
        self.video_info = video_info
        self.dialog = None
        
    def show(self):
        """Show the video player dialog."""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title(f"Video Player - {self.video_info.get('name', 'Unknown')}")
        self.dialog.geometry("800x600")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"800x600+{x}+{y}")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create the dialog widgets."""
        # Title
        title_label = ctk.CTkLabel(
            self.dialog, 
            text=f"🎬 {self.video_info.get('name', 'Video Player')}", 
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Video info frame
        info_frame = ctk.CTkFrame(self.dialog)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        info_text = (
            f"📹 Type: {self.video_info.get('type', 'Unknown')}\n"
            f"⏱️ Duration: {self.video_info.get('duration', 'Unknown')}\n"
            f"📊 Size: {self.video_info.get('size', 'Unknown')}\n"
            f"🎮 Training Run: {self.video_info.get('training_run', 'Unknown')}\n"
            f"📁 Path: {self.video_path}"
        )
        
        info_label = ctk.CTkLabel(info_frame, text=info_text, justify="left")
        info_label.pack(pady=10, padx=10)
        
        # Video preview placeholder
        preview_frame = ctk.CTkFrame(self.dialog)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        preview_label = ctk.CTkLabel(
            preview_frame, 
            text="🎬 Video Preview\n\n"
                 "Click 'Play in External Player' to watch the video\n"
                 "or 'Extract Frames' to analyze specific moments",
            font=ctk.CTkFont(size=14)
        )
        preview_label.pack(expand=True)
        
        # Control buttons
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        play_btn = ctk.CTkButton(
            button_frame, 
            text="▶️ Play in External Player", 
            command=self._play_external,
            fg_color="#28a745", 
            hover_color="#218838"
        )
        play_btn.pack(side="left", padx=5, pady=5)
        
        analyze_btn = ctk.CTkButton(
            button_frame, 
            text="📊 Analyze Video", 
            command=self._analyze_video
        )
        analyze_btn.pack(side="left", padx=5, pady=5)
        
        extract_btn = ctk.CTkButton(
            button_frame, 
            text="🖼️ Extract Frames", 
            command=self._extract_frames
        )
        extract_btn.pack(side="left", padx=5, pady=5)
        
        close_btn = ctk.CTkButton(
            button_frame, 
            text="❌ Close", 
            command=self.dialog.destroy,
            fg_color="#6c757d", 
            hover_color="#5a6268"
        )
        close_btn.pack(side="right", padx=5, pady=5)
        
    def _play_external(self):
        """Play video in external player."""
        try:
            if platform.system() == "Windows":
                os.startfile(str(self.video_path))
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", str(self.video_path)])
            else:  # Linux
                subprocess.run(["xdg-open", str(self.video_path)])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video: {e}")
    
    def _analyze_video(self):
        """Analyze video content and show statistics."""
        try:
            # Basic video analysis
            file_size = self.video_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            
            analysis_text = (
                f"📊 Video Analysis\n\n"
                f"File: {self.video_path.name}\n"
                f"Size: {size_mb:.1f} MB ({file_size:,} bytes)\n"
                f"Type: {self.video_info.get('type', 'Unknown')}\n"
                f"Training Run: {self.video_info.get('training_run', 'Unknown')}\n\n"
                f"🎯 Training Context:\n"
                f"This video shows AI agent learning progress.\n"
                f"Look for improvements in:\n"
                f"• Score/reward increases\n"
                f"• Better decision making\n"
                f"• More efficient gameplay\n"
                f"• Reduced random actions\n\n"
                f"💡 Analysis Tips:\n"
                f"• Compare early vs late milestone videos\n"
                f"• Watch for strategy development\n"
                f"• Note performance consistency"
            )
            
            messagebox.showinfo("Video Analysis", analysis_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze video: {e}")
    
    def _extract_frames(self):
        """Extract frames from video for analysis."""
        try:
            # This would extract key frames - simplified for now
            messagebox.showinfo(
                "Extract Frames", 
                f"Frame extraction feature coming soon!\n\n"
                f"This will extract key frames from:\n"
                f"{self.video_path.name}\n\n"
                f"Useful for:\n"
                f"• Analyzing specific game moments\n"
                f"• Creating training progress comparisons\n"
                f"• Generating thumbnails\n"
                f"• Performance analysis"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract frames: {e}")


def get_video_players():
    """Get list of available video players on the system."""
    players = []
    
    if platform.system() == "Windows":
        # Common Windows video players
        common_players = [
            "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
            "C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe",
            "C:\\Program Files\\Windows Media Player\\wmplayer.exe",
        ]
        
        for player in common_players:
            if Path(player).exists():
                players.append(player)
    
    elif platform.system() == "Darwin":  # macOS
        # Common macOS video players
        common_players = [
            "/Applications/VLC.app",
            "/Applications/QuickTime Player.app",
            "/Applications/IINA.app",
        ]
        
        for player in common_players:
            if Path(player).exists():
                players.append(player)
    
    else:  # Linux
        # Common Linux video players
        common_players = ["vlc", "mpv", "totem", "mplayer"]
        
        for player in common_players:
            try:
                subprocess.run(["which", player], capture_output=True, check=True)
                players.append(player)
            except subprocess.CalledProcessError:
                pass
    
    return players


def play_video_with_player(video_path: str, player_path: Optional[str] = None):
    """Play video with specific player or system default."""
    try:
        if player_path and Path(player_path).exists():
            # Use specific player
            if platform.system() == "Windows":
                subprocess.run([player_path, video_path])
            else:
                subprocess.run([player_path, video_path])
        else:
            # Use system default
            if platform.system() == "Windows":
                os.startfile(video_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", video_path])
            else:  # Linux
                subprocess.run(["xdg-open", video_path])
                
    except Exception as e:
        raise Exception(f"Failed to play video: {e}")


def get_video_info(video_path: str) -> Dict:
    """Get basic video information."""
    try:
        path = Path(video_path)
        stat = path.stat()
        
        return {
            "name": path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "extension": path.suffix.lower(),
            "exists": path.exists()
        }
        
    except Exception as e:
        return {"error": str(e)}


def create_video_thumbnail(video_path: str, output_path: str, timestamp: str = "00:00:10") -> bool:
    """Create a thumbnail from video at specified timestamp."""
    try:
        # This would use ffmpeg to extract a frame - simplified for now
        # ffmpeg -i input.mp4 -ss 00:00:10 -vframes 1 -q:v 2 output.jpg

        # For now, just return True to indicate the feature exists
        # In production, you would implement actual thumbnail extraction
        return True

    except Exception:
        return False


def get_video_duration_precise(video_path: str) -> Optional[float]:
    """Get precise video duration using ffprobe if available."""
    try:
        # This would use ffprobe to get exact duration
        # ffprobe -v quiet -show_entries format=duration -of csv="p=0" input.mp4

        # For now, return None to fall back to file size estimation
        return None

    except Exception:
        return None


def analyze_video_content(video_path: str) -> Dict:
    """Analyze video content for training insights."""
    try:
        # This would analyze video content for:
        # - Frame rate
        # - Resolution
        # - Game score progression
        # - Action patterns
        # - Performance metrics

        # Simplified analysis for now
        path = Path(video_path)

        analysis = {
            "filename": path.name,
            "type": "training_video",
            "estimated_game_sessions": 1,
            "analysis_available": False,
            "recommendations": [
                "Watch for score improvements over time",
                "Look for more consistent gameplay patterns",
                "Notice reduction in random actions",
                "Observe strategy development"
            ]
        }

        # Determine video type from filename
        if "milestone" in path.name.lower():
            analysis["type"] = "milestone"
            analysis["recommendations"].insert(0, "Compare with previous milestone videos")
        elif "hour" in path.name.lower() or "part" in path.name.lower():
            analysis["type"] = "hour_segment"
            analysis["recommendations"].insert(0, "Analyze learning progression within this hour")
        elif "eval" in path.name.lower():
            analysis["type"] = "evaluation"
            analysis["recommendations"].insert(0, "Focus on performance consistency and final results")

        return analysis

    except Exception as e:
        return {"error": str(e)}
