#!/usr/bin/env python3
"""
FFmpeg I/O wrapper for continuous video streaming.
Provides a thin wrapper around FFmpeg subprocess for real-time video encoding.
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np
import cv2


class FFmpegWriter:
    """
    FFmpeg subprocess wrapper for real-time video encoding.

    Supports both single continuous files and segmented output for safe recording.
    Also supports a mock mode for testing when FFmpeg is not available.
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        width: int,
        height: int,
        fps: int = 30,
        crf: int = 23,
        preset: str = "veryfast",
        codec: str = "libx264",
        pixel_format: str = "yuv420p",
        segment_time: Optional[int] = None,
        ffmpeg_path: Optional[str] = None,
        verbose: bool = True,
        mock_mode: bool = False
    ):
        """
        Initialize FFmpeg writer.

        Args:
            output_path: Output file path or directory for segments
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            crf: Constant Rate Factor (quality, 0-51, lower = better)
            preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            codec: Video codec (libx264, libx265, etc.)
            pixel_format: Pixel format (yuv420p for compatibility)
            segment_time: Segment duration in seconds (None for single file)
            ffmpeg_path: Path to FFmpeg executable
            verbose: Enable verbose logging
            mock_mode: Enable mock mode for testing without FFmpeg
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.codec = codec
        self.pixel_format = pixel_format
        self.segment_time = segment_time
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.verbose = verbose
        self.mock_mode = mock_mode

        self.process = None
        self.is_running = False
        self.frames_written = 0
        self.start_time = None
        self.error_output = []

        # Create output directory if needed
        if self.segment_time:
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _build_ffmpeg_command(self) -> list:
        """Build FFmpeg command line arguments."""
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output files
            "-f", "rawvideo",  # Input format
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",  # Input size
            "-pix_fmt", "bgr24",  # Input pixel format (OpenCV uses BGR)
            "-r", str(self.fps),  # Input framerate
            "-i", "-",  # Read from stdin
        ]
        
        # Video encoding options
        cmd.extend([
            "-vcodec", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", self.pixel_format,
            "-r", str(self.fps),  # Output framerate
        ])
        
        # Segmentation options
        if self.segment_time:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            segment_pattern = self.output_path / f"segment_{timestamp}_%03d.mp4"
            
            cmd.extend([
                "-f", "segment",
                "-segment_time", str(self.segment_time),
                "-segment_format", "mp4",
                "-reset_timestamps", "1",
                str(segment_pattern)
            ])
        else:
            cmd.append(str(self.output_path))
        
        return cmd
    
    def start(self) -> bool:
        """
        Start FFmpeg process or mock mode.

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            if self.verbose:
                print("Warning FFmpeg process already running")
            return False

        # Mock mode for testing
        if self.mock_mode:
            if self.verbose:
                print(f"[Test] Starting mock FFmpeg mode:")
                print(f"  - Resolution: {self.width}x{self.height}")
                print(f"  - FPS: {self.fps}")
                print(f"  - Output: {self.output_path} (mock)")
                if self.segment_time:
                    print(f"  - Segment time: {self.segment_time}s (mock)")

            self.is_running = True
            self.start_time = time.time()
            self.frames_written = 0

            if self.verbose:
                print("OK Mock FFmpeg mode started successfully")

            return True

        try:
            cmd = self._build_ffmpeg_command()

            if self.verbose:
                print(f"[Video] Starting FFmpeg process:")
                print(f"  - Command: {' '.join(cmd[:10])}...")
                print(f"  - Resolution: {self.width}x{self.height}")
                print(f"  - FPS: {self.fps}")
                print(f"  - Output: {self.output_path}")
                if self.segment_time:
                    print(f"  - Segment time: {self.segment_time}s")

            # Start FFmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered
            )

            # Start error monitoring thread
            self.error_thread = threading.Thread(
                target=self._monitor_errors,
                daemon=True
            )
            self.error_thread.start()

            self.is_running = True
            self.start_time = time.time()
            self.frames_written = 0

            if self.verbose:
                print("OK FFmpeg process started successfully")

            return True

        except Exception as e:
            if self.verbose:
                print(f"ERROR Failed to start FFmpeg: {e}")
            return False
    
    def _monitor_errors(self):
        """Monitor FFmpeg stderr for errors."""
        if not self.process or not self.process.stderr:
            return
        
        try:
            for line in iter(self.process.stderr.readline, b''):
                if line:
                    error_msg = line.decode('utf-8', errors='ignore').strip()
                    self.error_output.append(error_msg)
                    
                    # Log critical errors
                    if any(keyword in error_msg.lower() for keyword in ['error', 'failed', 'invalid']):
                        if self.verbose:
                            print(f"Warning FFmpeg error: {error_msg}")
        except Exception as e:
            if self.verbose:
                print(f"Warning Error monitoring thread failed: {e}")
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the video stream.

        Args:
            frame: Frame as numpy array (H, W, 3) in BGR format

        Returns:
            True if frame written successfully, False otherwise
        """
        if not self.is_running:
            return False

        # Mock mode - just count frames
        if self.mock_mode:
            self.frames_written += 1
            return True

        if not self.process or not self.process.stdin:
            return False

        try:
            # Ensure frame is correct size and format
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()

            self.frames_written += 1
            return True

        except BrokenPipeError:
            if self.verbose:
                print("Warning FFmpeg pipe broken - process may have terminated")
            self.is_running = False
            return False
        except Exception as e:
            if self.verbose:
                print(f"ERROR Failed to write frame: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current streaming statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            "is_running": self.is_running,
            "frames_written": self.frames_written,
            "elapsed_time": elapsed_time,
            "fps_actual": self.frames_written / elapsed_time if elapsed_time > 0 else 0,
            "fps_target": self.fps,
            "output_path": str(self.output_path),
            "segment_time": self.segment_time
        }
    
    def stop(self) -> bool:
        """
        Stop FFmpeg process gracefully or mock mode.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running:
            return True

        # Mock mode
        if self.mock_mode:
            if self.verbose:
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(f"Finished Mock FFmpeg mode stopped:")
                print(f"  - Frames written: {self.frames_written}")
                print(f"  - Duration: {elapsed:.1f}s")
                print(f"  - Average FPS: {self.frames_written / elapsed:.1f}" if elapsed > 0 else "  - Average FPS: N/A")

            self.is_running = False
            return True

        try:
            if self.process and self.process.stdin:
                self.process.stdin.close()

            if self.process:
                # Wait for process to finish
                self.process.wait(timeout=10)
                return_code = self.process.returncode

                if self.verbose:
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    print(f"Finished FFmpeg process stopped:")
                    print(f"  - Frames written: {self.frames_written}")
                    print(f"  - Duration: {elapsed:.1f}s")
                    print(f"  - Average FPS: {self.frames_written / elapsed:.1f}" if elapsed > 0 else "  - Average FPS: N/A")
                    print(f"  - Return code: {return_code}")

                if return_code != 0:
                    if self.verbose:
                        print("Warning FFmpeg process ended with non-zero return code")
                        if self.error_output:
                            print("Recent errors:")
                            for error in self.error_output[-5:]:
                                print(f"  {error}")

            self.is_running = False
            return True

        except subprocess.TimeoutExpired:
            if self.verbose:
                print("Warning FFmpeg process did not terminate gracefully, forcing...")

            if self.process:
                self.process.kill()
                self.process.wait()

            self.is_running = False
            return False
        except Exception as e:
            if self.verbose:
                print(f"ERROR Error stopping FFmpeg: {e}")
            self.is_running = False
            return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def test_ffmpeg_availability(ffmpeg_path: str = "ffmpeg") -> bool:
    """
    Test if FFmpeg is available and working.
    
    Args:
        ffmpeg_path: Path to FFmpeg executable
        
    Returns:
        True if FFmpeg is available, False otherwise
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    # Simple test
    print("[Test] Testing FFmpeg availability...")
    if test_ffmpeg_availability():
        print("OK FFmpeg is available")
    else:
        print("ERROR FFmpeg not found or not working")
