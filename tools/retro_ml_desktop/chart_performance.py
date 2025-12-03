#!/usr/bin/env python3
"""
Chart Performance Optimization Module

Provides performance enhancements for chart rendering including:
- Data downsampling using Largest-Triangle-Three-Buckets (LTTB) algorithm
- Frame rate limiting to prevent excessive redraws
- Incremental update tracking
"""

import time
import logging
from typing import List, Tuple, Optional
import numpy as np


class DataDownsampler:
    """
    Implements data downsampling algorithms for efficient chart rendering.
    
    Uses the Largest-Triangle-Three-Buckets (LTTB) algorithm which preserves
    visual characteristics while reducing the number of data points.
    """
    
    @staticmethod
    def lttb_downsample(x_data: List[float], y_data: List[float], 
                        threshold: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Downsample data using Largest-Triangle-Three-Buckets algorithm.
        
        This algorithm preserves the visual characteristics of the data while
        reducing the number of points. It's particularly effective for time-series
        data where maintaining the overall shape is important.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            threshold: Target number of points after downsampling
            
        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        # If data is already small enough, return as-is
        if len(x_data) <= threshold:
            return x_data, y_data
        
        # Convert to numpy arrays for faster computation
        x = np.array(x_data)
        y = np.array(y_data)
        
        # Always include first and last points
        sampled_x = [x[0]]
        sampled_y = [y[0]]
        
        # Bucket size (number of points per bucket)
        bucket_size = (len(x) - 2) / (threshold - 2)
        
        # Index of the point in the previous bucket
        a = 0
        
        for i in range(threshold - 2):
            # Calculate point average for next bucket
            avg_range_start = int((i + 1) * bucket_size) + 1
            avg_range_end = int((i + 2) * bucket_size) + 1
            avg_range_end = min(avg_range_end, len(x))
            
            avg_x = np.mean(x[avg_range_start:avg_range_end])
            avg_y = np.mean(y[avg_range_start:avg_range_end])
            
            # Get the range for this bucket
            range_start = int(i * bucket_size) + 1
            range_end = int((i + 1) * bucket_size) + 1
            
            # Point a (previous selected point)
            point_a_x = x[a]
            point_a_y = y[a]
            
            max_area = -1
            max_area_point = range_start
            
            # Find point with largest triangle area
            for idx in range(range_start, range_end):
                # Calculate triangle area
                area = abs(
                    (point_a_x - avg_x) * (y[idx] - point_a_y) -
                    (point_a_x - x[idx]) * (avg_y - point_a_y)
                ) * 0.5
                
                if area > max_area:
                    max_area = area
                    max_area_point = idx
            
            # Select point with largest area
            sampled_x.append(x[max_area_point])
            sampled_y.append(y[max_area_point])
            a = max_area_point
        
        # Always include last point
        sampled_x.append(x[-1])
        sampled_y.append(y[-1])
        
        return sampled_x, sampled_y


class FrameRateLimiter:
    """
    Limits the frame rate of chart updates to prevent excessive CPU usage.
    
    Ensures that chart updates don't happen more frequently than the specified
    maximum FPS, which improves performance and reduces CPU usage.
    """
    
    def __init__(self, max_fps: int = 30):
        """
        Initialize frame rate limiter.
        
        Args:
            max_fps: Maximum frames per second (default: 30)
        """
        self.max_fps = max_fps
        self.min_frame_time = 1.0 / max_fps
        self.last_update_time = 0
        self.logger = logging.getLogger(__name__)
    
    def should_update(self) -> bool:
        """
        Check if enough time has passed since last update.
        
        Returns:
            True if update should proceed, False otherwise
        """
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        if time_since_last_update >= self.min_frame_time:
            self.last_update_time = current_time
            return True
        
        return False
    
    def reset(self):
        """Reset the frame rate limiter."""
        self.last_update_time = 0


class IncrementalUpdateTracker:
    """
    Tracks incremental updates to avoid re-rendering unchanged data.
    
    Maintains a record of the last rendered data point for each run,
    allowing only new data to be added to the chart.
    """
    
    def __init__(self):
        """Initialize incremental update tracker."""
        self.last_rendered_indices = {}  # run_id -> last_index
        self.logger = logging.getLogger(__name__)
    
    def get_new_data_range(self, run_id: str, total_points: int) -> Optional[Tuple[int, int]]:
        """
        Get the range of new data points that need to be rendered.
        
        Args:
            run_id: Run identifier
            total_points: Total number of data points available
            
        Returns:
            Tuple of (start_index, end_index) for new data, or None if no new data
        """
        last_index = self.last_rendered_indices.get(run_id, 0)
        
        if total_points > last_index:
            return (last_index, total_points)
        
        return None
    
    def update_rendered_index(self, run_id: str, index: int):
        """
        Update the last rendered index for a run.
        
        Args:
            run_id: Run identifier
            index: Last rendered data point index
        """
        self.last_rendered_indices[run_id] = index
    
    def reset(self, run_id: Optional[str] = None):
        """
        Reset tracking for a specific run or all runs.
        
        Args:
            run_id: Run identifier to reset, or None to reset all
        """
        if run_id:
            self.last_rendered_indices.pop(run_id, None)
        else:
            self.last_rendered_indices.clear()

