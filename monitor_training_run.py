#!/usr/bin/env python3
"""
Real-time Training Monitor for run-54d8ae4e
Checks for jumps, irregularities, and performance issues during training.
Run this periodically to track training health.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Configuration
RUN_ID = "run-54d8ae4e"
DB_PATH = Path("ml_experiments.db")

def get_run_info(cursor):
    """Get basic run information."""
    cursor.execute("""
        SELECT run_id, experiment_name, status, current_timestep, config_json, start_time
        FROM experiment_runs
        WHERE run_id = ?
    """, (RUN_ID,))
    return cursor.fetchone()

def check_backward_jumps(cursor, run_id, lookback_minutes=10):
    """Check for backward timestep jumps in recent metrics."""
    lookback_time = datetime.now() - timedelta(minutes=lookback_minutes)
    
    cursor.execute("""
        SELECT 
            m1.id, m1.timestep, m1.timestamp,
            m2.id, m2.timestep, m2.timestamp
        FROM training_metrics m1
        JOIN training_metrics m2 ON m2.id = m1.id + 1
        WHERE m1.run_id = ?
        AND m1.timestamp > ?
        AND m2.timestep < m1.timestep
    """, (run_id, lookback_time.isoformat()))
    
    return cursor.fetchall()

def check_stalled_progress(cursor, run_id, lookback_minutes=5):
    """Check if progress has stalled (no new metrics)."""
    lookback_time = datetime.now() - timedelta(minutes=lookback_minutes)
    
    cursor.execute("""
        SELECT COUNT(*) FROM training_metrics
        WHERE run_id = ? AND timestamp > ?
    """, (run_id, lookback_time.isoformat()))
    
    return cursor.fetchone()[0]

def get_recent_stats(cursor, run_id, lookback_minutes=10):
    """Get statistics from recent metrics."""
    lookback_time = datetime.now() - timedelta(minutes=lookback_minutes)
    
    cursor.execute("""
        SELECT 
            COUNT(*) as count,
            AVG(fps) as avg_fps,
            MIN(fps) as min_fps,
            MAX(fps) as max_fps,
            AVG(gpu_percent) as avg_gpu,
            MIN(gpu_percent) as min_gpu,
            MAX(gpu_percent) as max_gpu,
            AVG(gpu_memory_mb) as avg_vram,
            MIN(timestep) as min_step,
            MAX(timestep) as max_step
        FROM training_metrics
        WHERE run_id = ? AND timestamp > ?
    """, (run_id, lookback_time.isoformat()))
    
    return cursor.fetchone()

def get_overall_stats(cursor, run_id):
    """Get overall training statistics."""
    cursor.execute("""
        SELECT 
            COUNT(*) as total_metrics,
            AVG(fps) as avg_fps,
            MIN(fps) as min_fps,
            MAX(fps) as max_fps,
            AVG(gpu_percent) as avg_gpu,
            MAX(gpu_percent) as max_gpu,
            MIN(timestep) as first_step,
            MAX(timestep) as latest_step,
            MIN(timestamp) as start_time,
            MAX(timestamp) as latest_time
        FROM training_metrics
        WHERE run_id = ?
    """, (run_id,))
    
    return cursor.fetchone()

def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get run info
    run_info = get_run_info(cursor)
    if not run_info:
        print(f"❌ Run not found: {RUN_ID}")
        conn.close()
        sys.exit(1)
    
    run_id, name, status, current_step, config_json, start_time = run_info
    config = json.loads(config_json)
    total_steps = config.get('total_timesteps', 0)
    
    # Header
    print("=" * 80)
    print(f"🔍 TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"\n📋 RUN: {run_id}")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   Progress: {current_step:,} / {total_steps:,} ({current_step/total_steps*100:.2f}%)")
    
    # Check for backward jumps
    backward_jumps = check_backward_jumps(cursor, run_id, lookback_minutes=10)
    if backward_jumps:
        print(f"\n⚠️  BACKWARD JUMPS DETECTED (last 10 min):")
        for jump in backward_jumps:
            print(f"   Metric {jump[0]} ({jump[1]:,}) → Metric {jump[3]} ({jump[4]:,})")
    else:
        print(f"\n✅ No backward jumps (last 10 min)")
    
    # Check for stalled progress
    recent_metrics = check_stalled_progress(cursor, run_id, lookback_minutes=5)
    if recent_metrics == 0:
        print(f"⚠️  WARNING: No new metrics in last 5 minutes - training may have stalled!")
    else:
        print(f"✅ Active: {recent_metrics} metrics in last 5 minutes")
    
    # Recent stats (last 10 minutes)
    recent = get_recent_stats(cursor, run_id, lookback_minutes=10)
    if recent and recent[0] > 0:
        count, avg_fps, min_fps, max_fps, avg_gpu, min_gpu, max_gpu, avg_vram, min_step, max_step = recent
        steps_progress = max_step - min_step
        
        print(f"\n📊 RECENT PERFORMANCE (last 10 min):")
        print(f"   Metrics collected: {count}")
        print(f"   Steps progressed: {steps_progress:,}")
        print(f"   FPS: {avg_fps:.1f} avg (min: {min_fps:.0f}, max: {max_fps:.0f})")
        print(f"   GPU: {avg_gpu:.1f}% avg (min: {min_gpu:.0f}%, max: {max_gpu:.0f}%)")
        print(f"   VRAM: {avg_vram:.0f} MB avg")
        
        # Check for performance issues
        if avg_fps < 400:
            print(f"   ⚠️  FPS below expected (< 400)")
        if avg_gpu < 10:
            print(f"   ⚠️  Low GPU utilization (< 10%)")
    
    # Overall stats
    overall = get_overall_stats(cursor, run_id)
    if overall:
        total_metrics, avg_fps, min_fps, max_fps, avg_gpu, max_gpu, first_step, latest_step, start_time_str, latest_time = overall
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total metrics: {total_metrics:,}")
        print(f"   Total steps: {latest_step:,}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Average GPU: {avg_gpu:.1f}%")
        
        # Calculate elapsed time and ETA
        try:
            start_dt = datetime.fromisoformat(start_time_str)
            latest_dt = datetime.fromisoformat(latest_time)
            elapsed = (latest_dt - start_dt).total_seconds()
            elapsed_hours = elapsed / 3600
            
            if latest_step > 0:
                steps_per_hour = latest_step / elapsed_hours
                remaining_steps = total_steps - latest_step
                remaining_hours = remaining_steps / steps_per_hour
                
                print(f"   Elapsed: {elapsed_hours:.2f} hours")
                print(f"   ETA: {remaining_hours:.2f} hours remaining")
                print(f"   Expected completion: {(datetime.now() + timedelta(hours=remaining_hours)).strftime('%Y-%m-%d %H:%M')}")
        except:
            pass
    
    conn.close()
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

