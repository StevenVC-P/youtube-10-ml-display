#!/usr/bin/env python3
"""Analyze run-6b6f5b95 in detail"""

import sqlite3
from pathlib import Path
from datetime import datetime

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

run_id = "run-6b6f5b95"

print("=" * 80)
print(f"DETAILED ANALYSIS: {run_id}")
print("=" * 80)

# Get run details
cursor.execute("""
    SELECT run_id, experiment_name, status, start_time, end_time, current_timestep, 
           best_reward, final_reward, config_json
    FROM experiment_runs
    WHERE run_id = ?
""", (run_id,))

run = cursor.fetchone()

if run:
    run_id, name, status, start, end, timestep, best_reward, final_reward, config = run
    
    print(f"\n📋 RUN DETAILS:")
    print(f"   Run ID: {run_id}")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   Started: {start}")
    print(f"   Ended: {end if end else 'Still running'}")
    print(f"   Current Timestep: {timestep:,}")
    print(f"   Best Reward: {best_reward}")
    print(f"   Final Reward: {final_reward}")
    
    # Calculate duration and FPS
    if start:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end) if end else datetime.now()
        duration = (end_dt - start_dt).total_seconds()
        
        print(f"\n⏱️  TIMING:")
        print(f"   Duration: {duration:.1f} seconds ({duration/60:.2f} minutes)")
        
        if timestep and duration > 0:
            avg_fps = timestep / duration
            print(f"   Average FPS: {avg_fps:,.0f} steps/second")
            
            # GPU determination
            if avg_fps > 2000:
                print(f"   ✅✅✅ EXCELLENT GPU PERFORMANCE!")
            elif avg_fps > 1500:
                print(f"   ✅✅ VERY GOOD GPU PERFORMANCE!")
            elif avg_fps > 1000:
                print(f"   ✅ GPU IS BEING USED!")
            elif avg_fps > 500:
                print(f"   ⚠️  Moderate - GPU may be underutilized")
            else:
                print(f"   ❌ Low FPS - likely CPU")
    
    # Get metrics summary
    cursor.execute("""
        SELECT COUNT(*), MIN(timestep), MAX(timestep), 
               AVG(fps), MIN(fps), MAX(fps),
               AVG(episode_reward_mean), MAX(episode_reward_mean),
               AVG(gpu_percent), MAX(gpu_percent),
               AVG(gpu_memory_mb), MAX(gpu_memory_mb)
        FROM training_metrics
        WHERE run_id = ?
    """, (run_id,))
    
    summary = cursor.fetchone()
    if summary:
        count, min_step, max_step, avg_fps, min_fps, max_fps, avg_reward, max_reward, avg_gpu, max_gpu, avg_vram, max_vram = summary
        
        print(f"\n📊 METRICS SUMMARY:")
        print(f"   Total Metric Records: {count:,}")
        print(f"   Step Range: {min_step:,} - {max_step:,}")
        print(f"\n   FPS Statistics:")
        print(f"     Average: {avg_fps:,.0f} steps/sec")
        print(f"     Min: {min_fps:,.0f} steps/sec")
        print(f"     Max: {max_fps:,.0f} steps/sec")
        print(f"\n   Reward Statistics:")
        print(f"     Average: {avg_reward:.2f}")
        print(f"     Max: {max_reward:.2f}")
        print(f"\n   GPU Usage:")
        print(f"     Average GPU%: {avg_gpu:.1f}%")
        print(f"     Max GPU%: {max_gpu:.1f}%")
        print(f"     Average VRAM: {avg_vram:.0f} MB")
        print(f"     Max VRAM: {max_vram:.0f} MB")
        
        if max_gpu > 50:
            print(f"     ✅ GPU WAS ACTIVELY USED! (Peak {max_gpu:.0f}%)")
        elif max_gpu > 20:
            print(f"     ⚠️  GPU was used but not heavily")
        else:
            print(f"     ❌ GPU usage very low")
    
    # Get latest 10 metrics
    cursor.execute("""
        SELECT timestep, fps, episode_reward_mean, gpu_percent, gpu_memory_mb, timestamp
        FROM training_metrics
        WHERE run_id = ?
        ORDER BY timestep DESC
        LIMIT 10
    """, (run_id,))
    
    latest_metrics = cursor.fetchall()
    if latest_metrics:
        print(f"\n📈 LATEST 10 METRICS:")
        print(f"   {'Step':<10} {'FPS':<8} {'Reward':<8} {'GPU%':<6} {'VRAM MB':<8} {'Time'}")
        print(f"   {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*19}")
        for ts, fps, reward, gpu, vram, time in latest_metrics:
            time_str = time.split('T')[1][:8] if 'T' in time else time[:8]
            print(f"   {ts:<10,} {fps:<8.0f} {reward:<8.2f} {gpu:<6.0f} {vram:<8.0f} {time_str}")

else:
    print(f"\n❌ Run not found: {run_id}")

conn.close()
print("\n" + "=" * 80)

