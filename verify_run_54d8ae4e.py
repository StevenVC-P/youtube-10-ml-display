#!/usr/bin/env python3
"""Verify the new run 54d8ae4e is using correct timesteps"""

import sqlite3
import json
from pathlib import Path

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Search for the run (partial ID match)
cursor.execute("""
    SELECT run_id, experiment_name, status, current_timestep, config_json, process_pid
    FROM experiment_runs
    WHERE run_id LIKE ?
    ORDER BY start_time DESC
    LIMIT 1
""", (f"%54d8ae4e%",))

run = cursor.fetchone()

if run:
    run_id, name, status, timestep, config_json, pid = run
    config = json.loads(config_json)
    
    total_timesteps = config.get('total_timesteps', 0)
    video_hours = config.get('video_length_hours', 0)
    n_envs = config.get('n_envs', 0)
    
    print("=" * 80)
    print(f"✅ FOUND NEW RUN: {run_id}")
    print("=" * 80)
    print(f"\n📋 RUN DETAILS:")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   PID: {pid}")
    print(f"   Current Timestep: {timestep:,}")
    
    print(f"\n⚙️  CONFIGURATION:")
    print(f"   Total Timesteps: {total_timesteps:,}")
    print(f"   Video Length Hours: {video_hours}")
    print(f"   N Envs: {n_envs}")
    print(f"   Game: {config.get('env_id', 'N/A')}")
    
    print(f"\n🔍 TIMESTEP VERIFICATION:")
    
    if video_hours:
        expected_timesteps = int(video_hours * 3600 * 450)
        print(f"   Requested Duration: {video_hours} hours")
        print(f"   Expected Timesteps (450 FPS): {expected_timesteps:,}")
        print(f"   Actual Timesteps: {total_timesteps:,}")
        
        if total_timesteps == expected_timesteps:
            print(f"\n   ✅✅✅ PERFECT! TIMESTEPS ARE CORRECT!")
            print(f"   🎉 Training will run for the full {video_hours} hours!")
        elif abs(total_timesteps - expected_timesteps) < 100000:
            print(f"\n   ✅ Close enough! Within 100K steps")
            diff_hours = abs(total_timesteps - expected_timesteps) / (450 * 3600)
            print(f"   Difference: ~{diff_hours:.2f} hours")
        else:
            print(f"\n   ⚠️  MISMATCH DETECTED!")
            print(f"   Difference: {total_timesteps - expected_timesteps:,} steps")
            diff_hours = (total_timesteps - expected_timesteps) / (450 * 3600)
            print(f"   Difference: ~{diff_hours:.2f} hours")
    else:
        print(f"   ⚠️  Video length hours not set in config")
    
    # Calculate estimated duration
    if total_timesteps > 0:
        estimated_fps = 450
        estimated_hours = total_timesteps / (estimated_fps * 3600)
        
        print(f"\n⏱️  ESTIMATED DURATION:")
        print(f"   At {estimated_fps} FPS: {estimated_hours:.2f} hours")
        
        if video_hours and abs(estimated_hours - video_hours) < 0.1:
            print(f"   ✅ Duration matches requested {video_hours} hours!")
    
    # Get latest metrics to check GPU usage
    cursor.execute("""
        SELECT timestep, fps, gpu_percent, gpu_memory_mb
        FROM training_metrics
        WHERE run_id = ?
        ORDER BY timestep DESC
        LIMIT 3
    """, (run_id,))
    
    metrics = cursor.fetchall()
    if metrics:
        print(f"\n📊 LATEST METRICS:")
        print(f"   {'Step':<12} {'FPS':<8} {'GPU%':<8} {'VRAM MB':<10}")
        print(f"   {'-'*12} {'-'*8} {'-'*8} {'-'*10}")
        for ts, fps, gpu, vram in metrics:
            print(f"   {ts:<12,} {fps:<8.0f} {gpu:<8.0f} {vram:<10.0f}")
        
        latest_gpu = metrics[0][2]
        latest_vram = metrics[0][3]
        
        print(f"\n🎮 GPU STATUS:")
        if latest_gpu > 10 or latest_vram > 1000:
            print(f"   ✅ GPU IS BEING USED!")
            print(f"   Current GPU: {latest_gpu:.0f}%")
            print(f"   Current VRAM: {latest_vram:.0f} MB")
        else:
            print(f"   ⏳ GPU warming up or metrics not yet available")
    else:
        print(f"\n📊 No metrics yet (training just started)")

else:
    print(f"\n❌ Run not found with ID containing: 54d8ae4e")
    print("\nSearching for recent runs:")
    cursor.execute("""
        SELECT run_id, experiment_name, start_time
        FROM experiment_runs
        ORDER BY start_time DESC
        LIMIT 5
    """)
    recent = cursor.fetchall()
    for r in recent:
        print(f"  - {r[0]} | {r[1]} | {r[2]}")

conn.close()
print("\n" + "=" * 80)

