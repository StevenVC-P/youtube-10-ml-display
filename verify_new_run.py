#!/usr/bin/env python3
"""Verify the new run is using correct timesteps"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

run_id = "run-974d17a6"

print("=" * 80)
print(f"VERIFYING NEW RUN: {run_id}")
print("=" * 80)

# Search for the run (partial ID match)
cursor.execute("""
    SELECT run_id, experiment_name, status, start_time, current_timestep, 
           best_reward, config_json, process_pid
    FROM experiment_runs
    WHERE run_id LIKE ?
    ORDER BY start_time DESC
    LIMIT 1
""", (f"{run_id}%",))

run = cursor.fetchone()

if run:
    run_id, name, status, start, timestep, best_reward, config_json, pid = run
    
    print(f"\n✅ FOUND NEW RUN!")
    print(f"\n📋 RUN DETAILS:")
    print(f"   Run ID: {run_id}")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   PID: {pid}")
    print(f"   Started: {start}")
    print(f"   Current Timestep: {timestep:,}")
    print(f"   Best Reward: {best_reward}")
    
    if config_json:
        config = json.loads(config_json)
        
        total_timesteps = config.get('total_timesteps', 0)
        video_hours = config.get('video_length_hours', 0)
        n_envs = config.get('n_envs', 0)
        
        print(f"\n⚙️  CONFIGURATION:")
        print(f"   Total Timesteps: {total_timesteps:,}")
        print(f"   Video Length Hours: {video_hours}")
        print(f"   N Envs: {n_envs}")
        print(f"   Game: {config.get('env_id', 'N/A')}")
        print(f"   Algorithm: {config.get('algorithm', 'N/A')}")
        print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
        
        # Verify timesteps are correct
        print(f"\n🔍 TIMESTEP VERIFICATION:")
        
        if video_hours:
            expected_timesteps = int(video_hours * 3600 * 450)
            print(f"   Requested Duration: {video_hours} hours")
            print(f"   Expected Timesteps: {expected_timesteps:,}")
            print(f"   Actual Timesteps: {total_timesteps:,}")
            
            if total_timesteps == expected_timesteps:
                print(f"   ✅✅✅ PERFECT! Timesteps are CORRECT!")
            elif abs(total_timesteps - expected_timesteps) < 1000000:
                print(f"   ✅ Close enough! Within 1M steps")
            else:
                print(f"   ⚠️  Mismatch detected!")
                print(f"   Difference: {total_timesteps - expected_timesteps:,} steps")
        
        # Calculate estimated duration
        if total_timesteps > 0:
            estimated_fps = 450  # Conservative estimate
            estimated_hours = total_timesteps / (estimated_fps * 3600)
            
            print(f"\n⏱️  ESTIMATED DURATION:")
            print(f"   At {estimated_fps} FPS: {estimated_hours:.2f} hours")
            
            if video_hours and abs(estimated_hours - video_hours) < 0.5:
                print(f"   ✅ Duration matches requested {video_hours} hours!")
            elif video_hours:
                print(f"   ⚠️  Duration mismatch: {estimated_hours:.2f}h vs {video_hours}h requested")
    
    # Get latest metrics
    cursor.execute("""
        SELECT timestep, fps, episode_reward_mean, gpu_percent, gpu_memory_mb, timestamp
        FROM training_metrics
        WHERE run_id = ?
        ORDER BY timestep DESC
        LIMIT 5
    """, (run_id,))
    
    metrics = cursor.fetchall()
    if metrics:
        print(f"\n📊 LATEST METRICS:")
        print(f"   {'Step':<10} {'FPS':<8} {'Reward':<8} {'GPU%':<6} {'VRAM MB':<8}")
        print(f"   {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
        for ts, fps, reward, gpu, vram, time in metrics:
            reward_str = f"{reward:.2f}" if reward is not None else "N/A"
            print(f"   {ts:<10,} {fps:<8.0f} {reward_str:<8} {gpu:<6.0f} {vram:<8.0f}")
        
        # Check GPU usage
        latest_gpu = metrics[0][3]
        latest_vram = metrics[0][4]
        
        print(f"\n🎮 GPU STATUS:")
        if latest_gpu > 50 or latest_vram > 1000:
            print(f"   ✅ GPU IS BEING USED!")
            print(f"   Current GPU: {latest_gpu:.0f}%")
            print(f"   Current VRAM: {latest_vram:.0f} MB")
        else:
            print(f"   ⚠️  GPU usage seems low (might be warming up)")
            print(f"   Current GPU: {latest_gpu:.0f}%")
            print(f"   Current VRAM: {latest_vram:.0f} MB")

else:
    print(f"\n❌ Run not found: {run_id}")
    print("\nSearching for recent runs:")
    cursor.execute("""
        SELECT run_id, experiment_name, start_time, current_timestep
        FROM experiment_runs
        ORDER BY start_time DESC
        LIMIT 5
    """)
    recent = cursor.fetchall()
    for r in recent:
        print(f"  - {r[0]} | {r[1]} | {r[2]} | {r[3]:,} steps")

conn.close()
print("\n" + "=" * 80)

