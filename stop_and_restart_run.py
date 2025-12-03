#!/usr/bin/env python3
"""Stop current run and prepare for restart with correct timesteps"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

run_id = "run-6b6f5b95"

print("=" * 80)
print(f"STOPPING RUN: {run_id}")
print("=" * 80)

# Get run details
cursor.execute("""
    SELECT run_id, experiment_name, status, start_time, current_timestep, 
           best_reward, config_json, process_pid
    FROM experiment_runs
    WHERE run_id = ?
""", (run_id,))

run = cursor.fetchone()

if run:
    run_id, name, status, start, timestep, best_reward, config_json, pid = run
    
    print(f"\n📋 CURRENT RUN:")
    print(f"   Run ID: {run_id}")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   PID: {pid}")
    print(f"   Current Timestep: {timestep:,}")
    print(f"   Best Reward: {best_reward}")
    
    if config_json:
        config = json.loads(config_json)
        print(f"\n⚙️  CURRENT CONFIG:")
        print(f"   Total Timesteps: {config.get('total_timesteps', 'N/A'):,}")
        print(f"   N Envs: {config.get('n_envs', 'N/A')}")
        print(f"   Game: {config.get('env_id', 'N/A')}")
    
    # Update status to stopped
    cursor.execute("""
        UPDATE experiment_runs
        SET status = 'stopped',
            end_time = ?
        WHERE run_id = ?
    """, (datetime.now().isoformat(), run_id))
    
    conn.commit()
    
    print(f"\n✅ Run status updated to 'stopped'")
    print(f"\n💡 NEXT STEPS:")
    print(f"   1. Kill the training process (PID: {pid})")
    print(f"   2. Start a new training with corrected timesteps")
    print(f"   3. New run will use: 16,200,000 timesteps for 10 hours")
    
    print(f"\n📊 CORRECTED CALCULATION:")
    print(f"   Target Duration: 10 hours")
    print(f"   Estimated FPS: 450 steps/sec")
    print(f"   Total Timesteps: 10 * 3600 * 450 = 16,200,000")
    print(f"   Old (Wrong): 10,000,000 (~6.1 hours)")
    print(f"   New (Correct): 16,200,000 (~10 hours)")

else:
    print(f"\n❌ Run not found: {run_id}")

conn.close()
print("\n" + "=" * 80)

