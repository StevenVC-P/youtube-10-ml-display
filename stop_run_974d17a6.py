#!/usr/bin/env python3
"""Stop the current run 974d17a6"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

run_id = "run-974d17a6"

print("=" * 80)
print(f"STOPPING RUN: {run_id}")
print("=" * 80)

# Get run details
cursor.execute("""
    SELECT run_id, experiment_name, status, process_pid, current_timestep, config_json
    FROM experiment_runs
    WHERE run_id = ?
""", (run_id,))

run = cursor.fetchone()

if run:
    run_id, name, status, pid, timestep, config_json = run
    
    print(f"\n📋 CURRENT RUN:")
    print(f"   Run ID: {run_id}")
    print(f"   Name: {name}")
    print(f"   Status: {status}")
    print(f"   PID: {pid}")
    print(f"   Current Timestep: {timestep:,}")
    
    if config_json:
        config = json.loads(config_json)
        print(f"   Total Timesteps: {config.get('total_timesteps', 'N/A'):,}")
        print(f"   Video Length Hours: {config.get('video_length_hours', 'N/A')}")
    
    # Update status to stopped
    cursor.execute("""
        UPDATE experiment_runs
        SET status = 'stopped',
            end_time = ?
        WHERE run_id = ?
    """, (datetime.now().isoformat(), run_id))
    
    conn.commit()
    
    print(f"\n✅ Run status updated to 'stopped'")
    print(f"\n💡 NEXT STEP:")
    print(f"   Kill process PID: {pid}")
    print(f"   Command: taskkill /F /PID {pid}")

else:
    print(f"\n❌ Run not found: {run_id}")

conn.close()
print("\n" + "=" * 80)

