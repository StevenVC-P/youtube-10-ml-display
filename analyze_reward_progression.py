#!/usr/bin/env python3
"""Analyze reward progression throughout training to check if video is showing correct timesteps"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

db_path = Path("ml_experiments.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

RUN_ID = "run-54d8ae4e"

# Get run info
cursor.execute("""
    SELECT experiment_name, config_json, start_time, end_time
    FROM experiment_runs
    WHERE run_id = ?
""", (RUN_ID,))

run_info = cursor.fetchone()
if not run_info:
    print(f"❌ Run {RUN_ID} not found!")
    exit(1)

name, config_json, start_time, end_time = run_info
config = json.loads(config_json)
total_timesteps = config.get('total_timesteps', 0)

print("=" * 80)
print(f"📊 REWARD PROGRESSION ANALYSIS: {RUN_ID}")
print("=" * 80)
print(f"Game: {name}")
print(f"Total Timesteps: {total_timesteps:,}")
print(f"Training Duration: {start_time} → {end_time}")
print("")

# Get reward progression in 10 segments (representing each hour of 10-hour training)
segment_size = total_timesteps // 10

print("=" * 80)
print("🎮 PERFORMANCE BY HOUR (10 segments):")
print("=" * 80)
print("")

for hour in range(1, 11):
    start_step = (hour - 1) * segment_size
    end_step = hour * segment_size
    
    # Get metrics in this range
    cursor.execute("""
        SELECT 
            COUNT(*) as metric_count,
            AVG(episode_reward_mean) as avg_reward,
            MAX(episode_reward_mean) as max_reward,
            MIN(episode_reward_mean) as min_reward,
            MIN(timestep) as first_step,
            MAX(timestep) as last_step
        FROM training_metrics
        WHERE run_id = ?
        AND timestep >= ?
        AND timestep < ?
        AND episode_reward_mean IS NOT NULL
    """, (RUN_ID, start_step, end_step))
    
    result = cursor.fetchone()
    metric_count, avg_reward, max_reward, min_reward, first_step, last_step = result
    
    if metric_count and metric_count > 0:
        print(f"Hour {hour:2d} (steps {start_step:,} - {end_step:,}):")
        print(f"  Metrics: {metric_count}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Max Reward: {max_reward:.2f}")
        print(f"  Min Reward: {min_reward:.2f}")
        
        # Highlight if this is unusually good for early training
        if hour == 1 and avg_reward > 10:
            print(f"  ⚠️  SUSPICIOUSLY HIGH for Hour 1! Expected ~1-5 for early training")
        elif hour == 10 and avg_reward < 10:
            print(f"  ⚠️  SUSPICIOUSLY LOW for Hour 10! Expected >15 for late training")
        
        print("")
    else:
        print(f"Hour {hour:2d}: No data")
        print("")

# Get overall progression
print("=" * 80)
print("📈 OVERALL REWARD PROGRESSION:")
print("=" * 80)
print("")

cursor.execute("""
    SELECT 
        timestep,
        episode_reward_mean,
        timestamp
    FROM training_metrics
    WHERE run_id = ?
    AND episode_reward_mean IS NOT NULL
    ORDER BY timestep ASC
""", (RUN_ID,))

all_rewards = cursor.fetchall()

if all_rewards:
    # Show first 10 metrics
    print("First 10 metrics (early training):")
    for i, (step, reward, ts) in enumerate(all_rewards[:10]):
        print(f"  Step {step:,}: Reward = {reward:.2f}")
    
    print("")
    
    # Show last 10 metrics
    print("Last 10 metrics (late training):")
    for i, (step, reward, ts) in enumerate(all_rewards[-10:]):
        print(f"  Step {step:,}: Reward = {reward:.2f}")
    
    print("")
    
    # Calculate improvement
    early_avg = sum(r[1] for r in all_rewards[:100]) / min(100, len(all_rewards))
    late_avg = sum(r[1] for r in all_rewards[-100:]) / min(100, len(all_rewards))
    
    print(f"Early Training Avg (first 100 metrics): {early_avg:.2f}")
    print(f"Late Training Avg (last 100 metrics): {late_avg:.2f}")
    print(f"Improvement: {late_avg - early_avg:.2f} ({((late_avg / early_avg - 1) * 100):.1f}% increase)")
    print("")

# Check if video might be showing wrong timesteps
print("=" * 80)
print("🎬 VIDEO GENERATION CHECK:")
print("=" * 80)
print("")

# If user says "first hour looks really good", check what "really good" means
cursor.execute("""
    SELECT AVG(episode_reward_mean)
    FROM training_metrics
    WHERE run_id = ?
    AND timestep < ?
    AND episode_reward_mean IS NOT NULL
""", (RUN_ID, segment_size))

first_hour_avg = cursor.fetchone()[0]

cursor.execute("""
    SELECT AVG(episode_reward_mean)
    FROM training_metrics
    WHERE run_id = ?
    AND timestep >= ?
    AND episode_reward_mean IS NOT NULL
""", (RUN_ID, total_timesteps - segment_size))

last_hour_avg = cursor.fetchone()[0]

print(f"First Hour Average Reward: {first_hour_avg:.2f}")
print(f"Last Hour Average Reward: {last_hour_avg:.2f}")
print("")

if first_hour_avg and last_hour_avg:
    if first_hour_avg > (last_hour_avg * 0.8):
        print("⚠️  WARNING: First hour performance is suspiciously close to last hour!")
        print("   This suggests the video might be showing the WRONG timesteps.")
        print("   Expected: First hour should be much worse than last hour.")
    else:
        print("✅ Performance progression looks normal.")
        print("   First hour is significantly worse than last hour, as expected.")

conn.close()
print("\n" + "=" * 80)

