#!/usr/bin/env python3
"""
Comprehensive Training Report Generator for run-54d8ae4e
Generates a detailed markdown report for the completed training run.
Run this after training completes to create a report for the video.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Configuration
RUN_ID = "run-54d8ae4e"
DB_PATH = Path("ml_experiments.db")
OUTPUT_FILE = Path(f"training_report_{RUN_ID}.md")

def analyze_training_run(cursor, run_id):
    """Perform comprehensive analysis of the training run."""
    
    # Get run info
    cursor.execute("""
        SELECT run_id, experiment_name, status, current_timestep, config_json, 
               start_time, end_time, best_reward
        FROM experiment_runs
        WHERE run_id = ?
    """, (run_id,))
    
    run_info = cursor.fetchone()
    if not run_info:
        return None
    
    run_id, name, status, current_step, config_json, start_time, end_time = run_info[:7]
    best_reward = run_info[7] if len(run_info) > 7 else None
    config = json.loads(config_json)
    
    # Get all metrics
    cursor.execute("""
        SELECT 
            COUNT(*) as total_metrics,
            MIN(timestep) as first_step,
            MAX(timestep) as last_step,
            AVG(fps) as avg_fps,
            MIN(fps) as min_fps,
            MAX(fps) as max_fps,
            AVG(gpu_percent) as avg_gpu,
            MIN(gpu_percent) as min_gpu,
            MAX(gpu_percent) as max_gpu,
            AVG(gpu_memory_mb) as avg_vram,
            MAX(gpu_memory_mb) as max_vram,
            MIN(timestamp) as first_time,
            MAX(timestamp) as last_time
        FROM training_metrics
        WHERE run_id = ?
    """, (run_id,))
    
    metrics_stats = cursor.fetchone()
    
    # Check for anomalies
    cursor.execute("""
        SELECT 
            m1.id, m1.timestep, m1.timestamp,
            m2.id, m2.timestep, m2.timestamp
        FROM training_metrics m1
        JOIN training_metrics m2 ON m2.id = m1.id + 1
        WHERE m1.run_id = ?
        AND m2.timestep < m1.timestep
    """, (run_id,))
    
    backward_jumps = cursor.fetchall()
    
    # Get reward progression
    cursor.execute("""
        SELECT timestep, episode_reward_mean
        FROM training_metrics
        WHERE run_id = ? AND episode_reward_mean IS NOT NULL
        ORDER BY timestep
    """, (run_id,))
    
    reward_data = cursor.fetchall()
    
    # Get FPS over time (sample every 100th metric)
    cursor.execute("""
        SELECT timestep, fps, timestamp
        FROM training_metrics
        WHERE run_id = ? AND id % 100 = 0
        ORDER BY timestep
    """, (run_id,))
    
    fps_samples = cursor.fetchall()
    
    return {
        'run_info': run_info,
        'config': config,
        'metrics_stats': metrics_stats,
        'backward_jumps': backward_jumps,
        'reward_data': reward_data,
        'fps_samples': fps_samples
    }

def generate_report(analysis):
    """Generate markdown report from analysis."""
    
    run_id, name, status, current_step, config_json, start_time, end_time = analysis['run_info'][:7]
    best_reward = analysis['run_info'][7] if len(analysis['run_info']) > 7 else None
    config = analysis['config']
    stats = analysis['metrics_stats']
    
    total_metrics, first_step, last_step, avg_fps, min_fps, max_fps = stats[:6]
    avg_gpu, min_gpu, max_gpu, avg_vram, max_vram, first_time, last_time = stats[6:]
    
    total_steps = config.get('total_timesteps', 0)
    target_hours = config.get('video_length_hours', 0)
    
    # Calculate duration
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time) if end_time else datetime.fromisoformat(last_time)
        duration = (end_dt - start_dt).total_seconds()
        duration_hours = duration / 3600
    except:
        duration_hours = 0
    
    # Build report
    report = []
    report.append(f"# Training Report: {name}")
    report.append(f"\n**Run ID:** `{run_id}`")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n")
    
    # Executive Summary
    report.append("## 📊 Executive Summary\n")
    report.append(f"- **Game:** {config.get('env_id', 'N/A')}")
    report.append(f"- **Algorithm:** {config.get('algorithm', 'PPO')}")
    report.append(f"- **Status:** {status}")
    report.append(f"- **Duration:** {duration_hours:.2f} hours")
    report.append(f"- **Total Steps:** {last_step:,} / {total_steps:,} ({last_step/total_steps*100:.1f}%)")
    if best_reward:
        report.append(f"- **Best Reward:** {best_reward:.2f}")
    report.append("")
    
    # Training Configuration
    report.append("## ⚙️ Training Configuration\n")
    report.append(f"- **Total Timesteps:** {total_steps:,}")
    report.append(f"- **Target Duration:** {target_hours} hours")
    report.append(f"- **Parallel Environments:** {config.get('n_envs', 'N/A')}")
    report.append(f"- **Learning Rate:** {config.get('learning_rate', 'N/A')}")
    report.append(f"- **Batch Size:** {config.get('batch_size', 'N/A')}")
    report.append("")
    
    # Performance Metrics
    report.append("## 🚀 Performance Metrics\n")
    report.append(f"- **Average FPS:** {avg_fps:.1f} steps/second")
    report.append(f"- **FPS Range:** {min_fps:.0f} - {max_fps:.0f}")
    report.append(f"- **Average GPU Utilization:** {avg_gpu:.1f}%")
    report.append(f"- **GPU Range:** {min_gpu:.0f}% - {max_gpu:.0f}%")
    report.append(f"- **Average VRAM Usage:** {avg_vram:.0f} MB")
    report.append(f"- **Peak VRAM Usage:** {max_vram:.0f} MB")
    report.append(f"- **Total Metrics Collected:** {total_metrics:,}")
    report.append("")
    
    # Anomaly Detection
    report.append("## 🔍 Anomaly Detection\n")
    if analysis['backward_jumps']:
        report.append(f"⚠️ **{len(analysis['backward_jumps'])} backward timestep jumps detected:**\n")
        for jump in analysis['backward_jumps'][:10]:
            report.append(f"- Metric {jump[0]} ({jump[1]:,}) → Metric {jump[3]} ({jump[4]:,})")
        if len(analysis['backward_jumps']) > 10:
            report.append(f"- ... and {len(analysis['backward_jumps']) - 10} more")
    else:
        report.append("✅ **No anomalies detected**")
        report.append("- No backward timestep jumps")
        report.append("- Smooth progression throughout training")
    report.append("")
    
    # Timeline
    report.append("## ⏱️ Training Timeline\n")
    report.append(f"- **Started:** {start_time}")
    report.append(f"- **Ended:** {end_time if end_time else 'In Progress'}")
    report.append(f"- **Duration:** {duration_hours:.2f} hours")
    if target_hours > 0:
        accuracy = (duration_hours / target_hours) * 100
        report.append(f"- **Target Accuracy:** {accuracy:.1f}% of {target_hours}h target")
    report.append("")
    
    return "\n".join(report)

def main():
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        sys.exit(1)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"🔍 Analyzing training run: {RUN_ID}")
    analysis = analyze_training_run(cursor, RUN_ID)
    
    if not analysis:
        print(f"❌ Run not found: {RUN_ID}")
        conn.close()
        sys.exit(1)
    
    print(f"📝 Generating report...")
    report = generate_report(analysis)
    
    # Save report
    OUTPUT_FILE.write_text(report, encoding='utf-8')
    print(f"✅ Report saved to: {OUTPUT_FILE}")
    print(f"\n{report}")
    
    conn.close()

if __name__ == "__main__":
    main()

