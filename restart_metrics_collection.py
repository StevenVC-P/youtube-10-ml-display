#!/usr/bin/env python3
"""
Restart MetricsCollector for stale training runs.

This script identifies training runs that are marked as "running" in the database
but don't have active metrics collection, and restarts the metrics collection for them.
"""

import sys
import time
from pathlib import Path
from tools.retro_ml_desktop.ml_database import MetricsDatabase
from tools.retro_ml_desktop.ml_collector import MetricsCollector
from tools.retro_ml_desktop.process_manager import ProcessManager
import psutil


def find_training_processes():
    """Find all running training processes."""
    training_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'status']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'train.py' in cmdline:
                    # Extract run ID from command line
                    run_id = None
                    for part in cmdline.split():
                        if part.startswith('run-'):
                            run_id = part
                            break
                    
                    if run_id:
                        training_processes.append({
                            'pid': proc.info['pid'],
                            'run_id': run_id,
                            'cmdline': cmdline,
                            'status': proc.info['status'],
                            'create_time': proc.info['create_time']
                        })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    return training_processes


def restart_metrics_collection():
    """Restart metrics collection for stale runs."""
    print("🔍 Checking for stale training runs...")
    
    # Initialize components
    project_root = Path('.')
    db = MetricsDatabase(str(project_root / "ml_experiments.db"))
    collector = MetricsCollector(db)
    process_manager = ProcessManager(str(project_root))
    
    # Get runs from database
    db_runs = db.get_experiment_runs(limit=10)
    running_db_runs = [run for run in db_runs if run.status == 'running']
    
    # Get active collectors
    active_collectors = collector.get_active_runs()
    
    # Find training processes
    training_processes = find_training_processes()
    
    print(f"📊 Found:")
    print(f"  - {len(running_db_runs)} runs marked as 'running' in database")
    print(f"  - {len(active_collectors)} active metrics collectors")
    print(f"  - {len(training_processes)} actual training processes")
    
    # Find stale runs (in database as running, but no active collector)
    stale_runs = []
    for db_run in running_db_runs:
        if db_run.run_id not in active_collectors:
            # Check if there's a corresponding training process
            matching_process = None
            for proc in training_processes:
                if proc['run_id'] == db_run.run_id:
                    matching_process = proc
                    break
            
            stale_runs.append({
                'db_run': db_run,
                'process': matching_process
            })
    
    if not stale_runs:
        print("✅ No stale runs found. All running processes have active metrics collection.")
        return
    
    print(f"\n🚨 Found {len(stale_runs)} stale runs:")
    
    for i, stale in enumerate(stale_runs, 1):
        db_run = stale['db_run']
        process = stale['process']
        
        print(f"\n{i}. Run: {db_run.run_id}")
        print(f"   Database status: {db_run.status}")
        print(f"   Start time: {db_run.start_time}")
        
        if process:
            print(f"   Process PID: {process['pid']} (status: {process['status']})")
            
            # Try to restart metrics collection
            try:
                def get_logs():
                    return process_manager.get_recent_logs(db_run.run_id) or ""
                
                collector.start_collection(
                    run_id=db_run.run_id,
                    log_source=get_logs,
                    pid=process['pid'],
                    interval=10.0
                )
                
                print(f"   ✅ Restarted metrics collection")
                
            except Exception as e:
                print(f"   ❌ Failed to restart metrics collection: {e}")
        else:
            print(f"   ❌ No corresponding training process found")
            print(f"   💡 Consider marking this run as 'failed' in database")
            
            # Update database to mark as failed
            try:
                db.update_experiment_run(db_run.run_id, status='failed', end_time=time.time())
                print(f"   ✅ Marked run as 'failed' in database")
            except Exception as e:
                print(f"   ❌ Failed to update database: {e}")
    
    # Show final status
    print(f"\n📈 Final status:")
    active_collectors_after = collector.get_active_runs()
    print(f"  - Active collectors: {len(active_collectors_after)}")
    for run_id in active_collectors_after:
        print(f"    • {run_id}")


if __name__ == "__main__":
    try:
        restart_metrics_collection()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
