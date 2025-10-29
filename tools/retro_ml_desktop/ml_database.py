#!/usr/bin/env python3
"""
ML Metrics Database System

Provides persistent storage and efficient querying for ML experiment data.
Designed for high-performance metric ingestion and analysis.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from .ml_metrics import TrainingMetrics, ExperimentRun, ExperimentConfig


class MetricsDatabase:
    """
    High-performance SQLite database for ML experiment tracking.
    
    Features:
    - Thread-safe operations
    - Efficient metric ingestion
    - Fast querying for visualization
    - Automatic schema migration
    - Data export capabilities
    """
    
    def __init__(self, db_path: str = "ml_experiments.db"):
        """
        Initialize the metrics database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._local = threading.local()
        
        # Initialize database schema
        self._init_database()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
        
        return self._local.connection
    
    def _init_database(self):
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            
            # Experiment runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    current_timestep INTEGER DEFAULT 0,
                    config_json TEXT,
                    best_reward REAL,
                    final_reward REAL,
                    convergence_timestep INTEGER,
                    model_path TEXT,
                    log_path TEXT,
                    video_path TEXT,
                    tensorboard_path TEXT,
                    git_commit TEXT,
                    python_version TEXT,
                    dependencies_json TEXT,
                    description TEXT,
                    tags_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    process_pid INTEGER,
                    process_paused INTEGER DEFAULT 0
                )
            """)
            
            # Training metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    timestep INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    episode_count INTEGER NOT NULL,
                    total_timesteps INTEGER NOT NULL,
                    progress_pct REAL NOT NULL,
                    episode_reward_mean REAL,
                    episode_reward_std REAL,
                    episode_length_mean REAL,
                    policy_loss REAL,
                    value_loss REAL,
                    entropy_loss REAL,
                    total_loss REAL,
                    learning_rate REAL,
                    kl_divergence REAL,
                    clip_fraction REAL,
                    explained_variance REAL,
                    fps REAL,
                    steps_per_second REAL,
                    cpu_percent REAL,
                    memory_mb REAL,
                    gpu_percent REAL,
                    gpu_memory_mb REAL,
                    gradient_norm REAL,
                    weight_norm REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id),
                    UNIQUE(run_id, timestep)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_timestep ON training_metrics(run_id, timestep)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON training_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON experiment_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON experiment_runs(start_time)")
            
            conn.commit()
    
    def create_experiment_run(self, run: ExperimentRun) -> bool:
        """
        Create a new experiment run.
        
        Args:
            run: ExperimentRun object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Serialize complex fields
                config_json = json.dumps(run.config.to_dict()) if run.config else None
                dependencies_json = json.dumps(run.dependencies) if run.dependencies else None
                tags_json = json.dumps(run.tags) if run.tags else None
                
                conn.execute("""
                    INSERT OR REPLACE INTO experiment_runs (
                        run_id, experiment_name, start_time, end_time, status,
                        current_timestep, config_json, best_reward, final_reward,
                        convergence_timestep, model_path, log_path, video_path,
                        tensorboard_path, git_commit, python_version,
                        dependencies_json, description, tags_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run.run_id, run.experiment_name,
                    run.start_time.isoformat(),
                    run.end_time.isoformat() if run.end_time else None,
                    run.status, run.current_timestep, config_json,
                    run.best_reward, run.final_reward, run.convergence_timestep,
                    run.model_path, run.log_path, run.video_path,
                    run.tensorboard_path, run.git_commit, run.python_version,
                    dependencies_json, run.description, tags_json,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create experiment run {run.run_id}: {e}")
            return False
    
    def update_experiment_run(self, run_id: str, **updates) -> bool:
        """
        Update experiment run fields.
        
        Args:
            run_id: Run ID to update
            **updates: Fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Build dynamic update query
                set_clauses = []
                values = []
                
                for field, value in updates.items():
                    if field in ['config', 'dependencies', 'tags'] and value is not None:
                        # Serialize complex fields
                        if field == 'config' and hasattr(value, 'to_dict'):
                            value = json.dumps(value.to_dict())
                            field = 'config_json'
                        elif field in ['dependencies', 'tags']:
                            value = json.dumps(value)
                            field = f'{field}_json'
                    elif field in ['start_time', 'end_time'] and value is not None:
                        value = value.isoformat()
                    
                    set_clauses.append(f"{field} = ?")
                    values.append(value)
                
                if not set_clauses:
                    return True
                
                # Add updated_at timestamp
                set_clauses.append("updated_at = ?")
                values.append(datetime.now().isoformat())
                values.append(run_id)
                
                query = f"UPDATE experiment_runs SET {', '.join(set_clauses)} WHERE run_id = ?"
                conn.execute(query, values)
                conn.commit()
                
                return conn.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Failed to update experiment run {run_id}: {e}")
            return False
    
    def add_training_metrics(self, metrics: TrainingMetrics) -> bool:
        """
        Add training metrics for a timestep.
        
        Args:
            metrics: TrainingMetrics object to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                conn.execute("""
                    INSERT OR REPLACE INTO training_metrics (
                        run_id, timestep, timestamp, episode_count, total_timesteps,
                        progress_pct, episode_reward_mean, episode_reward_std,
                        episode_length_mean, policy_loss, value_loss, entropy_loss,
                        total_loss, learning_rate, kl_divergence, clip_fraction,
                        explained_variance, fps, steps_per_second, cpu_percent,
                        memory_mb, gpu_percent, gpu_memory_mb, gradient_norm, weight_norm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.run_id, metrics.timestep, metrics.timestamp.isoformat(),
                    metrics.episode_count, metrics.total_timesteps, metrics.progress_pct,
                    metrics.episode_reward_mean, metrics.episode_reward_std,
                    metrics.episode_length_mean, metrics.policy_loss, metrics.value_loss,
                    metrics.entropy_loss, metrics.total_loss, metrics.learning_rate,
                    metrics.kl_divergence, metrics.clip_fraction, metrics.explained_variance,
                    metrics.fps, metrics.steps_per_second, metrics.cpu_percent,
                    metrics.memory_mb, metrics.gpu_percent, metrics.gpu_memory_mb,
                    metrics.gradient_norm, metrics.weight_norm
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add metrics for {metrics.run_id} at step {metrics.timestep}: {e}")
            return False
    
    def get_experiment_runs(self, status: Optional[str] = None, limit: Optional[int] = None) -> List[ExperimentRun]:
        """
        Get experiment runs with optional filtering.
        
        Args:
            status: Filter by status (running, completed, etc.)
            limit: Maximum number of runs to return
            
        Returns:
            List of ExperimentRun objects
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                query = "SELECT * FROM experiment_runs"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status)
                
                query += " ORDER BY start_time DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                runs = []
                for row in rows:
                    # Deserialize complex fields
                    config = None
                    if row['config_json']:
                        config_data = json.loads(row['config_json'])
                        config = ExperimentConfig.from_dict(config_data)
                    
                    dependencies = json.loads(row['dependencies_json']) if row['dependencies_json'] else None
                    tags = json.loads(row['tags_json']) if row['tags_json'] else None
                    
                    run = ExperimentRun(
                        run_id=row['run_id'],
                        experiment_name=row['experiment_name'],
                        start_time=datetime.fromisoformat(row['start_time']),
                        end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                        status=row['status'],
                        current_timestep=row['current_timestep'],
                        config=config,
                        best_reward=row['best_reward'],
                        final_reward=row['final_reward'],
                        convergence_timestep=row['convergence_timestep'],
                        model_path=row['model_path'],
                        log_path=row['log_path'],
                        video_path=row['video_path'],
                        tensorboard_path=row['tensorboard_path'],
                        git_commit=row['git_commit'],
                        python_version=row['python_version'],
                        dependencies=dependencies,
                        description=row['description'],
                        tags=tags
                    )
                    runs.append(run)
                
                return runs
                
        except Exception as e:
            self.logger.error(f"Failed to get experiment runs: {e}")
            return []
    
    def get_training_metrics(self, run_id: str, 
                           start_timestep: Optional[int] = None,
                           end_timestep: Optional[int] = None,
                           limit: Optional[int] = None) -> List[TrainingMetrics]:
        """
        Get training metrics for a run with optional filtering.
        
        Args:
            run_id: Run ID to get metrics for
            start_timestep: Minimum timestep (inclusive)
            end_timestep: Maximum timestep (inclusive)
            limit: Maximum number of metrics to return
            
        Returns:
            List of TrainingMetrics objects
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                query = "SELECT * FROM training_metrics WHERE run_id = ?"
                params = [run_id]
                
                if start_timestep is not None:
                    query += " AND timestep >= ?"
                    params.append(start_timestep)
                
                if end_timestep is not None:
                    query += " AND timestep <= ?"
                    params.append(end_timestep)
                
                query += " ORDER BY timestep ASC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metric = TrainingMetrics(
                        run_id=row['run_id'],
                        timestep=row['timestep'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        episode_count=row['episode_count'],
                        total_timesteps=row['total_timesteps'],
                        progress_pct=row['progress_pct'],
                        episode_reward_mean=row['episode_reward_mean'],
                        episode_reward_std=row['episode_reward_std'],
                        episode_length_mean=row['episode_length_mean'],
                        policy_loss=row['policy_loss'],
                        value_loss=row['value_loss'],
                        entropy_loss=row['entropy_loss'],
                        total_loss=row['total_loss'],
                        learning_rate=row['learning_rate'],
                        kl_divergence=row['kl_divergence'],
                        clip_fraction=row['clip_fraction'],
                        explained_variance=row['explained_variance'],
                        fps=row['fps'],
                        steps_per_second=row['steps_per_second'],
                        cpu_percent=row['cpu_percent'],
                        memory_mb=row['memory_mb'],
                        gpu_percent=row['gpu_percent'],
                        gpu_memory_mb=row['gpu_memory_mb'],
                        gradient_norm=row['gradient_norm'],
                        weight_norm=row['weight_norm']
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get training metrics for {run_id}: {e}")
            return []
    
    def get_latest_metrics(self, run_id: str) -> Optional[TrainingMetrics]:
        """Get the most recent metrics for a run."""
        metrics = self.get_training_metrics(run_id, limit=1)
        return metrics[0] if metrics else None
    
    def delete_experiment_run(self, run_id: str) -> bool:
        """
        Delete an experiment run and all its metrics.
        
        Args:
            run_id: Run ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()
                
                # Delete metrics first (foreign key constraint)
                conn.execute("DELETE FROM training_metrics WHERE run_id = ?", (run_id,))
                
                # Delete experiment run
                conn.execute("DELETE FROM experiment_runs WHERE run_id = ?", (run_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete experiment run {run_id}: {e}")
            return False
    
    def export_to_json(self, run_id: str, output_path: str) -> bool:
        """
        Export experiment run and metrics to JSON file.
        
        Args:
            run_id: Run ID to export
            output_path: Path to output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            runs = self.get_experiment_runs()
            run = next((r for r in runs if r.run_id == run_id), None)
            
            if not run:
                return False
            
            metrics = self.get_training_metrics(run_id)
            
            export_data = {
                "experiment_run": run.to_dict(),
                "training_metrics": [m.to_dict() for m in metrics]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export run {run_id}: {e}")
            return False
    
    def get_run_summary_stats(self, run_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a training run.

        Args:
            run_id: Run ID to analyze

        Returns:
            Dictionary with summary statistics
        """
        try:
            with self._lock:
                conn = self._get_connection()

                # Get basic run info
                run_cursor = conn.execute("SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,))
                run_row = run_cursor.fetchone()

                if not run_row:
                    return {}

                # Get metrics statistics
                stats_cursor = conn.execute("""
                    SELECT
                        COUNT(*) as total_metrics,
                        MIN(timestep) as min_timestep,
                        MAX(timestep) as max_timestep,
                        AVG(episode_reward_mean) as avg_reward,
                        MAX(episode_reward_mean) as max_reward,
                        MIN(episode_reward_mean) as min_reward,
                        AVG(fps) as avg_fps,
                        MAX(fps) as max_fps
                    FROM training_metrics
                    WHERE run_id = ? AND episode_reward_mean IS NOT NULL
                """, (run_id,))

                stats_row = stats_cursor.fetchone()

                return {
                    "run_id": run_id,
                    "status": run_row['status'],
                    "current_timestep": run_row['current_timestep'],
                    "total_metrics": stats_row['total_metrics'] if stats_row else 0,
                    "timestep_range": [stats_row['min_timestep'], stats_row['max_timestep']] if stats_row and stats_row['min_timestep'] else [0, 0],
                    "reward_stats": {
                        "mean": stats_row['avg_reward'],
                        "max": stats_row['max_reward'],
                        "min": stats_row['min_reward']
                    } if stats_row and stats_row['avg_reward'] else None,
                    "fps_stats": {
                        "mean": stats_row['avg_fps'],
                        "max": stats_row['max_fps']
                    } if stats_row and stats_row['avg_fps'] else None
                }

        except Exception as e:
            self.logger.error(f"Failed to get summary stats for {run_id}: {e}")
            return {}

    def update_process_info(self, run_id: str, pid: int = None, paused: bool = None):
        """Update process information for an experiment run."""
        with self._lock:
            conn = self._get_connection()

            updates = []
            params = []

            if pid is not None:
                updates.append("process_pid = ?")
                params.append(pid)

            if paused is not None:
                updates.append("process_paused = ?")
                params.append(1 if paused else 0)

            if updates:
                updates.append("updated_at = ?")
                params.append(datetime.now().isoformat())
                params.append(run_id)

                query = f"UPDATE experiment_runs SET {', '.join(updates)} WHERE run_id = ?"
                conn.execute(query, params)
                conn.commit()

    def get_paused_experiments(self) -> List[Dict[str, Any]]:
        """Get all paused experiments that can be resumed."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT run_id, experiment_name, start_time, current_timestep,
                       config_json, process_pid
                FROM experiment_runs
                WHERE process_paused = 1 AND status = 'running'
                ORDER BY start_time DESC
            """)

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_running_experiments_with_pids(self) -> List[Dict[str, Any]]:
        """Get all running experiments with their process PIDs."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT run_id, experiment_name, start_time, current_timestep,
                       process_pid, process_paused
                FROM experiment_runs
                WHERE status = 'running'
                ORDER BY start_time DESC
            """)

            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def close(self):
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
