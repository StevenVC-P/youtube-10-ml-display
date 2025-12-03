#!/usr/bin/env python3
"""
ML Metrics Database System

Provides persistent storage and efficient querying for ML experiment data.
Designed for high-performance metric ingestion and analysis.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
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
            
            # Video generation progress table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_generation_progress (
                    video_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    video_name TEXT NOT NULL,
                    video_path TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress_percentage REAL DEFAULT 0.0,
                    estimated_seconds_remaining INTEGER,
                    total_frames INTEGER,
                    processed_frames INTEGER DEFAULT 0,
                    started_at TEXT,
                    completed_at TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)

            # Experiments table (new experiment-centric architecture)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    game TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    preset TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    lineage_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created TEXT NOT NULL,
                    started TEXT,
                    completed TEXT,
                    tags_json TEXT,
                    notes TEXT,
                    progress_pct REAL DEFAULT 0.0,
                    current_timestep INTEGER DEFAULT 0,
                    elapsed_time REAL DEFAULT 0.0,
                    estimated_time_remaining REAL DEFAULT 0.0,
                    latest_metrics_json TEXT,
                    final_metrics_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Video artifacts table (links videos to experiments)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_artifacts (
                    video_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    type TEXT NOT NULL,
                    duration REAL,
                    size_mb REAL,
                    created TEXT NOT NULL,
                    tags_json TEXT,
                    metadata_json TEXT,
                    thumbnail_path TEXT,
                    avg_score REAL,
                    max_score REAL,
                    episode_count INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)

            # Training videos table (tracks video segments recorded during training)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    segment_number INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_mb REAL,
                    duration_seconds REAL,
                    frame_count INTEGER,
                    start_timestep INTEGER,
                    end_timestep INTEGER,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    status TEXT NOT NULL DEFAULT 'recording',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id),
                    UNIQUE(run_id, segment_number)
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_timestep ON training_metrics(run_id, timestep)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON training_metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON experiment_runs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_start_time ON experiment_runs(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_progress_status ON video_generation_progress(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_progress_run_id ON video_generation_progress(run_id)")

            # Indexes for new tables
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_game ON experiments(game)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_artifacts_experiment_id ON video_artifacts(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_artifacts_type ON video_artifacts(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_video_artifacts_created ON video_artifacts(created)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_videos_run_id ON training_videos(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_videos_status ON training_videos(status)")

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
                        dependencies_json, description, tags_json, created_at,
                        updated_at, process_pid, process_paused
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run.run_id, run.experiment_name,
                    run.start_time.isoformat(),
                    run.end_time.isoformat() if run.end_time else None,
                    run.status, run.current_timestep, config_json,
                    run.best_reward, run.final_reward, run.convergence_timestep,
                    run.model_path, run.log_path, run.video_path,
                    run.tensorboard_path, run.git_commit, run.python_version,
                    dependencies_json, run.description, tags_json,
                    datetime.now().isoformat(),  # created_at
                    datetime.now().isoformat(),  # updated_at
                    None,  # process_pid (will be updated separately)
                    0      # process_paused (default to False/0)
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
                cursor = conn.execute(query, values)
                conn.commit()

                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"Failed to update experiment run {run_id}: {e}")
            return False

    def get_experiment_run_field(self, run_id: str, field_name: str):
        """
        Get a specific field value from an experiment run.

        Args:
            run_id: Run ID to query
            field_name: Name of the field to retrieve

        Returns:
            Field value or None if not found
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(f"SELECT {field_name} FROM experiment_runs WHERE run_id = ?", (run_id,))
                result = cursor.fetchone()
                return result[0] if result else None

        except Exception as e:
            self.logger.error(f"Failed to get field {field_name} for run {run_id}: {e}")
            return None

    def auto_update_completed_runs(self) -> int:
        """
        Automatically mark runs with progress >= 100% as 'completed'.

        This prevents runs from showing as 'running' when they've actually finished.

        Returns:
            Number of runs updated
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Find runs that are marked as 'running' but have progress >= 100%
                cursor.execute("""
                    SELECT run_id, current_timestep, config_json
                    FROM experiment_runs
                    WHERE status = 'running'
                """)

                runs_to_update = []
                for row in cursor.fetchall():
                    run_id = row['run_id']
                    current_timestep = row['current_timestep'] or 0
                    config_json = row['config_json']

                    if config_json:
                        try:
                            config_data = json.loads(config_json)
                            total_timesteps = config_data.get('total_timesteps', 0)

                            if total_timesteps > 0:
                                progress = (current_timestep / total_timesteps) * 100

                                # Mark as completed if progress >= 100%
                                if progress >= 100.0:
                                    runs_to_update.append(run_id)
                        except (json.JSONDecodeError, KeyError):
                            continue

                # Update the runs
                for run_id in runs_to_update:
                    cursor.execute("""
                        UPDATE experiment_runs
                        SET status = 'completed', end_time = ?
                        WHERE run_id = ? AND status = 'running'
                    """, (datetime.now().isoformat(), run_id))

                conn.commit()

                if runs_to_update:
                    self.logger.info(f"Auto-updated {len(runs_to_update)} completed runs: {runs_to_update}")

                return len(runs_to_update)

        except Exception as e:
            self.logger.error(f"Failed to auto-update completed runs: {e}")
            return 0

    def get_runs_by_game(self, env_id: str, exclude_active: bool = False) -> List[ExperimentRun]:
        """
        Get all experiment runs for a specific game/environment.

        Args:
            env_id: Environment ID to filter by
            exclude_active: If True, exclude runs with status 'running'

        Returns:
            List of ExperimentRun objects for the specified game
        """
        try:
            with self._lock:
                conn = self._get_connection()

                query = "SELECT * FROM experiment_runs WHERE config_json LIKE ?"
                params = [f'%"env_id": "{env_id}"%']

                if exclude_active:
                    query += " AND status != 'running'"

                query += " ORDER BY start_time DESC"

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
            self.logger.error(f"Failed to get runs for game {env_id}: {e}")
            return []

    def is_run_active(self, run_id: str) -> bool:
        """
        Check if a run is currently active (status = 'running').

        Args:
            run_id: Run ID to check

        Returns:
            True if the run is active, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("SELECT status FROM experiment_runs WHERE run_id = ?", (run_id,))
                result = cursor.fetchone()

                return result and result[0] == 'running'

        except Exception as e:
            self.logger.error(f"Failed to check if run {run_id} is active: {e}")
            return False

    def mark_run_inactive(self, run_id: str, final_status: str = 'completed') -> bool:
        """
        Mark a run as inactive (not running).

        Args:
            run_id: Run ID to update
            final_status: Final status to set (completed, failed, stopped, etc.)

        Returns:
            True if successful, False otherwise
        """
        return self.update_experiment_run(
            run_id,
            status=final_status,
            end_time=datetime.now()
        )
    
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
        try:
            with self._lock:
                conn = self._get_connection()

                # Query for the latest metrics by ordering timestep DESC
                cursor = conn.execute("""
                    SELECT * FROM training_metrics
                    WHERE run_id = ?
                    ORDER BY timestep DESC
                    LIMIT 1
                """, (run_id,))

                row = cursor.fetchone()

                if row:
                    return TrainingMetrics(
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

                return None

        except Exception as e:
            self.logger.error(f"Failed to get latest metrics for {run_id}: {e}")
            return None
    
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

    # Video Generation Progress Methods

    def create_video_generation(self, video_id: str, run_id: str, video_name: str,
                                video_path: str = None, total_frames: int = None) -> bool:
        """Create a new video generation progress entry."""
        try:
            with self._lock:
                conn = self._get_connection()
                now = datetime.now().isoformat()
                conn.execute("""
                    INSERT INTO video_generation_progress
                    (video_id, run_id, video_name, video_path, status, started_at, total_frames, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'in-progress', ?, ?, ?, ?)
                """, (video_id, run_id, video_name, video_path, now, total_frames, now, now))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating video generation entry: {e}")
            return False

    def update_video_generation_progress(self, video_id: str, processed_frames: int = None,
                                        progress_percentage: float = None,
                                        estimated_seconds_remaining: int = None) -> bool:
        """Update video generation progress."""
        try:
            with self._lock:
                conn = self._get_connection()
                updates = []
                params = []

                if processed_frames is not None:
                    updates.append("processed_frames = ?")
                    params.append(processed_frames)

                if progress_percentage is not None:
                    updates.append("progress_percentage = ?")
                    params.append(progress_percentage)

                if estimated_seconds_remaining is not None:
                    updates.append("estimated_seconds_remaining = ?")
                    params.append(estimated_seconds_remaining)

                if updates:
                    updates.append("updated_at = ?")
                    params.append(datetime.now().isoformat())
                    params.append(video_id)

                    query = f"UPDATE video_generation_progress SET {', '.join(updates)} WHERE video_id = ?"
                    conn.execute(query, params)
                    conn.commit()
                    return True
                return False
        except Exception as e:
            print(f"Error updating video generation progress: {e}")
            return False

    def complete_video_generation(self, video_id: str, video_path: str = None,
                                  success: bool = True, error_message: str = None) -> bool:
        """Mark video generation as completed or failed."""
        try:
            with self._lock:
                conn = self._get_connection()
                now = datetime.now().isoformat()
                status = 'completed' if success else 'failed'

                updates = ["status = ?", "completed_at = ?", "updated_at = ?"]
                params = [status, now, now]

                if video_path:
                    updates.append("video_path = ?")
                    params.append(video_path)

                if error_message:
                    updates.append("error_message = ?")
                    params.append(error_message)

                if success:
                    updates.append("progress_percentage = ?")
                    params.append(100.0)

                params.append(video_id)

                query = f"UPDATE video_generation_progress SET {', '.join(updates)} WHERE video_id = ?"
                conn.execute(query, params)
                conn.commit()
                return True
        except Exception as e:
            print(f"Error completing video generation: {e}")
            return False

    def get_video_generation_progress(self, video_id: str = None, run_id: str = None,
                                     status: str = None) -> List[Dict[str, Any]]:
        """Get video generation progress entries."""
        try:
            with self._lock:
                conn = self._get_connection()

                query = "SELECT * FROM video_generation_progress WHERE 1=1"
                params = []

                if video_id:
                    query += " AND video_id = ?"
                    params.append(video_id)

                if run_id:
                    query += " AND run_id = ?"
                    params.append(run_id)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY created_at DESC"

                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error getting video generation progress: {e}")
            return []

    def get_in_progress_videos(self) -> List[Dict[str, Any]]:
        """Get all videos currently being generated."""
        return self.get_video_generation_progress(status='in-progress')

    def cleanup_old_video_progress(self, days: int = 7) -> int:
        """Clean up old completed/failed video generation entries."""
        try:
            with self._lock:
                conn = self._get_connection()
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

                cursor = conn.execute("""
                    DELETE FROM video_generation_progress
                    WHERE status IN ('completed', 'failed')
                    AND completed_at < ?
                """, (cutoff_date,))

                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
        except Exception as e:
            print(f"Error cleaning up old video progress: {e}")
            return 0

    # ========== Experiment Management (New) ==========

    def save_experiment(self, experiment_dict: Dict[str, Any]) -> bool:
        """
        Save or update an experiment.

        Args:
            experiment_dict: Dictionary representation of Experiment

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = self._get_connection()

                conn.execute("""
                    INSERT OR REPLACE INTO experiments (
                        experiment_id, name, game, algorithm, preset,
                        config_json, lineage_json, status, created, started,
                        completed, tags_json, notes, progress_pct,
                        current_timestep, elapsed_time, estimated_time_remaining,
                        latest_metrics_json, final_metrics_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_dict['id'],
                    experiment_dict['name'],
                    experiment_dict['game'],
                    experiment_dict['algorithm'],
                    experiment_dict['preset'],
                    json.dumps(experiment_dict['config']),
                    json.dumps(experiment_dict['lineage']),
                    experiment_dict['status'],
                    experiment_dict['created'],
                    experiment_dict.get('started'),
                    experiment_dict.get('completed'),
                    json.dumps(experiment_dict.get('tags', [])),
                    experiment_dict.get('notes', ''),
                    experiment_dict.get('progress_pct', 0.0),
                    experiment_dict.get('current_timestep', 0),
                    experiment_dict.get('elapsed_time', 0.0),
                    experiment_dict.get('estimated_time_remaining', 0.0),
                    json.dumps(experiment_dict.get('latest_metrics', {})),
                    json.dumps(experiment_dict.get('final_metrics', {})),
                    datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to save experiment: {e}")
            return False

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary representation of experiment or None
        """
        try:
            with self._lock:
                conn = self._get_connection()
                cursor = conn.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ?",
                    (experiment_id,)
                )
                row = cursor.fetchone()

                if row:
                    return self._row_to_experiment_dict(row)
                return None

        except Exception as e:
            self.logger.error(f"Failed to get experiment: {e}")
            return None

    def list_experiments(
        self,
        status: Optional[str] = None,
        game: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            status: Optional status filter
            game: Optional game filter
            limit: Maximum number to return

        Returns:
            List of experiment dictionaries
        """
        try:
            with self._lock:
                conn = self._get_connection()

                query = "SELECT * FROM experiments WHERE 1=1"
                params = []

                if status:
                    query += " AND status = ?"
                    params.append(status)

                if game:
                    query += " AND game = ?"
                    params.append(game)

                query += " ORDER BY created DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                return [self._row_to_experiment_dict(row) for row in cursor.fetchall()]

        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return []

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = self._get_connection()
                conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete experiment: {e}")
            return False

    def _row_to_experiment_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert database row to experiment dictionary."""
        return {
            'id': row['experiment_id'],
            'name': row['name'],
            'game': row['game'],
            'algorithm': row['algorithm'],
            'preset': row['preset'],
            'config': json.loads(row['config_json']),
            'lineage': json.loads(row['lineage_json']),
            'status': row['status'],
            'created': row['created'],
            'started': row['started'],
            'completed': row['completed'],
            'tags': json.loads(row['tags_json']) if row['tags_json'] else [],
            'notes': row['notes'] or '',
            'progress_pct': row['progress_pct'],
            'current_timestep': row['current_timestep'],
            'elapsed_time': row['elapsed_time'],
            'estimated_time_remaining': row['estimated_time_remaining'],
            'latest_metrics': json.loads(row['latest_metrics_json']) if row['latest_metrics_json'] else {},
            'final_metrics': json.loads(row['final_metrics_json']) if row['final_metrics_json'] else {}
        }

    # ========== Video Artifact Management (New) ==========

    def add_video_artifact(
        self,
        video_id: str,
        experiment_id: str,
        path: str,
        video_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a video artifact linked to an experiment.

        Args:
            video_id: Unique video ID
            experiment_id: Parent experiment ID
            path: Path to video file
            video_type: Type ('milestone', 'hour', 'evaluation')
            metadata: Optional metadata dictionary

        Returns:
            True if successful
        """
        try:
            from pathlib import Path as PathLib
            video_path = PathLib(path)

            # Get file size
            size_mb = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0

            # Extract metadata
            metadata = metadata or {}
            tags = metadata.get('tags', [])
            duration = metadata.get('duration')
            avg_score = metadata.get('avg_score')
            max_score = metadata.get('max_score')
            episode_count = metadata.get('episode_count')

            with self._lock:
                conn = self._get_connection()

                conn.execute("""
                    INSERT OR REPLACE INTO video_artifacts (
                        video_id, experiment_id, path, type, duration,
                        size_mb, created, tags_json, metadata_json,
                        avg_score, max_score, episode_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id,
                    experiment_id,
                    str(path),
                    video_type,
                    duration,
                    size_mb,
                    datetime.now().isoformat(),
                    json.dumps(tags),
                    json.dumps(metadata),
                    avg_score,
                    max_score,
                    episode_count,
                    datetime.now().isoformat()
                ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to add video artifact: {e}")
            return False

    def get_video_artifacts(
        self,
        experiment_id: Optional[str] = None,
        video_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get video artifacts with optional filtering.

        Args:
            experiment_id: Optional experiment ID filter
            video_type: Optional type filter
            limit: Maximum number to return

        Returns:
            List of video artifact dictionaries
        """
        try:
            with self._lock:
                conn = self._get_connection()

                query = "SELECT * FROM video_artifacts WHERE 1=1"
                params = []

                if experiment_id:
                    query += " AND experiment_id = ?"
                    params.append(experiment_id)

                if video_type:
                    query += " AND type = ?"
                    params.append(video_type)

                query += " ORDER BY created DESC LIMIT ?"
                params.append(limit)

                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                videos = []

                for row in cursor.fetchall():
                    video_dict = dict(zip(columns, row))
                    # Parse JSON fields
                    if video_dict.get('tags_json'):
                        video_dict['tags'] = json.loads(video_dict['tags_json'])
                    if video_dict.get('metadata_json'):
                        video_dict['metadata'] = json.loads(video_dict['metadata_json'])
                    videos.append(video_dict)

                return videos

        except Exception as e:
            self.logger.error(f"Failed to get video artifacts: {e}")
            return []

    def delete_video_artifact(self, video_id: str) -> bool:
        """
        Delete a video artifact.

        Args:
            video_id: Video ID

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = self._get_connection()
                conn.execute("DELETE FROM video_artifacts WHERE video_id = ?", (video_id,))
                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to delete video artifact: {e}")
            return False

    # ========================================================================
    # Training Videos Methods
    # ========================================================================

    def create_training_video_segment(
        self,
        run_id: str,
        segment_number: int,
        file_path: str,
        start_timestep: int = 0,
        start_time: Optional[str] = None
    ) -> bool:
        """
        Create a new training video segment record.

        Args:
            run_id: Training run ID
            segment_number: Segment number (1, 2, 3, ...)
            file_path: Path to the video file
            start_timestep: Starting timestep for this segment
            start_time: Start time (ISO format, defaults to now)

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = self._get_connection()

                if start_time is None:
                    start_time = datetime.now().isoformat()

                conn.execute("""
                    INSERT INTO training_videos (
                        run_id, segment_number, file_path, start_timestep, start_time, status
                    ) VALUES (?, ?, ?, ?, ?, 'recording')
                """, (run_id, segment_number, file_path, start_timestep, start_time))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to create training video segment: {e}")
            return False

    def update_training_video_segment(
        self,
        run_id: str,
        segment_number: int,
        file_size_mb: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        frame_count: Optional[int] = None,
        end_timestep: Optional[int] = None,
        end_time: Optional[str] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Update a training video segment record.

        Args:
            run_id: Training run ID
            segment_number: Segment number
            file_size_mb: File size in MB
            duration_seconds: Duration in seconds
            frame_count: Number of frames
            end_timestep: Ending timestep for this segment
            end_time: End time (ISO format)
            status: Status ('recording', 'completed', 'failed')

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = self._get_connection()

                # Build dynamic update query
                updates = []
                params = []

                if file_size_mb is not None:
                    updates.append("file_size_mb = ?")
                    params.append(file_size_mb)

                if duration_seconds is not None:
                    updates.append("duration_seconds = ?")
                    params.append(duration_seconds)

                if frame_count is not None:
                    updates.append("frame_count = ?")
                    params.append(frame_count)

                if end_timestep is not None:
                    updates.append("end_timestep = ?")
                    params.append(end_timestep)

                if end_time is not None:
                    updates.append("end_time = ?")
                    params.append(end_time)

                if status is not None:
                    updates.append("status = ?")
                    params.append(status)

                if not updates:
                    return True  # Nothing to update

                updates.append("updated_at = CURRENT_TIMESTAMP")

                query = f"""
                    UPDATE training_videos
                    SET {', '.join(updates)}
                    WHERE run_id = ? AND segment_number = ?
                """
                params.extend([run_id, segment_number])

                conn.execute(query, params)
                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"Failed to update training video segment: {e}")
            return False

    def get_training_videos(
        self,
        run_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training video segments with optional filtering.

        Args:
            run_id: Optional run ID filter
            status: Optional status filter

        Returns:
            List of training video segment dictionaries
        """
        try:
            with self._lock:
                conn = self._get_connection()

                query = "SELECT * FROM training_videos WHERE 1=1"
                params = []

                if run_id:
                    query += " AND run_id = ?"
                    params.append(run_id)

                if status:
                    query += " AND status = ?"
                    params.append(status)

                query += " ORDER BY run_id, segment_number"

                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                videos = []

                for row in cursor.fetchall():
                    video_dict = dict(zip(columns, row))
                    videos.append(video_dict)

                return videos

        except Exception as e:
            self.logger.error(f"Failed to get training videos: {e}")
            return []
