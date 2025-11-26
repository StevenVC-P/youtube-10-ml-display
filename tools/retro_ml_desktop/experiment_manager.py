"""
Experiment Manager - CRUD Operations for ML Experiments

Manages the full lifecycle of training experiments from creation to completion.
Provides experiment-centric architecture with lineage tracking and artifact management.

Usage:
    manager = ExperimentManager(database)

    # Create new experiment
    experiment = manager.create_experiment(
        name="Breakout Epic Training",
        game="BreakoutNoFrameskip-v4",
        algorithm="PPO",
        preset="epic"
    )

    # Update experiment status
    manager.update_experiment_status(experiment.id, "running")

    # Link video artifact
    manager.add_video_artifact(experiment.id, video_path, metadata)
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full hyperparameter configuration for an experiment."""

    # Core training parameters
    algorithm: str  # 'PPO' or 'DQN'
    total_timesteps: int
    learning_rate: float
    n_steps: int  # PPO: steps per update
    batch_size: int
    gamma: float  # Discount factor
    gae_lambda: float  # PPO: GAE lambda
    ent_coef: float  # Entropy coefficient
    vf_coef: float  # Value function coefficient
    max_grad_norm: float

    # Video generation
    video_length_hours: float
    milestone_percentages: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    clip_seconds: int = 90  # Length of milestone clips

    # Environment
    n_envs: int = 8  # Number of parallel environments
    frame_stack: int = 4

    # Model architecture
    policy: str = "CnnPolicy"
    net_arch: Optional[List] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def from_preset(cls, preset: str, video_length_hours: float = 4.0) -> 'ExperimentConfig':
        """
        Create config from preset name.

        Args:
            preset: 'quick' (30m), 'standard' (1-4h), 'epic' (10h)
            video_length_hours: Target video length

        Returns:
            ExperimentConfig instance
        """
        # Calculate timesteps based on video length
        # Assuming ~30 FPS, 8 envs, and want real-time video
        seconds = video_length_hours * 3600
        timesteps = int(seconds * 30 * 8)  # FPS * envs

        if preset == "quick":
            return cls(
                algorithm="PPO",
                total_timesteps=min(timesteps, 1_000_000),  # Cap at 1M for quick
                learning_rate=3e-4,
                n_steps=128,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                video_length_hours=video_length_hours,
                n_envs=8
            )
        elif preset == "standard":
            return cls(
                algorithm="PPO",
                total_timesteps=timesteps,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                video_length_hours=video_length_hours,
                n_envs=8
            )
        elif preset == "epic":
            return cls(
                algorithm="PPO",
                total_timesteps=timesteps,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                video_length_hours=video_length_hours,
                n_envs=8
            )
        else:
            raise ValueError(f"Unknown preset: {preset}")


@dataclass
class ExperimentLineage:
    """Tracks experiment lineage for continued training."""

    parent_experiment_id: Optional[str] = None
    checkpoint_source: Optional[str] = None
    generation: int = 1  # 1 = new, 2+ = continued from parent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentLineage':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Experiment:
    """
    Represents a single ML training experiment.

    An experiment is the core unit of organization - it represents one training session
    with all associated configuration, metadata, and artifacts.
    """

    id: str
    name: str
    game: str
    algorithm: str
    preset: str
    config: ExperimentConfig
    lineage: ExperimentLineage
    status: str = "pending"  # pending, running, completed, failed, paused
    created: datetime = field(default_factory=datetime.now)
    started: Optional[datetime] = None
    completed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Computed fields
    progress_pct: float = 0.0
    current_timestep: int = 0
    elapsed_time: float = 0.0  # seconds
    estimated_time_remaining: float = 0.0  # seconds

    # Metrics
    latest_metrics: Dict[str, Any] = field(default_factory=dict)
    final_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary for storage."""
        return {
            'id': self.id,
            'name': self.name,
            'game': self.game,
            'algorithm': self.algorithm,
            'preset': self.preset,
            'config': self.config.to_dict(),
            'lineage': self.lineage.to_dict(),
            'status': self.status,
            'created': self.created.isoformat(),
            'started': self.started.isoformat() if self.started else None,
            'completed': self.completed.isoformat() if self.completed else None,
            'tags': self.tags,
            'notes': self.notes,
            'progress_pct': self.progress_pct,
            'current_timestep': self.current_timestep,
            'elapsed_time': self.elapsed_time,
            'estimated_time_remaining': self.estimated_time_remaining,
            'latest_metrics': self.latest_metrics,
            'final_metrics': self.final_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experiment':
        """Create experiment from dictionary."""
        # Parse config
        config = ExperimentConfig.from_dict(data['config'])

        # Parse lineage
        lineage = ExperimentLineage.from_dict(data['lineage'])

        # Parse dates
        created = datetime.fromisoformat(data['created'])
        started = datetime.fromisoformat(data['started']) if data.get('started') else None
        completed = datetime.fromisoformat(data['completed']) if data.get('completed') else None

        return cls(
            id=data['id'],
            name=data['name'],
            game=data['game'],
            algorithm=data['algorithm'],
            preset=data['preset'],
            config=config,
            lineage=lineage,
            status=data['status'],
            created=created,
            started=started,
            completed=completed,
            tags=data.get('tags', []),
            notes=data.get('notes', ''),
            progress_pct=data.get('progress_pct', 0.0),
            current_timestep=data.get('current_timestep', 0),
            elapsed_time=data.get('elapsed_time', 0.0),
            estimated_time_remaining=data.get('estimated_time_remaining', 0.0),
            latest_metrics=data.get('latest_metrics', {}),
            final_metrics=data.get('final_metrics', {})
        )


class ExperimentManager:
    """Manages ML experiment lifecycle and persistence."""

    def __init__(self, database):
        """
        Initialize experiment manager.

        Args:
            database: MetricsDatabase instance
        """
        self.database = database
        self._ensure_tables()
        logger.info("ExperimentManager initialized")

    def _ensure_tables(self) -> None:
        """Ensure required database tables exist."""
        # This will be implemented when we update ml_database.py
        # For now, just log
        logger.debug("Checking experiment tables...")

    def create_experiment(
        self,
        name: str,
        game: str,
        algorithm: str = "PPO",
        preset: str = "standard",
        video_length_hours: float = 4.0,
        parent_experiment_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = ""
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Human-readable experiment name
            game: Game environment ID (e.g., "BreakoutNoFrameskip-v4")
            algorithm: "PPO" or "DQN"
            preset: "quick", "standard", or "epic"
            video_length_hours: Target video length
            parent_experiment_id: Optional parent for continued training
            tags: Optional list of tags
            notes: Optional notes

        Returns:
            Experiment instance
        """
        # Generate experiment ID
        from tools.retro_ml_desktop.process_manager import generate_run_id
        experiment_id = generate_run_id()

        # Create config from preset
        config = ExperimentConfig.from_preset(preset, video_length_hours)
        config.algorithm = algorithm

        # Create lineage
        lineage = ExperimentLineage()
        if parent_experiment_id:
            lineage.parent_experiment_id = parent_experiment_id
            # Get parent to determine generation
            parent = self.get_experiment(parent_experiment_id)
            if parent:
                lineage.generation = parent.lineage.generation + 1

        # Create experiment
        experiment = Experiment(
            id=experiment_id,
            name=name,
            game=game,
            algorithm=algorithm,
            preset=preset,
            config=config,
            lineage=lineage,
            status="pending",
            tags=tags or [],
            notes=notes
        )

        # Persist to database
        self._save_experiment(experiment)

        logger.info(f"Created experiment {experiment_id}: {name}")
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """
        Get experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment instance or None if not found
        """
        # This will be implemented when we update ml_database.py
        # For now, return None
        logger.debug(f"Getting experiment {experiment_id}")
        return None

    def update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status ('running', 'completed', 'failed', 'paused')
            metrics: Optional metrics to update

        Returns:
            True if updated successfully
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            logger.warning(f"Experiment {experiment_id} not found")
            return False

        experiment.status = status

        if status == "running" and not experiment.started:
            experiment.started = datetime.now()

        if status in ("completed", "failed"):
            experiment.completed = datetime.now()
            if metrics:
                experiment.final_metrics = metrics

        if metrics:
            experiment.latest_metrics = metrics

        self._save_experiment(experiment)
        logger.info(f"Updated experiment {experiment_id} status to {status}")
        return True

    def update_experiment_progress(
        self,
        experiment_id: str,
        progress_pct: float,
        current_timestep: int,
        elapsed_time: float,
        estimated_time_remaining: float,
        metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update experiment progress.

        Args:
            experiment_id: Experiment ID
            progress_pct: Progress percentage (0-100)
            current_timestep: Current training timestep
            elapsed_time: Elapsed time in seconds
            estimated_time_remaining: Estimated remaining time in seconds
            metrics: Optional latest metrics

        Returns:
            True if updated successfully
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        experiment.progress_pct = progress_pct
        experiment.current_timestep = current_timestep
        experiment.elapsed_time = elapsed_time
        experiment.estimated_time_remaining = estimated_time_remaining

        if metrics:
            experiment.latest_metrics = metrics

        self._save_experiment(experiment)
        return True

    def list_experiments(
        self,
        status: Optional[str] = None,
        game: Optional[str] = None,
        limit: int = 10
    ) -> List[Experiment]:
        """
        List experiments with optional filtering.

        Args:
            status: Optional status filter
            game: Optional game filter
            limit: Maximum number of experiments to return

        Returns:
            List of Experiment instances (newest first)
        """
        # This will be implemented when we update ml_database.py
        logger.debug(f"Listing experiments (status={status}, game={game}, limit={limit})")
        return []

    def get_recent_experiments(self, limit: int = 5) -> List[Experiment]:
        """
        Get most recent experiments.

        Args:
            limit: Maximum number to return

        Returns:
            List of Experiment instances (newest first)
        """
        return self.list_experiments(limit=limit)

    def get_active_experiments(self) -> List[Experiment]:
        """
        Get currently active (running) experiments.

        Returns:
            List of running Experiment instances
        """
        return self.list_experiments(status="running")

    def _save_experiment(self, experiment: Experiment) -> None:
        """
        Save experiment to database.

        Args:
            experiment: Experiment to save
        """
        # This will be implemented when we update ml_database.py
        logger.debug(f"Saving experiment {experiment.id}")

    def delete_experiment(self, experiment_id: str) -> bool:
        """
        Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting experiment {experiment_id}")
        # This will be implemented when we update ml_database.py
        return False
