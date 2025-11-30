"""
Training Engine - Core training orchestration

This module provides the main training loop and experiment execution.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from retro_ml.core.experiments.config import ExperimentConfig
from retro_ml.core.metrics.event_bus import MetricEventBus, MetricEvent, EventType
from retro_ml.core.artifacts.writers import (
    MetricsWriter,
    LogWriter,
    ModelWriter,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result of a training experiment"""
    experiment_id: str
    success: bool
    total_timesteps: int
    final_reward: Optional[float]
    training_time: float
    metrics_path: Path
    model_path: Path
    log_path: Path
    error: Optional[str] = None


class TrainingEngine:
    """
    Core training engine that orchestrates the training process.
    
    This class handles:
    - Environment creation
    - Model initialization
    - Training loop execution
    - Metric collection via event bus
    - Artifact generation
    """
    
    def __init__(self, config: ExperimentConfig, event_bus: Optional[MetricEventBus] = None):
        """
        Initialize training engine.
        
        Args:
            config: Experiment configuration
            event_bus: Optional event bus for metrics (creates new one if None)
        """
        self.config = config
        self.event_bus = event_bus or MetricEventBus()
        
        # Set up logging
        self._setup_logging()
        
        # Initialize artifact writers
        self.metrics_writer = MetricsWriter(config, self.event_bus)
        self.log_writer = LogWriter(config, self.event_bus)
        self.model_writer = ModelWriter(config)
        
        # Training state
        self.model: Optional[PPO] = None
        self.env: Optional[gym.Env] = None
        
    def _setup_logging(self) -> None:
        """Set up logging for this experiment"""
        # Create output directory
        self.config.create_output_dirs()
        
        # Configure logger
        log_handler = logging.FileHandler(self.config.log_path)
        log_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(log_handler)
        logger.setLevel(logging.INFO)
        
    def _create_env(self) -> gym.Env:
        """Create and wrap the training environment"""
        logger.info(f"Creating environment: {self.config.game_id}")
        
        def make_env():
            env = gym.make(self.config.game_id)
            return env
        
        # Create vectorized environment
        env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
        
        # Add frame stacking (common for Atari)
        env = VecFrameStack(env, n_stack=4)
        
        return env
    
    def _create_model(self) -> PPO:
        """Create the RL model"""
        logger.info(f"Creating {self.config.algorithm} model")
        
        # Determine device
        device = "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Create model
        model = PPO(
            "CnnPolicy",
            self.env,
            learning_rate=self.config.learning_rate,
            n_steps=128,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=self.config.seed,
            device=device,
            verbose=1,
        )
        
        return model
    
    def run(self) -> ExperimentResult:
        """
        Run the training experiment.
        
        Returns:
            ExperimentResult with training outcomes
        """
        start_time = time.time()
        
        try:
            # Emit start event
            self.event_bus.emit(MetricEvent(
                event_type=EventType.RUN_STARTED,
                experiment_id=self.config.experiment_id,
                data={"config": self.config.model_dump()}
            ))
            
            logger.info(f"Starting experiment {self.config.experiment_id}")
            logger.info(f"Game: {self.config.game_id}")
            logger.info(f"Algorithm: {self.config.algorithm}")
            logger.info(f"Total timesteps: {self.config.total_timesteps}")
            
            # Create environment and model
            self.env = self._create_env()
            self.model = self._create_model()
            
            # Train the model
            logger.info("Starting training...")
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                progress_bar=True,
            )

            logger.info("Training completed successfully")

            # Save model
            logger.info(f"Saving model to {self.config.model_path}")
            self.model_writer.save_model(self.model)

            # Calculate final metrics
            training_time = time.time() - start_time

            # Emit completion event
            self.event_bus.emit(MetricEvent(
                event_type=EventType.RUN_COMPLETED,
                experiment_id=self.config.experiment_id,
                data={
                    "total_timesteps": self.config.total_timesteps,
                    "training_time": training_time,
                }
            ))

            # Finalize artifact writers
            self.metrics_writer.finalize()
            self.log_writer.finalize()

            logger.info(f"Experiment completed in {training_time:.2f} seconds")

            return ExperimentResult(
                experiment_id=self.config.experiment_id,
                success=True,
                total_timesteps=self.config.total_timesteps,
                final_reward=None,  # TODO: Calculate from metrics
                training_time=training_time,
                metrics_path=self.config.metrics_path,
                model_path=self.config.model_path,
                log_path=self.config.log_path,
            )

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            training_time = time.time() - start_time

            # Emit error event
            self.event_bus.emit(MetricEvent(
                event_type=EventType.ERROR,
                experiment_id=self.config.experiment_id,
                data={"error": str(e)}
            ))

            return ExperimentResult(
                experiment_id=self.config.experiment_id,
                success=False,
                total_timesteps=0,
                final_reward=None,
                training_time=training_time,
                metrics_path=self.config.metrics_path,
                model_path=self.config.model_path,
                log_path=self.config.log_path,
                error=str(e),
            )

        finally:
            # Clean up
            if self.env is not None:
                self.env.close()


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Convenience function to run a training experiment.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResult with training outcomes
    """
    engine = TrainingEngine(config)
    return engine.run()

