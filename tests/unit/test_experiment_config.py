"""
Unit tests for ExperimentConfig

Tests cover:
- Validation of required fields
- Default value handling
- Path generation
- Serialization/deserialization
- Run type to timesteps mapping
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from retro_ml.core.experiments.config import ExperimentConfig, RunType


class TestExperimentConfig:
    """Test suite for ExperimentConfig"""
    
    def test_initialization_with_minimal_params(self):
        """Test that config initializes with only required params"""
        config = ExperimentConfig(game_id="BreakoutNoFrameskip-v4")
        
        assert config.game_id == "BreakoutNoFrameskip-v4"
        assert config.algorithm == "PPO"
        assert config.run_type == RunType.QUICK
        assert config.experiment_id is not None
        assert len(config.experiment_id) > 0
    
    def test_initialization_with_all_params(self):
        """Test that config initializes with all params"""
        config = ExperimentConfig(
            game_id="PongNoFrameskip-v4",
            algorithm="A2C",
            seed=42,
            total_timesteps=100000,
            run_type=RunType.SHORT,
            output_dir=Path("D:/test"),
        )
        
        assert config.game_id == "PongNoFrameskip-v4"
        assert config.algorithm == "A2C"
        assert config.seed == 42
        assert config.total_timesteps == 100000
        assert config.run_type == RunType.SHORT
        assert config.output_dir == Path("D:/test")
    
    def test_unique_experiment_ids(self):
        """Test that each config gets a unique experiment ID"""
        config1 = ExperimentConfig(game_id="BreakoutNoFrameskip-v4")
        config2 = ExperimentConfig(game_id="BreakoutNoFrameskip-v4")
        
        assert config1.experiment_id != config2.experiment_id
    
    def test_invalid_game_id_raises_error(self):
        """Test that empty game_id raises validation error"""
        with pytest.raises(ValidationError):
            ExperimentConfig(game_id="")
    
    def test_invalid_algorithm_raises_error(self):
        """Test that invalid algorithm raises validation error"""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                game_id="BreakoutNoFrameskip-v4",
                algorithm="INVALID"
            )
    
    def test_algorithm_case_insensitive(self):
        """Test that algorithm is case-insensitive"""
        config = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            algorithm="ppo"
        )
        assert config.algorithm == "PPO"
    
    def test_negative_timesteps_raises_error(self):
        """Test that negative timesteps raises validation error"""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                game_id="BreakoutNoFrameskip-v4",
                total_timesteps=-1000
            )
    
    def test_zero_timesteps_raises_error(self):
        """Test that zero timesteps raises validation error"""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                game_id="BreakoutNoFrameskip-v4",
                total_timesteps=0
            )
    
    def test_run_type_sets_timesteps(self):
        """Test that run_type automatically sets total_timesteps"""
        config_quick = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.QUICK
        )
        assert config_quick.total_timesteps == 10_000
        
        config_short = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.SHORT
        )
        assert config_short.total_timesteps == 500_000
        
        config_medium = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.MEDIUM
        )
        assert config_medium.total_timesteps == 2_000_000
        
        config_long = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.LONG
        )
        assert config_long.total_timesteps == 10_000_000
    
    def test_explicit_timesteps_override_run_type(self):
        """Test that explicit timesteps override run_type"""
        config = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            run_type=RunType.SHORT,
            total_timesteps=999999
        )
        assert config.total_timesteps == 999999
    
    def test_experiment_dir_property(self):
        """Test that experiment_dir property works correctly"""
        config = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            output_dir=Path("D:/test")
        )
        expected = Path("D:/test") / config.experiment_id
        assert config.experiment_dir == expected
    
    def test_artifact_paths(self):
        """Test that artifact path properties work correctly"""
        config = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            output_dir=Path("D:/test")
        )
        
        exp_dir = config.experiment_dir
        assert config.metrics_path == exp_dir / "metrics.json"
        assert config.model_path == exp_dir / "model.zip"
        assert config.log_path == exp_dir / "training.log"
        assert config.video_path == exp_dir / "video.mp4"
    
    def test_serialization_to_json(self):
        """Test that config can be serialized to JSON"""
        config = ExperimentConfig(
            game_id="BreakoutNoFrameskip-v4",
            seed=42
        )
        
        json_str = config.model_dump_json()
        data = json.loads(json_str)
        
        assert data["game_id"] == "BreakoutNoFrameskip-v4"
        assert data["seed"] == 42
        assert data["algorithm"] == "PPO"
    
    def test_deserialization_from_dict(self):
        """Test that config can be created from dict"""
        data = {
            "game_id": "PongNoFrameskip-v4",
            "algorithm": "DQN",
            "seed": 123,
        }
        
        config = ExperimentConfig(**data)
        assert config.game_id == "PongNoFrameskip-v4"
        assert config.algorithm == "DQN"
        assert config.seed == 123

