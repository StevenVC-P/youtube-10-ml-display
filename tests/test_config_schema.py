"""
Test config schema validation for YouTube 10 ML Display project.
Ensures required keys exist and types are correct.
"""

import pytest
import yaml
from pathlib import Path
from conf.config import load_config


class TestConfigSchema:
    """Test suite for config schema validation."""
    
    @pytest.fixture
    def config(self):
        """Load the config for testing."""
        return load_config("conf/config.yaml")
    
    def test_config_file_exists(self):
        """Test that config file exists."""
        config_path = Path("conf/config.yaml")
        assert config_path.exists(), "Config file conf/config.yaml must exist"
    
    def test_config_is_valid_yaml(self):
        """Test that config file is valid YAML."""
        with open("conf/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        assert config is not None, "Config file must contain valid YAML"
    
    def test_required_top_level_keys(self, config):
        """Test that all required top-level keys are present."""
        required_keys = [
            'project_name', 'seed', 'paths', 'game', 
            'train', 'recording', 'stream', 'render'
        ]
        
        for key in required_keys:
            assert key in config, f"Required top-level key '{key}' is missing"
    
    def test_project_name_type(self, config):
        """Test project_name is a string."""
        assert isinstance(config['project_name'], str), "project_name must be a string"
        assert len(config['project_name']) > 0, "project_name cannot be empty"
    
    def test_seed_type(self, config):
        """Test seed is an integer."""
        assert isinstance(config['seed'], int), "seed must be an integer"
        assert config['seed'] >= 0, "seed must be non-negative"
    
    def test_paths_section(self, config):
        """Test paths section has required keys and correct types."""
        required_paths = [
            'videos_milestones', 'videos_eval', 'videos_parts',
            'logs_tb', 'models', 'ffmpeg_path'
        ]
        
        assert 'paths' in config, "paths section is required"
        paths = config['paths']
        
        for path_key in required_paths:
            assert path_key in paths, f"Required path key '{path_key}' is missing"
            assert isinstance(paths[path_key], str), f"Path '{path_key}' must be a string"
    
    def test_game_section(self, config):
        """Test game section has required keys and correct types."""
        required_game_keys = ['env_id', 'frame_stack', 'resize', 'grayscale', 'max_skip']
        
        assert 'game' in config, "game section is required"
        game = config['game']
        
        for key in required_game_keys:
            assert key in game, f"Required game key '{key}' is missing"
        
        # Type checks
        assert isinstance(game['env_id'], str), "env_id must be a string"
        assert isinstance(game['frame_stack'], int), "frame_stack must be an integer"
        assert isinstance(game['resize'], list), "resize must be a list"
        assert len(game['resize']) == 2, "resize must have exactly 2 elements"
        assert isinstance(game['grayscale'], bool), "grayscale must be a boolean"
        assert isinstance(game['max_skip'], int), "max_skip must be an integer"
    
    def test_train_section(self, config):
        """Test train section has required keys and correct types."""
        required_train_keys = [
            'algo', 'policy', 'total_timesteps', 'vec_envs', 'checkpoint_every_sec'
        ]
        
        assert 'train' in config, "train section is required"
        train = config['train']
        
        for key in required_train_keys:
            assert key in train, f"Required train key '{key}' is missing"
        
        # Type checks
        assert isinstance(train['algo'], str), "algo must be a string"
        assert isinstance(train['policy'], str), "policy must be a string"
        assert isinstance(train['total_timesteps'], int), "total_timesteps must be an integer"
        assert isinstance(train['vec_envs'], int), "vec_envs must be an integer"
        assert isinstance(train['checkpoint_every_sec'], int), "checkpoint_every_sec must be an integer"
        
        # Value checks
        assert train['total_timesteps'] > 0, "total_timesteps must be positive"
        assert train['vec_envs'] > 0, "vec_envs must be positive"
        assert train['checkpoint_every_sec'] > 0, "checkpoint_every_sec must be positive"
    
    def test_recording_section(self, config):
        """Test recording section has required keys and correct types."""
        required_recording_keys = [
            'fps', 'crf', 'milestone_clip_seconds', 'eval_clip_seconds',
            'milestones_pct', 'eval_every_steps', 'eval_episodes'
        ]
        
        assert 'recording' in config, "recording section is required"
        recording = config['recording']
        
        for key in required_recording_keys:
            assert key in recording, f"Required recording key '{key}' is missing"
        
        # Type checks
        assert isinstance(recording['fps'], int), "fps must be an integer"
        assert isinstance(recording['crf'], int), "crf must be an integer"
        assert isinstance(recording['milestones_pct'], list), "milestones_pct must be a list"
        assert isinstance(recording['eval_every_steps'], int), "eval_every_steps must be an integer"
        assert isinstance(recording['eval_episodes'], int), "eval_episodes must be an integer"
        
        # Value checks
        assert recording['fps'] > 0, "fps must be positive"
        assert 0 <= recording['crf'] <= 51, "crf must be between 0 and 51"
        assert len(recording['milestones_pct']) > 0, "milestones_pct cannot be empty"
    
    def test_stream_section(self, config):
        """Test stream section has required keys and correct types."""
        required_stream_keys = [
            'enabled', 'grid', 'pane_size', 'fps', 'overlay_hud',
            'checkpoint_poll_sec', 'save_mode', 'segment_seconds',
            'output_basename', 'preset', 'crf'
        ]
        
        assert 'stream' in config, "stream section is required"
        stream = config['stream']
        
        for key in required_stream_keys:
            assert key in stream, f"Required stream key '{key}' is missing"
        
        # Type checks
        assert isinstance(stream['enabled'], bool), "enabled must be a boolean"
        assert isinstance(stream['grid'], int), "grid must be an integer"
        assert isinstance(stream['pane_size'], list), "pane_size must be a list"
        assert len(stream['pane_size']) == 2, "pane_size must have exactly 2 elements"
        assert isinstance(stream['fps'], int), "fps must be an integer"
        assert isinstance(stream['overlay_hud'], bool), "overlay_hud must be a boolean"
        assert isinstance(stream['save_mode'], str), "save_mode must be a string"
        assert isinstance(stream['output_basename'], str), "output_basename must be a string"
        
        # Value checks
        assert stream['grid'] in [1, 4, 9], "grid must be 1, 4, or 9"
        assert stream['save_mode'] in ['single', 'segments'], "save_mode must be 'single' or 'segments'"
    
    def test_render_section(self, config):
        """Test render section has required keys and correct types."""
        required_render_keys = ['target_hours', 'music_path', 'add_titles']
        
        assert 'render' in config, "render section is required"
        render = config['render']
        
        for key in required_render_keys:
            assert key in render, f"Required render key '{key}' is missing"
        
        # Type checks
        assert isinstance(render['target_hours'], int), "target_hours must be an integer"
        assert isinstance(render['music_path'], str), "music_path must be a string"
        assert isinstance(render['add_titles'], bool), "add_titles must be a boolean"
        
        # Value checks
        assert render['target_hours'] > 0, "target_hours must be positive"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
