"""
Test environment factory for YouTube 10 ML Display project.
Verifies environment creation, observation shapes, and action spaces.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

# Register ALE environments with multiple approaches
def register_ale_environments():
    """Register ALE environments using multiple fallback methods"""
    try:
        # Method 1: Standard registration
        import ale_py
        gym.register_envs(ale_py)
        return True
    except ImportError:
        print("Warning: ale_py not available")
    except Exception as e:
        print(f"Warning: Standard ALE registration failed: {e}")

    try:
        # Method 2: Try importing gymnasium atari environments
        import gymnasium.envs.atari
        return True
    except Exception as e:
        print(f"Warning: Gymnasium atari import failed: {e}")

    return False

# Attempt registration
_ale_registered = register_ale_environments()

from conf.config import load_config
from envs.make_env import (
    make_eval_env,
    make_vec_env,
    get_env_info,
    validate_env_creation
)
from envs.atari_wrappers import (
    MaxAndSkipWrapper,
    GrayScaleWrapper,
    ResizeWrapper,
    FrameStackWrapper,
    apply_atari_wrappers,
    make_atari_env
)


class TestAtariWrappers:
    """Test individual Atari wrappers."""
    
    @pytest.fixture
    def base_env(self):
        """Create a base Atari environment for testing."""
        return gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config("conf/config.yaml")
    
    def test_max_and_skip_wrapper(self, base_env):
        """Test MaxAndSkipWrapper functionality."""
        wrapped_env = MaxAndSkipWrapper(base_env, skip=4)
        
        obs, _ = wrapped_env.reset()
        assert obs.shape == base_env.observation_space.shape
        
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        
        wrapped_env.close()
    
    def test_grayscale_wrapper(self, base_env):
        """Test GrayScaleWrapper functionality."""
        wrapped_env = GrayScaleWrapper(base_env, keep_dim=True)
        
        obs, _ = wrapped_env.reset()
        # Should convert from RGB (H,W,3) to grayscale (H,W,1)
        expected_shape = base_env.observation_space.shape[:2] + (1,)
        assert obs.shape == expected_shape
        assert obs.dtype == np.uint8
        
        wrapped_env.close()
    
    def test_resize_wrapper(self, base_env):
        """Test ResizeWrapper functionality."""
        target_size = (84, 84)
        wrapped_env = ResizeWrapper(base_env, size=target_size)
        
        obs, _ = wrapped_env.reset()
        # Should resize to target dimensions
        expected_shape = target_size + (base_env.observation_space.shape[2],)
        assert obs.shape == expected_shape
        assert obs.dtype == np.uint8
        
        wrapped_env.close()
    
    def test_frame_stack_wrapper(self, base_env):
        """Test FrameStackWrapper functionality."""
        num_stack = 4
        wrapped_env = FrameStackWrapper(base_env, num_stack=num_stack)
        
        obs, _ = wrapped_env.reset()
        # Should stack frames along last axis
        expected_shape = base_env.observation_space.shape + (num_stack,)
        assert obs.shape == expected_shape
        assert obs.dtype == np.uint8
        
        # Test that frames are properly stacked after steps
        action = wrapped_env.action_space.sample()
        obs2, _, _, _, _ = wrapped_env.step(action)
        assert obs2.shape == expected_shape
        
        wrapped_env.close()
    
    def test_full_wrapper_pipeline(self, config):
        """Test the complete Atari wrapper pipeline."""
        env_id = config['game']['env_id']
        base_env = gym.make(env_id, render_mode="rgb_array")
        
        # Apply Atari wrappers
        wrapped_env = apply_atari_wrappers(base_env, config)

        obs, _ = wrapped_env.reset()

        # Check final observation shape based on config
        game_config = config['game']
        expected_height, expected_width = game_config['resize']
        expected_channels = 1 if game_config['grayscale'] else 3
        expected_stack = game_config['frame_stack']

        expected_shape = (expected_height, expected_width, expected_channels, expected_stack)
        assert obs.shape == expected_shape
        assert obs.dtype == np.uint8
        
        # Test environment step
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        assert obs.shape == expected_shape
        
        wrapped_env.close()


class TestEnvironmentFactory:
    """Test environment factory functions."""
    
    @pytest.fixture
    def config(self):
        """Load test configuration."""
        return load_config("conf/config.yaml")
    
    def test_make_eval_env(self, config):
        """Test evaluation environment creation."""
        env = make_eval_env(config, seed=42)
        
        # Test environment properties
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)
        
        # Test environment functionality
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.uint8
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_make_vec_env(self, config):
        """Test vectorized environment creation."""
        n_envs = 2
        vec_env = make_vec_env(
            config, 
            seed=42, 
            n_envs=n_envs, 
            vec_env_cls=DummyVecEnv
        )
        
        # Test vectorized environment properties
        assert vec_env.num_envs == n_envs
        assert isinstance(vec_env.observation_space, spaces.Box)
        assert isinstance(vec_env.action_space, spaces.Discrete)
        
        # Test vectorized environment functionality
        obs = vec_env.reset()
        assert obs.shape[0] == n_envs
        assert obs.dtype == np.uint8
        
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        
        assert obs.shape[0] == n_envs
        assert len(rewards) == n_envs
        assert len(dones) == n_envs
        assert len(infos) == n_envs
        
        vec_env.close()
    
    def test_get_env_info(self, config):
        """Test environment info extraction."""
        env_info = get_env_info(config)
        
        # Check required info fields
        assert 'env_id' in env_info
        assert 'observation_space' in env_info
        assert 'action_space' in env_info
        assert 'observation_shape' in env_info
        assert 'n_actions' in env_info
        
        # Check info values
        assert env_info['env_id'] == config['game']['env_id']
        assert isinstance(env_info['observation_space'], spaces.Box)
        assert isinstance(env_info['action_space'], spaces.Discrete)
        assert isinstance(env_info['observation_shape'], tuple)
        assert isinstance(env_info['n_actions'], (int, np.integer))
        assert env_info['n_actions'] > 0
    
    def test_observation_shape_matches_config(self, config):
        """Test that observation shape matches configuration."""
        env = make_eval_env(config, seed=42)
        obs, _ = env.reset()

        game_config = config['game']
        expected_height, expected_width = game_config['resize']
        expected_channels = 1 if game_config['grayscale'] else 3
        expected_stack = game_config['frame_stack']
        expected_shape = (expected_height, expected_width, expected_channels, expected_stack)

        assert obs.shape == expected_shape, f"Expected {expected_shape}, got {obs.shape}"

        env.close()
    
    def test_action_space_is_discrete(self, config):
        """Test that action space is discrete for Atari games."""
        env = make_eval_env(config, seed=42)
        
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n > 0
        
        # Test that we can sample valid actions
        for _ in range(10):
            action = env.action_space.sample()
            assert 0 <= action < env.action_space.n
        
        env.close()
    
    def test_environment_determinism(self, config):
        """Test that environments are deterministic with same seed."""
        seed = 12345
        
        # Create two environments with same seed
        env1 = make_eval_env(config, seed=seed)
        env2 = make_eval_env(config, seed=seed)
        
        # Reset both environments
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Initial observations should be identical
        np.testing.assert_array_equal(obs1, obs2)
        
        # Take same actions and compare
        for _ in range(5):
            action = env1.action_space.sample()
            obs1, reward1, term1, trunc1, _ = env1.step(action)
            obs2, reward2, term2, trunc2, _ = env2.step(action)
            
            np.testing.assert_array_equal(obs1, obs2)
            assert reward1 == reward2
            assert term1 == term2
            assert trunc1 == trunc2
        
        env1.close()
        env2.close()
    
    def test_environment_creation_integration(self, config):
        """Test the integrated environment creation function."""
        success = validate_env_creation(config)
        assert success, "Environment creation test should pass"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
