#!/usr/bin/env python3
"""Check what's available in environment factory"""

try:
    import envs.make_env as env_module
    
    print("Available functions in envs.make_env:")
    functions = [name for name in dir(env_module) if not name.startswith('_') and callable(getattr(env_module, name))]
    for func in functions:
        print(f"  - {func}")
    
    print("\nTesting basic environment creation:")
    from conf.config import load_config
    config = load_config('conf/config.yaml')
    
    if hasattr(env_module, 'make_eval_env'):
        env = env_module.make_eval_env(config, seed=42)
        obs, _ = env.reset()
        print(f"✅ Environment works! Obs shape: {obs.shape}")
        env.close()
    else:
        print("❌ make_eval_env not found")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()