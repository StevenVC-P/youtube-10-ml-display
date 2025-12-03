"""
Verify Episode Tracking in Training Data

This script shows the actual episode counts from training data
to verify that episodes are being tracked correctly.
"""

from tools.retro_ml_desktop.ml_database import MetricsDatabase

def verify_episode_tracking(run_id: str = "run-54d8ae4e"):
    """Verify episode tracking in training data."""

    db = MetricsDatabase()
    
    print(f"\n{'='*70}")
    print(f"EPISODE TRACKING VERIFICATION - {run_id}")
    print(f"{'='*70}\n")
    
    # Get episode counts at each milestone
    milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Get total timesteps from config
    with db._lock:
        conn = db._get_connection()
        cursor = conn.execute(
            "SELECT config_json FROM experiment_runs WHERE run_id = ?",
            (run_id,)
        )
        row = cursor.fetchone()
        
        if row and row['config_json']:
            import json
            config_data = json.loads(row['config_json'])
            total_timesteps = config_data.get('total_timesteps', 0)
            
            print(f"Total Timesteps: {total_timesteps:,}\n")
            print(f"{'Milestone':<12} {'Timestep':<15} {'Episodes':<15} {'Avg Reward':<15}")
            print(f"{'-'*70}")
            
            for pct in milestones:
                checkpoint_timestep = int(total_timesteps * (pct / 100.0))
                
                # Get episode count at this timestep
                cursor = conn.execute("""
                    SELECT episode_count, episode_reward_mean
                    FROM training_metrics
                    WHERE run_id = ? AND timestep <= ? AND episode_count > 0
                    ORDER BY timestep DESC
                    LIMIT 1
                """, (run_id, checkpoint_timestep))
                
                row = cursor.fetchone()
                
                if row:
                    episode_count = row['episode_count'] or 0
                    avg_reward = row['episode_reward_mean'] or 0.0
                    
                    print(f"{pct}%{'':<9} {checkpoint_timestep:<15,} {episode_count:<15,} {avg_reward:<15.2f}")
                else:
                    print(f"{pct}%{'':<9} {checkpoint_timestep:<15,} {'N/A':<15} {'N/A':<15}")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print(f"{'='*70}")
    print("- 'Episodes' shows the TOTAL number of episodes completed during training")
    print("- With 16 parallel environments, episodes complete frequently")
    print("- The video now shows BOTH:")
    print("  1. Training episodes completed (from database)")
    print("  2. Current playback episode (counting during video playback)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    verify_episode_tracking()

