#!/usr/bin/env python3
"""
Milestone Checkpoint Callback

Saves model checkpoints at milestone percentages WITHOUT recording videos.
This allows training to complete uninterrupted, and videos can be generated
post-training using the saved checkpoints.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Set

from stable_baselines3.common.callbacks import BaseCallback


class MilestoneCheckpointCallback(BaseCallback):
    """
    Callback that saves model checkpoints at milestone percentages.
    
    This replaces video recording during training to prevent blocking.
    Videos can be generated post-training using PostTrainingVideoGenerator.
    """

    def __init__(
        self,
        save_path: str,
        total_timesteps: int,
        milestones_pct: List[float] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.total_timesteps = total_timesteps
        self.milestones_pct = set(milestones_pct)
        self.completed_milestones: Set[float] = set()

        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)

        if self.verbose >= 1:
            print(f"[Checkpoint] MilestoneCheckpointCallback initialized:")
            print(f"  - Save path: {self.save_path}")
            print(f"  - Total timesteps: {self.total_timesteps:,}")
            print(f"  - Milestones: {sorted(self.milestones_pct)}")
            if self.total_timesteps > 0:
                milestone_steps = [int((pct / 100) * self.total_timesteps) for pct in sorted(self.milestones_pct)]
                print(f"  - Milestone steps: {[f'{step:,}' for step in milestone_steps]}")

    def _on_step(self) -> bool:
        """Check if we've reached a milestone and save checkpoint."""
        current_step = self.num_timesteps

        # Safety check for division by zero
        if self.total_timesteps == 0:
            return True  # Skip if we can't determine total timesteps

        current_pct = (current_step / self.total_timesteps) * 100
        
        # Check if we've reached a new milestone
        for milestone_pct in self.milestones_pct:
            if (current_pct >= milestone_pct and 
                milestone_pct not in self.completed_milestones):
                
                if self.verbose >= 1:
                    print(f"[Milestone] Milestone reached: {milestone_pct}% at step {current_step:,}")
                
                # Save checkpoint
                self._save_milestone_checkpoint(milestone_pct, current_step)
                self.completed_milestones.add(milestone_pct)
        
        return True

    def _save_milestone_checkpoint(self, milestone_pct: float, step: int):
        """Save model checkpoint at milestone."""
        try:
            # Create checkpoint filename
            checkpoint_filename = f"checkpoint_step_{step:08d}_pct_{milestone_pct:.0f}.zip"
            checkpoint_path = self.save_path / checkpoint_filename
            
            if self.verbose >= 1:
                print(f"[Checkpoint] Saving milestone checkpoint: {checkpoint_filename}")
            
            # Save model
            self.model.save(str(checkpoint_path))
            
            if self.verbose >= 2:
                print(f"[Checkpoint] ✅ Checkpoint saved: {checkpoint_path}")
                
        except Exception as e:
            print(f"[Checkpoint] ❌ Error saving checkpoint: {e}")

    def _on_training_end(self) -> None:
        """Save final model when training completes."""
        try:
            final_model_path = self.save_path / "final_model.zip"
            self.model.save(str(final_model_path))
            
            if self.verbose >= 1:
                print(f"[Checkpoint] ✅ Final model saved: {final_model_path}")
                
        except Exception as e:
            print(f"[Checkpoint] ❌ Error saving final model: {e}")
