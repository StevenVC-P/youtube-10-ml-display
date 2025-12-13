#!/usr/bin/env python3
"""
Automated training script to run multiple Space Invaders training sessions.
This script will:
1. Start a fresh 5-minute Space Invaders training
2. Wait for completion and verify video
3. Start a continuation run (5 more minutes)
4. Start a parallel fresh run (5 minutes)
"""

import sys
import time
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.retro_ml_desktop.process_manager import ProcessManager, ResourceLimits
from tools.retro_ml_desktop.ml_database import MetricsDatabase
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_process_info(process_manager, run_id):
    """Get process info for a specific run_id."""
    processes = process_manager.get_processes()
    for proc in processes:
        if proc.id == run_id:
            return proc
    return None


def wait_for_completion(process_manager, run_id, max_wait_minutes=30):
    """Wait for a training run to complete."""
    logger.info(f"Waiting for run {run_id} to complete (max {max_wait_minutes} minutes)...")
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    last_log_time = 0

    while time.time() - start_time < max_wait_seconds:
        proc_info = get_process_info(process_manager, run_id)

        if proc_info is None:
            logger.error(f"Run {run_id} not found!")
            return False

        if proc_info.status == 'completed':
            logger.info(f"[OK] Run {run_id} completed successfully!")
            return True
        elif proc_info.status == 'failed':
            logger.error(f"[ERROR] Run {run_id} failed!")
            return False

        # Log progress every 30 seconds
        current_time = time.time()
        if current_time - last_log_time >= 30:
            logger.info(f"Run {run_id}: status={proc_info.status}")
            last_log_time = current_time

        time.sleep(5)

    logger.error(f"[TIMEOUT] Run {run_id} timed out after {max_wait_minutes} minutes")
    return False


def verify_video_exists(run_id, video_dir='video/milestones'):
    """Check if video was created for the run."""
    video_path = Path(video_dir)

    # Look for any video files containing the run_id
    videos = list(video_path.glob(f'*{run_id}*.mp4'))

    if videos:
        logger.info(f"[OK] Found {len(videos)} video(s) for run {run_id}:")
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            logger.info(f"   - {video.name} ({size_mb:.2f} MB)")
        return True
    else:
        logger.warning(f"[WARNING] No video found for run {run_id}")
        return False


def main():
    logger.info("=" * 80)
    logger.info("Starting Automated Training Runs")
    logger.info("=" * 80)

    # Initialize managers with correct database path
    from tools.retro_ml_desktop.config_manager import ConfigManager
    config_manager = ConfigManager(project_root)
    db_path = str(config_manager.get_database_path())
    db = MetricsDatabase(db_path)
    process_manager = ProcessManager(project_root=str(project_root))
    process_manager.set_database(db)  # Phase 1: Initialize Experiment Manager
    
    # Configuration for 5-minute training
    game = 'ALE/SpaceInvaders-v5'  # Use ALE prefix for Atari environments
    algorithm = 'ppo'
    timesteps_5min = int(0.0833 * 1000000)  # 5 min = 0.0833 hours × 1M
    vec_envs = 1  # Use 1 environment (multiprocessing is failing with BrokenPipeError on Python 3.13)

    resources = ResourceLimits(
        cpu_affinity=list(range(12)),
        memory_limit_gb=16,
        priority='normal',
        gpu_id=None
    )
    
    # ========================================================================
    # RUN 1: Fresh Space Invaders, 5 minutes
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RUN 1: Starting fresh Space Invaders training (5 minutes)")
    logger.info("=" * 80)
    
    try:
        run1_id = process_manager.create_process(
            game=game,
            algorithm=algorithm,
            total_timesteps=timesteps_5min,
            vec_envs=vec_envs,
            save_freq=10000,
            resources=resources
        )
        logger.info(f"[OK] Run 1 started with ID: {run1_id}")

        # Wait for Run 1 to complete
        if not wait_for_completion(process_manager, run1_id, max_wait_minutes=15):
            logger.error("Run 1 failed to complete. Aborting.")
            return 1

        # Verify video
        verify_video_exists(run1_id)

    except Exception as e:
        logger.error(f"[ERROR] Failed to start Run 1: {e}")
        return 1

    # Small delay before starting next runs
    logger.info("\nWaiting 10 seconds before starting next runs...")
    time.sleep(10)

    # ========================================================================
    # RUN 2: Continue from Run 1, 5 more minutes
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RUN 2: Continuing from Run 1 (5 more minutes)")
    logger.info("=" * 80)
    
    try:
        # Get the checkpoint path from Run 1
        checkpoint_path = f"models/checkpoints/{run1_id}/latest.zip"

        if not Path(checkpoint_path).exists():
            logger.error(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            logger.info("Attempting to find any checkpoint for Run 1...")
            checkpoint_dir = Path(f"models/checkpoints/{run1_id}")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.zip"))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    logger.info(f"Found checkpoint: {checkpoint_path}")

        run2_id = process_manager.create_process(
            game=game,
            algorithm=algorithm,
            total_timesteps=timesteps_5min,
            vec_envs=vec_envs,
            save_freq=10000,
            resources=resources,
            resume_from_checkpoint=checkpoint_path
        )
        logger.info(f"[OK] Run 2 started with ID: {run2_id} (continuing from {run1_id})")

    except Exception as e:
        logger.error(f"[ERROR] Failed to start Run 2: {e}")
        logger.info("Continuing with Run 3...")

    # ========================================================================
    # RUN 3: Fresh Space Invaders, 5 minutes (parallel)
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RUN 3: Starting fresh Space Invaders training (5 minutes, parallel)")
    logger.info("=" * 80)
    
    try:
        run3_id = process_manager.create_process(
            game=game,
            algorithm=algorithm,
            total_timesteps=timesteps_5min,
            vec_envs=vec_envs,
            save_freq=10000,
            resources=resources
        )
        logger.info(f"[OK] Run 3 started with ID: {run3_id}")

    except Exception as e:
        logger.error(f"[ERROR] Failed to start Run 3: {e}")
        return 1

    # ========================================================================
    # Wait for both Run 2 and Run 3 to complete
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("Waiting for Run 2 and Run 3 to complete...")
    logger.info("=" * 80)
    
    # Monitor both runs
    run2_complete = False
    run3_complete = False
    start_time = time.time()
    max_wait = 15 * 60  # 15 minutes
    
    while time.time() - start_time < max_wait:
        if not run2_complete:
            proc2 = get_process_info(process_manager, run2_id)
            if proc2 and proc2.status in ['completed', 'failed']:
                run2_complete = True
                logger.info(f"[OK] Run 2 finished with status: {proc2.status}")
                verify_video_exists(run2_id)

        if not run3_complete:
            proc3 = get_process_info(process_manager, run3_id)
            if proc3 and proc3.status in ['completed', 'failed']:
                run3_complete = True
                logger.info(f"[OK] Run 3 finished with status: {proc3.status}")
                verify_video_exists(run3_id)

        if run2_complete and run3_complete:
            break

        time.sleep(10)

    # ========================================================================
    # Final Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n[OK] Run 1 (Fresh, 5 min): {run1_id}")
    verify_video_exists(run1_id)

    logger.info(f"\n[OK] Run 2 (Continue, 5 min): {run2_id}")
    verify_video_exists(run2_id)

    logger.info(f"\n[OK] Run 3 (Fresh, 5 min): {run3_id}")
    verify_video_exists(run3_id)

    logger.info("\n" + "=" * 80)
    logger.info("All training runs completed!")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

