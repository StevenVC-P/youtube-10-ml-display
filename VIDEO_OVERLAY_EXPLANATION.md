# Video Overlay Explanation

## Issue
Time-lapse and post-processed videos created from training footage don't show the neural network visualization overlays (stats, network diagram, etc.) that appear in the milestone videos.

## Root Cause

There are **two types of videos** in the system:

### 1. **Training Videos** (Raw Footage)
- **Location**: `video/training/{run_id}/`
- **Created by**: `RecordVideo` wrapper during training
- **Content**: Raw game footage only (no overlays)
- **Purpose**: Capture actual training gameplay with minimal performance impact
- **Example**: `env_0-episode-0.mp4`, `env_0-episode-1.mp4`, etc.

### 2. **Milestone Videos** (With Overlays)
- **Location**: `video/milestones/`
- **Created by**: `PostTrainingVideoGenerator` after training completes
- **Content**: Game footage + neural network visualization + stats overlay
- **Purpose**: Show AI learning progress with detailed analytics
- **Example**: `step_00006000_pct_10_analytics.mp4`

**The overlay includes:**
- Title: "BREAKOUT - PPO Neural Activity Viewer"
- Training progress: "Post-Training Evaluation | Progress: 0.04% | lr=2.5e-04"
- Frame count, Progress %, Episode #, Ep Reward
- Policy Distribution (action probabilities)
- Value Estimate
- Neural network diagram with real layer activations

## Why Training Videos Don't Have Overlays

1. **Performance**: Adding overlays during training would significantly slow down training (20-30% overhead)
2. **Model Access**: Overlays require access to the model's internal state (layer activations, policy distribution, value estimates)
3. **Design**: Training videos are meant to be lightweight captures; overlays are added post-training

## Current Behavior

When you create a time-lapse or progression video using the post-processing tools:

- **Input**: Raw training videos from `video/training/{run_id}/`
- **Output**: Processed video with raw footage (no overlays)
- **Result**: You see the game being played, but no neural network visualization

## Solutions

### Option 1: Use Milestone Videos (Recommended)

**For progression videos**, the system already uses milestone videos which have overlays:

```python
# In create_milestone_progression_video()
# This already uses milestone videos with overlays ✅
video_file = self.video_training_dir / run_id / f"env_0-episode-{milestone_episode}.mp4"
```

**For time-lapses**, you can create a time-lapse from milestone videos instead of training videos.

### Option 2: Add Overlay Support to Post-Processing (Complex)

This would require:
1. Loading the trained model for the run
2. Re-running the model on each frame to extract analytics
3. Applying overlays using the same code as `PostTrainingVideoGenerator`

**Challenges:**
- Requires model checkpoint access
- Computationally expensive (need to run inference on every frame)
- Complex implementation

### Option 3: Record Training Videos with Overlays (Not Recommended)

This would require:
1. Modifying the `RecordVideo` wrapper to add overlays during training
2. Significant performance impact (20-30% slower training)
3. Increased storage requirements (larger video files)

## Recommended Approach

**For now, document the limitation and provide two workflows:**

### Workflow A: Quick Time-Lapse (No Overlays)
1. Use training videos from `video/training/{run_id}/`
2. Create time-lapse with raw footage
3. Fast, but no neural network visualization

### Workflow B: Detailed Progression (With Overlays)
1. Use milestone videos from `video/milestones/`
2. Create progression video showing AI improvement
3. Includes full neural network visualization

## Future Enhancement

Add a checkbox in the UI:
- ☐ **Include neural network overlays** (slower, requires model)
- ☑ **Raw footage only** (faster, default)

When "Include neural network overlays" is checked:
1. Load the model checkpoint for the run
2. Re-process each frame with overlay generation
3. Create enhanced time-lapse with full visualization

## Summary

**Current State:**
- ✅ Milestone videos have overlays
- ✅ Progression videos use milestone videos (have overlays)
- ❌ Time-lapse videos use training videos (no overlays)
- ❌ Training videos are raw footage only

**User Expectation:**
- Users expect time-lapse videos to show the same information as milestone videos

**Solution:**
- Document the two video types clearly
- Provide option to create time-lapses from milestone videos
- Consider adding overlay support as a future enhancement

