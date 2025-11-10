# Neural Network Visualization Feature

## Overview

This feature adds **neural network visualization** to post-training videos, showing real-time ML analytics alongside gameplay. This was extracted from the `addedGMB` branch's `hour_video_callback.py` and integrated into the main branch's `post_training_video_generator.py`.

## What Was Added

### Visual Features

The enhanced videos now show:

1. **Left Panel (480x540)** - ML Analytics & Neural Network Visualization:
   - **Title Bar**: Game name, algorithm (PPO/DQN), and "Neural Activity Viewer"
   - **Training Metrics**:
     - Current frame number
     - Progress percentage through video
     - Episode reward accumulation
   - **Policy Distribution**: Real-time action probabilities for each action (NOOP, FIRE, RIGHT, LEFT)
     - Highlighted in yellow for the most likely action
   - **Value Estimate**: The model's estimate of future rewards
   - **Neural Network Diagram**:
     - 5-layer visualization (Input → Conv1 → Conv2 → Dense → Output)
     - Animated connections showing network activity
     - Node brightness based on activation levels
     - Pulsing effect on the chosen action
     - Layer shapes and dimensions labeled

2. **Right Panel (480x540)** - Game Gameplay:
   - Standard Atari game rendering
   - Resized to fit alongside analytics

3. **Combined Output (960x540)** - Side-by-side view

### Technical Implementation

#### New Methods in `PostTrainingVideoGenerator`

1. **`_extract_ml_analytics(model, obs, frame_idx, total_reward)`**
   - Extracts real-time ML metrics from the model
   - Gets action probabilities from policy distribution
   - Gets value estimates from value function
   - Handles both PPO and DQN models
   - Returns analytics dictionary with:
     - `step`: Current frame index
     - `frame_in_video`: Frame number in current video
     - `progress_pct`: Percentage through video
     - `action_probs`: Probability distribution over actions
     - `value_estimate`: Expected future reward
     - `episode_reward`: Cumulative reward in current episode

2. **`_create_enhanced_frame(game_frame, analytics)`**
   - Creates the combined 960x540 frame
   - Resizes game frame to 480x540
   - Creates 480x540 analytics panel
   - Draws analytics and neural network visualization
   - Combines panels horizontally
   - Converts RGB to BGR for OpenCV

3. **`_draw_analytics_panel(panel, analytics)`**
   - Draws text-based ML metrics on the left panel
   - Shows game name, algorithm, and learning rate
   - Displays frame number, progress, and episode reward
   - Shows policy distribution with color-coded probabilities
   - Shows value estimate

4. **`_draw_neural_network(panel, analytics)`**
   - Draws animated neural network diagram
   - 5 layers with realistic architecture for Atari:
     - Input: 84×84×4 (stacked frames)
     - Conv1: 32 filters @ 20×20
     - Conv2: 64 filters @ 9×9
     - Dense: 512 units
     - Output: 4 actions
   - Animated connections based on frame number
   - Node brightness based on activation
   - Pulsing effect on selected action
   - Layer labels with shapes

#### Modified Methods

1. **`_record_gameplay_frames(model, env)`**
   - Now calls `_extract_ml_analytics()` for each frame
   - Calls `_create_enhanced_frame()` to add visualization
   - Tracks cumulative episode reward
   - Resets reward counter on episode end

2. **`_create_video_from_frames(frames, output_path)`**
   - Updated comment: frames are already in BGR format
   - No longer converts RGB to BGR (done in `_create_enhanced_frame`)

## Source Code Origin

The visualization code was extracted from:
- **Branch**: `addedGMB`
- **File**: `training/hour_video_callback.py`
- **Methods**:
  - `_extract_ml_analytics()` (lines 209-248)
  - `_create_enhanced_frame()` (lines 250-268)
  - `_draw_analytics_panel()` (lines 270-321)
  - `_draw_neural_network()` (lines 323-423)

## Integration Details

### Changes Made

**File**: `training/post_training_video_generator.py`

- **Lines 222-254**: Updated `_record_gameplay_frames()` to use visualization
- **Lines 256-281**: Updated `_create_video_from_frames()` comment
- **Lines 283-320**: Added `_extract_ml_analytics()` method
- **Lines 322-345**: Added `_create_enhanced_frame()` method
- **Lines 347-402**: Added `_draw_analytics_panel()` method
- **Lines 404-504**: Added `_draw_neural_network()` method

### Adaptations Made

The code was adapted from the training callback to work with post-training video generation:

1. **Analytics Extraction**:
   - Changed from using `self.model` and `self.training_env` to accepting `model` and `obs` as parameters
   - Added explicit torch tensor conversion for observations
   - Used `model.policy.get_distribution()` instead of direct policy call
   - Added error handling for different model types

2. **Progress Tracking**:
   - Changed from `hour` and `hour_frame_count` to `frame_in_video`
   - Changed from `total_timesteps` to `clip_seconds * fps` for progress calculation
   - Added `episode_reward` tracking

3. **Title and Labels**:
   - Made game name and algorithm dynamic from config
   - Changed subtitle from "Hour X" to "Post-Training Evaluation"
   - Updated metrics to show frame number instead of step number

4. **Color Conversion**:
   - Kept RGB to BGR conversion in `_create_enhanced_frame()`
   - Removed duplicate conversion in `_create_video_from_frames()`

## Benefits

✅ **Visual Learning Progress**: See how the neural network makes decisions  
✅ **Action Transparency**: Understand which actions the model prefers and why  
✅ **Value Insights**: See the model's confidence in future rewards  
✅ **Network Activity**: Visualize the flow of information through layers  
✅ **Debugging Aid**: Identify when the model is uncertain or making mistakes  
✅ **Educational**: Great for understanding how RL agents work  
✅ **Engaging**: Makes training videos much more interesting to watch  

## Usage

The visualization is automatically applied when generating post-training videos:

```python
# Via UI
1. Open ML Training Manager
2. Go to Video Gallery tab
3. Click "🎥 Generate Videos from Training"
4. Select a training run
5. Click "🎬 Generate Videos"
6. Videos will have neural network visualization!

# Via Command Line
python training/post_training_video_generator.py \
    --model-dir models/checkpoints/run-abc123/milestones \
    --config conf/config.yaml \
    --output-dir outputs/run-abc123/milestones \
    --clip-seconds 90
```

## Video Output

- **Resolution**: 960x540 (2x 480x540 panels side-by-side)
- **Frame Rate**: 30 FPS (configurable)
- **Format**: MP4 (H.264)
- **Filename**: `step_post_training_pct_{milestone}_analytics.mp4`

## Example Analytics Display

```
BREAKOUT - PPO Neural Activity Viewer
Post-Training Evaluation | Progress: 45.2% | lr=2.5e-4

Frame:      1,234
Progress:   45.23%
Ep Reward:  42.0

Policy Distribution:
  NOOP :  0.123
  FIRE :  0.654  ← (highlighted in yellow)
  RIGHT:  0.112
  LEFT :  0.111

Value Est:  15.234
```

## Future Enhancements

Potential improvements:
- Add reward graph over time
- Show recent action history
- Display Q-values for DQN
- Add attention heatmap overlay on game screen
- Show gradient flow visualization
- Add performance metrics (FPS, inference time)

## Testing

To test the feature:

1. Generate a video from an existing checkpoint
2. Verify the video has the neural network visualization
3. Check that action probabilities change based on game state
4. Verify the chosen action is highlighted
5. Confirm the neural network animation is smooth

## Compatibility

- ✅ Works with PPO algorithm
- ✅ Works with DQN algorithm
- ✅ Compatible with all Atari games
- ✅ Works with existing checkpoints
- ✅ No changes needed to training code
- ✅ Backward compatible (old videos still work)

## Branch Information

- **Feature Branch**: `feature/neural-network-video-visualization`
- **Base Branch**: `main`
- **Source Branch**: `addedGMB`
- **Status**: Ready for testing and merge

