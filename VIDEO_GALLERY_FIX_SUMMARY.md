# Video Gallery Fix - Training Videos Not Appearing

## Problem

When completing a training session, videos were not appearing in the Video Gallery tab. Users couldn't see their training progress videos.

## Root Cause

The issue had TWO parts:

### Part 1: Videos Were Not Being Generated During Training

Video recording is **intentionally disabled** during training to avoid slowing down the training process:

- `milestone_clip_seconds` is set to 0 (line 227 in process_manager.py)
- `eval_clip_seconds` is set to 0 (line 228 in process_manager.py)
- Videos are meant to be generated **after training** using the `PostTrainingVideoGenerator`
- However, there was **no UI button** to trigger post-training video generation

### Part 2: Path Resolution Problem (for when videos ARE generated)

When videos are generated, there was also a path resolution problem:

1. **Relative Path Storage**: Training processes store video output paths as relative paths in the configuration:

   - `videos_milestones`: `outputs/{run_id}/milestones`
   - `videos_eval`: `outputs/{run_id}/eval`
   - `videos_parts`: `outputs/{run_id}/parts`

2. **Missing Path Resolution**: The `get_process_output_paths()` method in `process_manager.py` was returning these relative paths without resolving them to absolute paths.

3. **Gallery Scan Failure**: The Video Gallery tried to scan these paths using `Path(milestones_path).exists()`, which failed because:
   - Relative paths were being checked against the current working directory
   - Not against the project root where the videos actually exist

## Solution

### 1. Added "Generate Videos" Button to Video Gallery

**File**: `tools/retro_ml_desktop/main_simple.py`

**New Feature**: Added a "🎥 Generate Videos from Training" button that:

- Shows all completed training runs
- Allows users to select a run
- Lets users configure video clip length
- Generates milestone videos from saved checkpoints
- Shows progress in the log panel
- Automatically refreshes the video gallery when complete

**How to Use**:

1. Complete a training session (or use an existing completed run)
2. Go to the Video Gallery tab
3. Click "🎥 Generate Videos from Training"
4. Select a training run from the list
5. Configure clip length (default: 90 seconds)
6. Click "🎬 Generate Videos"
7. Wait for generation to complete
8. Videos will appear in the gallery!

### 2. Fixed `process_manager.py` - `get_process_output_paths()` Method

**File**: `tools/retro_ml_desktop/process_manager.py`

**Changes**:

- Added a helper function `resolve_path()` that converts relative paths to absolute paths
- All paths returned by `get_process_output_paths()` are now absolute paths resolved relative to `self.project_root`
- If a path is already absolute, it's returned as-is
- If a path is relative, it's resolved: `self.project_root / path`

**Code Changes**:

```python
def get_process_output_paths(self, process_id: str) -> Dict[str, str]:
    """Get the output paths for a specific process.

    Returns absolute paths resolved relative to the project root.
    """
    # ... existing validation code ...

    # Helper function to resolve paths to absolute
    def resolve_path(path_str: str) -> str:
        if not path_str:
            return ''
        path = Path(path_str)
        # If already absolute, return as-is; otherwise resolve relative to project root
        if path.is_absolute():
            return str(path)
        else:
            return str(self.project_root / path)

    # Extract and resolve all paths to absolute paths
    videos_milestones = resolve_path(paths.get('videos_milestones', ''))
    videos_base = str(Path(videos_milestones).parent) if videos_milestones else ''

    return {
        'videos_base': videos_base,
        'videos_milestones': videos_milestones,
        'videos_eval': resolve_path(paths.get('videos_eval', '')),
        'videos_parts': resolve_path(paths.get('videos_parts', '')),
        'models': resolve_path(paths.get('models', '')),
        'logs_tb': resolve_path(paths.get('logs_tb', ''))
    }
```

### 2. Enhanced Video Discovery Logging

**File**: `tools/retro_ml_desktop/main_simple.py`

**Changes**:

- Added detailed logging to the `_discover_videos()` method to help debug video discovery
- Logs show:
  - Number of processes being scanned
  - Each process's video directories being checked
  - Whether directories exist or not
  - Number of videos found in each location
  - Total videos discovered

**Benefits**:

- Users can now see in the log panel exactly where the system is looking for videos
- Easy to identify if directories don't exist or are empty
- Helps diagnose path configuration issues

### 3. Improved Error Handling in Video Scanning

**File**: `tools/retro_ml_desktop/main_simple.py`

**Changes**:

- Enhanced `_scan_process_videos()` to provide better feedback
- Logs warnings when expected directories don't exist
- Provides detailed error messages with stack traces for debugging
- Continues scanning even if one directory fails

## How It Works Now

### Training Workflow

1. **Training Process Creation**:

   - Process is created with video recording **disabled** (for performance)
   - Directories are created at absolute paths: `project_root / outputs / {run_id} / milestones`
   - Model checkpoints are saved during training

2. **Post-Training Video Generation** (NEW!):

   - User clicks "🎥 Generate Videos from Training" button
   - Selects a completed training run
   - System loads saved checkpoints
   - Generates videos showing agent performance at different training milestones (10%, 20%, 30%, etc.)
   - Videos are saved to: `D:/Python projects/ML project/Atari ML project/outputs/{run_id}/milestones/`

3. **Video Gallery Refresh**:
   - Gallery calls `get_process_output_paths(process_id)`
   - Method returns absolute paths: `D:/Python projects/ML project/Atari ML project/outputs/{run_id}/milestones`
   - Gallery scans these absolute paths successfully
   - Videos are found and displayed

## Testing

To verify the fix works:

1. **Start a Training Session**:

   - Create a new training process
   - Let it run for a few minutes to generate videos

2. **Check the Log Panel**:

   - Look for messages like:
     ```
     🔍 Scanning videos for 1 training processes...
       📂 Checking process Breakout-PPO (ID: abc12345...)
          Milestones: D:/Python projects/ML project/Atari ML project/outputs/run-abc12345/milestones
          Eval: D:/Python projects/ML project/Atari ML project/outputs/run-abc12345/eval
          Parts: D:/Python projects/ML project/Atari ML project/outputs/run-abc12345/parts
          ✅ Found 3 video(s)
     ✅ Video discovery complete: Found 3 total video(s)
     ```

3. **Switch to Video Gallery Tab**:

   - Click "🔄 Refresh Videos" button
   - Videos should now appear in the list
   - You should see videos with types: Milestone, Evaluation, Hour

4. **Play Videos**:
   - Select a video from the list
   - Click "▶️ Play Video" to watch it
   - Or click "🎬 Video Player" for more options

## Additional Features

The Video Gallery now provides:

- **Automatic Discovery**: Scans all training process output directories
- **Multiple Video Types**: Milestone, Evaluation, and Hour/Part videos
- **Filtering**: Filter by video type
- **Detailed Info**: Shows duration, size, creation date, and training run
- **Easy Playback**: One-click video playback in your default player
- **Comprehensive Logging**: See exactly what's being scanned and found

## Common Video Locations

Videos are saved to these locations by default:

- **Milestone Videos**: `outputs/{run_id}/milestones/`
- **Evaluation Videos**: `outputs/{run_id}/eval/`
- **Hour/Part Videos**: `outputs/{run_id}/parts/`

If you specified a custom output path (e.g., D drive), videos will be there instead:

- **Custom Path**: `{custom_path}/{run_id}/milestones/`

## Troubleshooting

If videos still don't appear:

1. **Check the Log Panel**: Look for warning messages about missing directories
2. **Verify Video Creation**: Navigate to the output directory manually and check if videos exist
3. **Check Permissions**: Ensure the application has read access to the video directories
4. **Refresh Manually**: Click the "🔄 Refresh Videos" button in the Video Gallery
5. **Check Process Status**: Ensure the training process actually generated videos (some configurations may skip video generation)

## Future Enhancements

Potential improvements for the Video Gallery:

- Video thumbnails/previews
- In-app video player (currently uses external player)
- Video comparison tools
- Automatic refresh when new videos are created
- Video metadata extraction (duration, resolution, etc.)
- Batch video operations (delete multiple, export, etc.)
