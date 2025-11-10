# Video Gallery - User Guide

## Overview
The Video Gallery provides a centralized location to view all training videos generated during your ML experiments. Videos are automatically discovered from all training runs and organized by type.

## Accessing the Video Gallery

1. Open the ML Training Manager desktop application
2. Click on the **"Video Gallery"** tab at the top
3. The gallery will automatically scan for videos from all training runs

## Video Types

The gallery displays three types of videos:

### 🎯 Milestone Videos
- **Location**: `outputs/{run_id}/milestones/`
- **Purpose**: Capture agent performance at specific training progress milestones (0%, 1%, 5%, 10%, etc.)
- **Use Case**: Compare how the agent improves over time
- **Typical Duration**: 90 seconds per video

### 📊 Evaluation Videos
- **Location**: `outputs/{run_id}/eval/`
- **Purpose**: Full evaluation episodes to assess agent performance
- **Use Case**: Detailed performance analysis at specific checkpoints
- **Typical Duration**: 2 minutes per video

### ⏱️ Hour/Part Videos
- **Location**: `outputs/{run_id}/parts/`
- **Purpose**: Continuous training footage split into manageable segments
- **Use Case**: Create long-form training progression videos
- **Typical Duration**: 30 minutes to 1 hour per segment

## Using the Video Gallery

### Refreshing the Video List

Click the **"🔄 Refresh Videos"** button to:
- Scan all training process directories
- Discover newly created videos
- Update the video list

The refresh happens automatically when you open the tab, but you can manually refresh at any time.

### Filtering Videos

Use the **Filter** dropdown to show only specific video types:
- **All Videos**: Show everything (default)
- **Milestone Videos**: Only milestone clips
- **Hour Videos**: Only hour/part segments
- **Evaluation Videos**: Only evaluation episodes

### Video Information

Each video in the list shows:
- **Video Name**: The filename of the video
- **Type**: Milestone, Evaluation, or Hour
- **Duration**: Estimated video length
- **File Size**: Size in MB
- **Created**: When the video was created
- **Training Run**: Which training session created it

### Playing Videos

#### Quick Play
1. Select a video from the list
2. Click **"▶️ Play Video"**
3. Video opens in your default system video player (VLC, Windows Media Player, etc.)

#### Enhanced Video Player
1. Select a video from the list
2. Click **"🎬 Video Player"**
3. Opens a dialog with:
   - Detailed video information
   - Analysis tools
   - Frame extraction options
   - External player launch

#### Quick Preview
1. Select a video from the list
2. Click **"👁️ Quick Preview"**
3. Shows a preview window with video details

### Video Information Dialog

Click **"ℹ️ Video Info"** to see:
- Full file path
- Exact file size
- Creation and modification dates
- Video metadata
- Training context

### Managing Videos

#### Opening Video Folder
Click **"📁 Open Video Folder"** to:
- Open the outputs directory in File Explorer
- Browse all video files directly
- Access videos outside the application

#### Deleting Videos
1. Select a video from the list
2. Click **"🗑️ Delete Video"**
3. Confirm the deletion
4. Video is permanently removed from disk

⚠️ **Warning**: Deleted videos cannot be recovered!

## Understanding Video Paths

### Default Configuration
Videos are saved to: `D:/Python projects/ML project/Atari ML project/outputs/{run_id}/`

Where `{run_id}` is a unique identifier like: `run-20240115-143022-abc123`

### Custom Output Paths
If you specified a custom output path (e.g., D drive for more space):
- Videos are saved to: `{custom_path}/{run_id}/`
- The gallery automatically finds them

### Video Discovery Process

The gallery scans:
1. **Active Training Processes**: Checks output paths for all running/completed processes
2. **Default Outputs Directory**: Scans `outputs/` for any training runs
3. **Custom Paths**: Includes any custom output locations you specified

## Troubleshooting

### No Videos Appearing

**Check the Log Panel** (bottom of the screen):
- Look for messages like: `🔍 Scanning videos for X training processes...`
- Check if directories exist: `✅ Found X video(s)` or `⚠️ Directory does not exist`

**Common Causes**:
1. **Training hasn't generated videos yet**: Wait for training to progress
2. **Videos disabled in config**: Check if video recording is enabled
3. **Wrong output path**: Verify the output directory in process settings
4. **Permissions issue**: Ensure the app can read the video directories

**Solutions**:
1. Click **"🔄 Refresh Videos"** to rescan
2. Check the training process is actually running
3. Navigate to the output folder manually to verify videos exist
4. Check the log panel for error messages

### Videos Not Playing

**Possible Issues**:
1. **No video player installed**: Install VLC or another video player
2. **Corrupted video file**: Video may not have finished encoding
3. **File permissions**: Check you have read access to the file

**Solutions**:
1. Install VLC Media Player (recommended)
2. Try opening the video file directly from File Explorer
3. Check if the file size is reasonable (not 0 bytes)

### Videos in Wrong Location

If videos are being saved to an unexpected location:
1. Check the training process configuration
2. Look at the log panel for the actual paths being used
3. Verify the custom output path setting (if used)

## Tips for Best Results

### Organizing Videos
- Use meaningful training run names
- Keep related experiments in the same output directory
- Regularly clean up old videos to save disk space

### Video Quality
- Milestone videos are great for quick progress checks
- Evaluation videos provide detailed performance analysis
- Hour videos are best for creating compilation/progress videos

### Disk Space Management
- Videos can be large (especially hour segments)
- Monitor disk space if running long training sessions
- Delete old videos you no longer need
- Consider using D drive or external storage for videos

### Performance Analysis
- Compare milestone videos from different training runs
- Watch for improvements in agent behavior over time
- Use evaluation videos to assess final performance
- Create compilations from hour segments for presentations

## Advanced Features

### Video Analysis (Coming Soon)
- Automatic score tracking from video
- Performance metrics extraction
- Side-by-side comparison tools
- Thumbnail generation

### Batch Operations (Coming Soon)
- Select multiple videos
- Batch delete
- Batch export
- Create compilations

### In-App Player (Coming Soon)
- Play videos directly in the application
- Frame-by-frame analysis
- Annotation tools
- Screenshot capture

## Keyboard Shortcuts

- **F5**: Refresh video list
- **Delete**: Delete selected video (with confirmation)
- **Enter**: Play selected video
- **Ctrl+O**: Open video folder

## Video File Formats

Supported formats:
- **MP4** (recommended, most common)
- **AVI** (alternative format)
- **MOV** (QuickTime format)
- **MKV** (Matroska format)
- **WEBM** (Web format)

## Getting Help

If you encounter issues:
1. Check the log panel for error messages
2. Verify video files exist in the output directory
3. Ensure you have a video player installed
4. Check file permissions
5. Try refreshing the video list

For persistent issues, check the application logs or contact support.

