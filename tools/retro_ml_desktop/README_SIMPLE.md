# Retro ML Desktop - Simple Process Manager

A **much simpler** CustomTkinter desktop application that runs your existing ML training scripts directly as Python processes. **No Docker required!**

## Why This Is Better

✅ **No Docker complexity** - Uses your existing Python environment  
✅ **Direct process control** - Runs `training/train.py` directly  
✅ **Simpler setup** - Just install Python dependencies  
✅ **Better performance** - No container overhead  
✅ **Easier debugging** - Direct access to Python processes  
✅ **Works immediately** - Uses your existing training pipeline

## Quick Start

### 1. Install Dependencies

```bash
pip install -r tools/retro_ml_desktop/requirements.txt
```

### 2. Run the Application

```bash
python -m tools.retro_ml_desktop.main_simple
```

That's it! No Docker, no containers, no complexity.

## How It Works

The application directly runs your existing `training/train.py` script as Python subprocesses with:

- **Resource Control**: CPU core affinity, memory monitoring, GPU selection
- **Process Management**: Start, stop, remove training processes
- **Live Monitoring**: Real-time system metrics and process status
- **Log Streaming**: Live logs from training processes
- **Multiple Runs**: Run multiple training sessions simultaneously

## Features

### 🖥️ **Enhanced UI**

- **Training Processes Tab**: List of running/finished training processes
- **Progress Tracking Tab**: Detailed progress for training and video generation
- **Logs Tab**: Live log streaming from all processes
- **Sidebar**: System status and start training button

### 🎮 **Training Presets with Video Control**

- **Epic 10-Hour**: Full YouTube journey (10M timesteps, 10 hours of videos)
- **Epic 5-Hour**: Shorter journey (5M timesteps, 5 hours of videos)
- **PPO**: High-performance training (4M timesteps, 4 hours of videos)
- **DQN**: Medium-performance training (6M timesteps, 3 hours of videos)
- **Quick Test**: Fast testing (100K timesteps, 6 minutes of videos)

### 🔧 **Resource Management**

- **CPU Cores**: Set number of CPU cores to use
- **Memory Limit**: Monitor memory usage (warning only)
- **GPU Selection**: Choose specific GPU or auto-select
- **Process Priority**: Set process priority (low/normal/high)

### 📊 **System Monitoring**

- **CPU**: Usage percentage and core count
- **Memory**: Used/total memory with percentage
- **GPU**: Available GPUs and current usage

## Usage

### Starting Training

1. Click **"Start Training"** in the sidebar
2. Choose your configuration:
   - **Preset**: Epic 10-Hour, Epic 5-Hour, PPO, DQN, or Quick Test
   - **Game**: Select from 6 Atari games
   - **Algorithm**: PPO or DQN
   - **Training Length**: Total timesteps (controls training duration)
   - **Video Generation**: Target hours and number of milestone videos
   - **Resources**: CPU cores, memory, GPU (with auto-detection and recommendations)
3. **Optional**: Click "🔧 Advanced Resource Selection" for detailed system analysis
4. Click **"Start Training"**

### Advanced Resource Selection

The **Advanced Resource Selection** dialog provides:

#### 🔍 **Auto-Detection Features**

- **CPU Analysis**: Detects physical vs logical cores, current usage, availability
- **GPU Discovery**: Finds all GPUs with memory, utilization, temperature monitoring
- **Memory Assessment**: Shows total, available, and recommended allocation
- **Smart Recommendations**: Suggests optimal configuration based on system state

#### 🎯 **Intelligent Recommendations**

- **CPU**: Recommends 75% of available physical cores (minimum 2)
- **GPU**: Auto-selects GPU with most free memory
- **Memory**: Suggests 75% of system memory (leaves 25% for stability)
- **Performance Tips**: Provides optimization guidance

#### 📊 **Real-Time System Information**

- **CPU Cores**: Shows which cores are available vs busy
- **GPU Status**: Displays memory usage, temperature, utilization per GPU
- **Memory Status**: Current usage and availability
- **Resource Conflicts**: Warns about potential bottlenecks

The app will:

- Run `training/train.py` with your parameters
- Stream logs to the Logs tab
- Show the process in the Training Processes tab
- Monitor system resources

### Managing Processes

- **View Processes**: See all training processes with status and PID
- **Track Progress**: Detailed progress tracking for training and video generation
- **Stop Process**: Gracefully terminate a training process
- **Pause Process**: Suspend training (can be resumed later)
- **Resume Process**: Continue a paused training process
- **Remove Process**: Remove finished processes from the list
- **Clear Selected Data**: Delete training outputs for a specific process
- **Clear ALL Data**: Delete all training data (with double confirmation)
- **Live Logs**: Watch training progress in real-time

### Progress Tracking

The **Progress Tracking** tab shows detailed information for each training process:

#### 🧠 **Training Progress**

- **Current vs Total Timesteps**: Shows exact progress through training
- **Progress Percentage**: Visual indication of training completion
- **Estimated Time Remaining**: Calculated based on current training speed
- **Checkpoints Saved**: Number of model checkpoints created

#### 🎬 **Video Generation Progress**

- **Milestone Videos**: Number of milestone videos completed vs target
- **Hour Videos**: Number of 1-hour videos generated vs target hours
- **Total Duration**: Total video content generated so far
- **Video Status**: Real-time status of video generation process

#### 📁 **Output Files**

- **Model Checkpoints**: Count of saved model files
- **Milestone Videos**: Count of milestone video files
- **Hour Videos**: Count of hour-long video files
- **Output Directory**: Location of all generated files

### Training Control Features

#### 🎮 **Process Control**

- **Start**: Launch new training with custom configuration
- **Stop**: Gracefully terminate training (saves current progress)
- **Pause**: Suspend training process (keeps memory state)
- **Resume**: Continue paused training from exact same point
- **Remove**: Remove completed/stopped processes from list

#### 🗑️ **Data Management**

- **Clear Selected Data**: Delete all outputs for one training run
  - Removes model checkpoints
  - Deletes generated videos
  - Clears training logs
  - Preserves other training runs
- **Clear ALL Data**: Nuclear option - deletes everything
  - Requires double confirmation
  - Removes all training data from all runs
  - Resets the entire training environment

#### ⚠️ **Safety Features**

- **Graceful Shutdown**: Stop processes save current progress before terminating
- **Confirmation Dialogs**: All destructive actions require user confirmation
- **Double Confirmation**: Clearing all data requires two confirmations
- **Process Isolation**: Each training run has isolated data directories

### Enhanced User Experience

#### 📖 **Comprehensive Explanations**

- **Training Configuration**: Detailed explanations for timesteps, video generation
- **Resource Selection**: Clear guidance on CPU, GPU, and memory allocation
- **Performance Tips**: Built-in recommendations for optimal training
- **Real-Time Help**: Contextual tooltips and explanations throughout the UI

#### 🔧 **Smart Resource Management**

- **Auto-Detection**: Automatically discovers and analyzes system resources
- **Intelligent Defaults**: Sets optimal configuration based on your hardware
- **Resource Monitoring**: Real-time tracking of CPU, GPU, and memory usage
- **Conflict Prevention**: Warns about potential resource conflicts

#### 🎯 **Guided Configuration**

- **Preset Explanations**: Clear descriptions of what each preset does
- **Training Duration Estimates**: Shows expected training time for different configurations
- **Video Generation Guide**: Explains how milestone videos are created
- **Resource Impact**: Shows how different settings affect performance

## Configuration

### Training Presets

Edit `tools/retro_ml_desktop/training_presets.yaml`:

```yaml
presets:
  atari_ppo:
    description: "PPO training for Atari games - high performance"
    total_timesteps: 4000000
    vec_envs: 16
    save_freq: 200000
    default_resources:
      cpu_cores: 6
      memory_limit_gb: 16
      priority: "normal"
      gpu_id: "auto"
```

### Supported Games

- BreakoutNoFrameskip-v4
- PongNoFrameskip-v4
- SpaceInvadersNoFrameskip-v4
- AsteroidsNoFrameskip-v4
- MsPacmanNoFrameskip-v4
- FroggerNoFrameskip-v4

## Command Line Equivalent

When you start training through the UI, it runs commands like:

```bash
python training/train.py \
  --env BreakoutNoFrameskip-v4 \
  --algo ppo \
  --total-steps 4000000 \
  --vec-envs 16 \
  --save-freq 200000 \
  --run-id run-a1b2c3d4 \
  --video-dir outputs/run-a1b2c3d4 \
  --model-dir models/checkpoints \
  --log-dir logs/tb
```

With environment variables:

- `CUDA_VISIBLE_DEVICES=0` (if GPU selected)
- `OMP_NUM_THREADS=6` (based on CPU cores)

## Advantages Over Docker Version

| Feature            | Docker Version                   | Simple Version            |
| ------------------ | -------------------------------- | ------------------------- |
| **Setup**          | Install Docker + NVIDIA runtime  | Just `pip install`        |
| **Performance**    | Container overhead               | Native performance        |
| **Debugging**      | Container logs only              | Direct Python debugging   |
| **Resource Usage** | Higher memory usage              | Lower memory usage        |
| **Complexity**     | High (images, volumes, networks) | Low (just processes)      |
| **Startup Time**   | Slower (container creation)      | Faster (direct execution) |
| **File Access**    | Volume mounts required           | Direct file system access |

## System Requirements

- **Python 3.8+** with your existing ML environment
- **NVIDIA GPU** (optional, for GPU training)
- **Windows/Linux/macOS** (cross-platform)

## Troubleshooting

### "Training script not found"

- Ensure you're running from the project root
- Check that `training/train.py` exists

### "Failed to start training process"

- Check that your Python environment has all ML dependencies
- Verify the training script works manually
- Check available system resources

### GPU not detected

- Install GPU monitoring: `pip install nvidia-ml-py GPUtil`
- Check GPU availability: `nvidia-smi`
- Use CPU-only training (set GPU to "none")

### Process won't stop

- Use "Remove Selected" to remove stuck processes
- Check Task Manager for orphaned Python processes

## Files

```
tools/retro_ml_desktop/
├── main_simple.py           # Simple UI application (no Docker)
├── process_manager.py       # Direct Python process management
├── monitor.py              # System monitoring (same as before)
├── training_presets.yaml   # Simplified presets
├── requirements.txt        # Dependencies (no Docker)
└── README_SIMPLE.md        # This file
```

## Next Steps

This simple version gives you all the benefits of the desktop UI without any Docker complexity. You can:

1. **Start multiple training runs** with different games/algorithms
2. **Monitor system resources** in real-time
3. **View live training logs** from all processes
4. **Control resource usage** per training process
5. **Manage training lifecycle** (start/stop/remove)

The application works with your existing training setup and requires no additional infrastructure!

---

**TL;DR**: Run `python -m tools.retro_ml_desktop.main_simple` for a Docker-free ML training manager that works with your existing setup.
