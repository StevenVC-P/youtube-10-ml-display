# Retro ML Desktop - Container Manager

A CustomTkinter-based desktop application for launching and managing multiple ML training runs concurrently with Docker containers.

## Features

- **System Monitoring**: Real-time CPU, memory, and GPU monitoring
- **Container Management**: Start, stop, and remove Docker containers
- **Training Presets**: Configurable presets for different training scenarios
- **Live Log Streaming**: Real-time container log viewing
- **Resource Control**: Set CPU, memory, GPU, and shared memory limits
- **Multi-Game Support**: Support for multiple Atari games
- **Algorithm Selection**: PPO and DQN algorithm support

## Prerequisites

### Required Software

1. **Python 3.8+** with pip
2. **Docker Desktop** (Windows/macOS) or **Docker Engine** (Linux)
3. **NVIDIA Docker Runtime** (for GPU support)

### GPU Support (Optional but Recommended)

For GPU-accelerated training, you need:

- NVIDIA GPU with CUDA support
- NVIDIA drivers
- NVIDIA Container Toolkit

#### Windows GPU Setup

1. Install [NVIDIA drivers](https://www.nvidia.com/drivers/)
2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
3. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Linux GPU Setup

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Installation

### 1. Install Dependencies

From the project root directory:

```bash
# Install Python dependencies
pip install -r tools/retro_ml_desktop/requirements.txt
```

### 2. Build Docker Images

Ensure you have the required Docker images built. The application expects:

- `sb3-atari:latest` - Stable Baselines3 with Atari environments

Build the image (example):

```bash
# From project root
docker build -t sb3-atari:latest -f docker/Dockerfile.atari .
```

### 3. Create Output Directories

```bash
# From project root
mkdir -p outputs datasets
```

## Usage

### Starting the Application

From the project root directory:

```bash
python -m tools.retro_ml_desktop.main
```

### Application Interface

The application has three main tabs:

#### 1. Dashboard Tab

- **CPU Metrics**: Usage percentage, frequency, core count
- **Memory Metrics**: Used/total memory, usage percentage
- **GPU Metrics**: Load, memory usage, temperature (if available)

#### 2. Containers Tab

- **Container List**: Shows all training containers with status
- **Controls**: Refresh, Stop Selected, Remove Selected
- **Container Info**: Name, image, status, creation time

#### 3. Logs Tab

- **Live Log Viewer**: Real-time streaming of container logs
- **Auto-scroll**: Automatically scrolls to show latest logs
- **Clear Logs**: Button to clear the log display

### Starting a Training Run

1. Click **"Start Training"** in the sidebar
2. Configure the training parameters:
   - **Preset**: Choose from predefined configurations
   - **Game**: Select Atari game environment
   - **Algorithm**: Choose PPO or DQN
   - **Run ID**: Unique identifier for this run
   - **Resource Limits**: Set CPU, memory, GPU constraints
3. Click **"Start Training"**

The application will:
- Create a new Docker container
- Start training with specified parameters
- Begin streaming logs to the Logs tab
- Update the container list

### Folder Configuration

Use the sidebar to configure:

- **Outputs Directory**: Where training artifacts are saved
- **Datasets Directory**: Where datasets are mounted (if needed)

These directories are mounted into containers as `/outputs` and `/datasets`.

## Configuration

### Training Presets

Edit `tools/retro_ml_desktop/training_presets.yaml` to customize:

- **Container images**
- **Training commands**
- **Default resource limits**
- **Environment variables**
- **Volume mounts**

Example preset:

```yaml
presets:
  atari_ppo:
    image: "sb3-atari:latest"
    command: >
      python -u train.py
      --env {game}
      --algo {algo}
      --total-steps 4000000
      --run-id {run_id}
      --video-dir /outputs/{run_id}
      --save-freq 200000
    workdir: "/workspace"
    volumes:
      - "{host_outputs}:/outputs"
      - "{host_datasets}:/datasets"
    env:
      - "WANDB_MODE=disabled"
      - "OMP_NUM_THREADS={omp_threads}"
    gpu: "auto"
    default_resources:
      cpus: 6
      mem_limit: "16g"
      shm_size: "2g"
```

### Supported Games

- BreakoutNoFrameskip-v4
- PongNoFrameskip-v4
- SpaceInvadersNoFrameskip-v4
- AsteroidsNoFrameskip-v4
- MsPacmanNoFrameskip-v4
- FroggerNoFrameskip-v4

### Resource Limits

- **CPUs**: Number of CPU cores (float, e.g., 2.5)
- **Memory**: Memory limit (string, e.g., "16g", "2048m")
- **Shared Memory**: Shared memory size (string, e.g., "2g")
- **GPU**: GPU allocation ("auto", "none", "0", "1", "0,1")

## Troubleshooting

### Common Issues

#### "Failed to connect to Docker daemon"

**Cause**: Docker is not running or not accessible.

**Solutions**:
- Start Docker Desktop (Windows/macOS)
- Start Docker service: `sudo systemctl start docker` (Linux)
- Check Docker permissions: Add user to docker group (Linux)

#### "Docker image 'sb3-atari:latest' not found"

**Cause**: Required Docker image is not built.

**Solution**: Build the image:
```bash
docker build -t sb3-atari:latest -f docker/Dockerfile.atari .
```

#### "GPU/NVIDIA runtime error"

**Cause**: NVIDIA Docker runtime not installed or configured.

**Solutions**:
- Install NVIDIA Container Toolkit
- Restart Docker after installation
- Test with: `docker run --gpus all nvidia/cuda:11.0-base nvidia-smi`

#### "No GPUs detected"

**Cause**: GPU monitoring libraries not installed or no GPU available.

**Solutions**:
- Install GPU monitoring: `pip install nvidia-ml-py GPUtil`
- Check GPU availability: `nvidia-smi`
- Use CPU-only training (set GPU to "none")

#### Permission denied errors (Linux)

**Cause**: User not in docker group.

**Solution**:
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```

### Log Analysis

Container logs are streamed in real-time to the Logs tab. Look for:

- **Training progress**: Episode rewards, timesteps
- **Error messages**: Python exceptions, CUDA errors
- **Resource warnings**: OOM errors, GPU memory issues

### Performance Tips

1. **Resource Allocation**: Don't over-allocate resources
2. **Multiple Runs**: Monitor total system usage when running multiple containers
3. **GPU Memory**: Each container needs dedicated GPU memory
4. **Disk Space**: Ensure sufficient space in outputs directory

## Development

### Project Structure

```
tools/retro_ml_desktop/
├── __init__.py              # Package initialization
├── main.py                  # Main application and UI
├── docker_manager.py        # Docker container management
├── monitor.py              # System monitoring
├── training_presets.yaml   # Training configurations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

### Extending the Application

To add new features:

1. **New Games**: Add to `games` list in `training_presets.yaml`
2. **New Algorithms**: Add to `algorithms` list
3. **New Presets**: Add new preset configurations
4. **Custom Monitoring**: Extend `SystemMonitor` class
5. **UI Enhancements**: Modify `RetroMLDesktop` class

## License

This project is part of the larger ML training repository. See the main project LICENSE file.
