# YouTube 10 ML Display

A machine learning project for creating continuous ~10-hour YouTube content showing an AI agent learning to play Atari games. This system implements a dual-track approach with fast training and real-time visualization.

## 🎯 Project Overview

This project creates engaging YouTube content by:

- **Track A**: Fast PPO training on vectorized Atari environments
- **Track B**: Real-time visualization with grid display and progress tracking
- **Output**: Continuous ~10-hour video showing learning progression

### Current Status: Sprint 1 Complete ✅

- ✅ Environment factory with Atari wrappers
- ✅ Comprehensive test suite (12/12 tests passing)
- ✅ BreakoutNoFrameskip-v4 validated with shape (84, 84, 1, 4)
- 🚀 Ready for Sprint 2: PPO Trainer

**Target OS:** Windows 10/11 (PowerShell)
**Python:** 3.13+ (virtualenv: `.venv`)

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Windows 10/11 with PowerShell
- Git
- FFmpeg (for video generation)

### Installation

1. **Clone and setup**:

```powershell
git clone <repository-url>
cd "Atari ML project"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. **Install dependencies**:

```powershell
pip install -r requirements.txt
pip install ale-py autorom "gymnasium[atari]"
AutoROM --accept-license
```

3. **Validate setup**:

```powershell
python -m pytest tests/test_env_factory.py -v
```

Expected output: `12 passed` with Atari environment working.

## Project Structure

```
youtube-10-ml-display/
  conf/
    config.yaml              # Main configuration file
  envs/
    atari_wrappers.py       # Atari environment wrappers
    make_env.py             # Environment factory
  agents/
    algo_factory.py         # Algorithm factory (PPO/DQN)
  training/
    train.py                # Main training script
    eval.py                 # Evaluation script
    callbacks.py            # Training callbacks
  stream/
    stream_eval.py          # Continuous evaluation streamer
    ffmpeg_io.py            # FFmpeg wrapper
  video/
    milestones/             # Milestone video clips
    eval/                   # Evaluation video clips
    render/
      parts/                # Video segments for final render
  models/
    checkpoints/            # Model checkpoints
  logs/
    tb/                     # TensorBoard logs
  video_tools/
    build_manifest.py       # Build video manifest
    render_supercut.py      # Create final supercut
    concat_segments.py      # Concatenate video segments
  scripts/
    estimate_storage.py     # Storage estimation utility
  tests/
    test_env_factory.py     # Environment tests
    test_config_schema.py   # Config validation tests
```

## Common Commands

### Training

```powershell
# Start full training
python training\train.py --config conf\config.yaml

# Start training with time limit (dry run)
python training\train.py --config conf\config.yaml --dryrun-seconds 120
```

### Evaluation

```powershell
# Evaluate from checkpoint
python training\eval.py --checkpoint models\checkpoints\latest.zip --episodes 2 --seconds 120
```

### Streaming

```powershell
# Start continuous streamer
python stream\stream_eval.py --config conf\config.yaml --grid 4 --fps 30 --save-mode segments --segment-seconds 1800
```

### Video Processing

```powershell
# Build video manifest
python video_tools\build_manifest.py > manifest.csv

# Create final supercut
python video_tools\render_supercut.py --manifest manifest.csv --target-hours 10 --music assets\music.mp3

# Concatenate segments
python video_tools\concat_segments.py --input video\render\parts --output video\render\youtube_10h.mp4
```

## Configuration

The main configuration is in `conf/config.yaml`. Key settings:

- **Game**: Switch between Atari games by changing `game.env_id`
- **Training**: Adjust PPO hyperparameters in `train` section
- **Recording**: Control video quality and milestone percentages
- **Streaming**: Configure real-time streaming grid and output format

## Requirements

- Python 3.11+
- FFmpeg (for video processing)
- Stable-Baselines3
- Gymnasium[atari]
- PyTorch
- OpenCV
- PyYAML

## Development

### Running Tests

```powershell
# Run config schema tests
python -m pytest tests\test_config_schema.py -v

# Run all tests
python -m pytest tests\ -v
```

### Storage Estimation

```powershell
# Estimate storage requirements
python scripts\estimate_storage.py --hours 10 --fps 30 --quality high
```

## Multi-Game Support

To switch to a different Atari game, simply update the config:

```yaml
game:
  env_id: "ALE/Pong-v5" # or any other Atari game
```

No code changes required!

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Ensure FFmpeg is in PATH or set absolute path in config
2. **ROM license**: Run `pip install gymnasium[accept-rom-license]` for Atari ROMs
3. **CUDA issues**: Ensure PyTorch CUDA version matches your GPU drivers

### Getting Help

Check the logs in `logs/tb/` for training metrics and errors.

## License

This project is for educational and research purposes.
