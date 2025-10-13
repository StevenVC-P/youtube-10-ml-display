# 🎮 Atari ML Epic Training Project

A machine learning project for creating epic 10-hour neural network learning journey videos. Watch AI agents progress from random play to mastery across multiple Atari games through continuous "epic" training sessions.

## 🎯 Project Overview

This project creates engaging long-form content by:

- **Epic Training System**: Sequential 10-hour training sessions with model continuity
- **Multi-Game Support**: 6 Atari games (Breakout, Pong, Space Invaders, Asteroids, Pacman, Frogger)
- **Progressive Learning**: Each epic builds on the previous epic's final model
- **Automated Pipeline**: Sequential epic launcher for hands-off training

### Current Status: Epic Training System Complete ✅

- ✅ **Epic 1 (Breakout)**: Complete - 10 hours from scratch to competent play
- ✅ **Epic 2 (Breakout)**: Complete - Advanced mastery training
- 🔄 **Epic 3 (Breakout)**: Currently running - Perfect play optimization
- ⏳ **Epics 4-10**: Queued for automatic sequential launch
- 🎮 **Multi-Game Ready**: Pong, Space Invaders, Asteroids, Pacman, Frogger

**Target OS:** Windows 10/11 (PowerShell)
**Python:** 3.11+ (virtualenv: `.venv`)

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Windows 10/11 with PowerShell
- Git
- FFmpeg (for video generation)

### Installation

1. **Clone and setup**:

```powershell
git clone <repository-url>
cd "atari-ml-epic-training"
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

### Epic Training Usage

```powershell
# Start a single epic (10 hours)
python epic_training/scripts/train_epic_continuous.py --game breakout --epic 1

# Launch sequential epics (auto-progression)
python epic_training/scripts/launch_breakout_epics_3_to_10.py

# Check epic status
python epic_training/scripts/check_epic_status.py

# Test a different game
python epic_training/scripts/train_epic_continuous.py --game pong --epic 1 --test

# Launch epic for different game
python epic_training/scripts/train_epic_continuous.py --game pong --epic 1 --hours 10
```

## Project Structure

```
atari-ml-epic-training/
  📁 Core Components
  ├── agents/              # ML algorithm implementations
  ├── envs/               # Environment wrappers and configurations
  ├── training/           # Training scripts and callbacks
  └── conf/               # Configuration files

  📁 Epic Training System
  └── epic_training/
      ├── scripts/        # Epic training and launcher scripts
      ├── configs/        # Epic-specific configurations
      └── utils/          # Epic training utilities

  📁 Games & Data
  ├── games/              # Per-game epic training data
  │   ├── breakout/       # Breakout epic directories (epic_001, epic_002, etc.)
  │   ├── pong/           # Pong epic directories
  │   └── ...             # Other supported games (space_invaders, asteroids, etc.)
  ├── models/             # Shared model checkpoints
  ├── logs/               # Training logs and TensorBoard data
  └── video/              # Generated video content

  📁 Tools & Utilities
  ├── tools/              # Analysis and utility scripts
  ├── tests/              # Test suite
  └── docs/               # Documentation
```

## Epic Training System

### Supported Games

| Game               | Environment ID              | Difficulty | Description                           |
| ------------------ | --------------------------- | ---------- | ------------------------------------- |
| **breakout**       | BreakoutNoFrameskip-v4      | ⭐⭐⭐     | Classic brick-breaking game           |
| **pong**           | PongNoFrameskip-v4          | ⭐⭐       | Simple paddle game                    |
| **space_invaders** | SpaceInvadersNoFrameskip-v4 | ⭐⭐⭐     | Shoot descending aliens               |
| **asteroids**      | AsteroidsNoFrameskip-v4     | ⭐⭐⭐⭐   | Navigate and shoot asteroids          |
| **pacman**         | MsPacmanNoFrameskip-v4      | ⭐⭐⭐     | Maze navigation and pellet collection |
| **frogger**        | FroggerNoFrameskip-v4       | ⭐⭐⭐     | Cross roads and rivers safely         |

### Epic Progression

Each game follows a 3-epic progression:

1. **Epic 1 - From Scratch**: 10 hours of training from random play to basic competency
2. **Epic 2 - Advanced Mastery**: 10 hours building on Epic 1's model for advanced play
3. **Epic 3 - Perfect Play**: 10 hours optimizing from advanced to expert-level performance

### Memory Configuration

The system uses ultra-low memory configuration for stability:

- **1 vectorized environment** (vs 8 standard)
- **32 batch size** (vs 256 standard)
- **16 n_steps** (vs 128 standard)
- **Slower but reliable**: ~33 hours per epic, no crashes

## Common Commands

### Epic Training

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
