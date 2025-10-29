# ML Training System - Current Architecture

## Overview

This document describes the current architecture of the ML training system as of October 2025. The system has evolved from a Docker-based container approach to a streamlined desktop application with comprehensive experiment tracking.

## Technology Stack

### Desktop Application
- **Framework**: CustomTkinter (Modern Python GUI)
- **Architecture**: Single-process desktop app with subprocess management
- **Platform**: Windows (with cross-platform compatibility)

### ML Training Pipeline
- **Framework**: Stable-Baselines3 (PPO, DQN algorithms)
- **Environment**: Gymnasium/ALE for Atari games
- **Backend**: PyTorch with CUDA support
- **Process Management**: Direct Python subprocess execution

### Experiment Tracking
- **Database**: SQLite with WAL mode for concurrent access
- **Metrics Collection**: Real-time log parsing and system monitoring
- **Visualization**: Matplotlib with interactive charts
- **Data Models**: Comprehensive experiment and metrics tracking

### Error Handling & Diagnostics
- **CUDA Diagnostics**: Intelligent GPU error detection and solutions
- **User-Friendly Errors**: Actionable error messages with troubleshooting steps
- **System Analysis**: Comprehensive hardware and software diagnostics

## Core Components

### 1. Desktop Application (`tools/retro_ml_desktop/main_simple.py`)
**Purpose**: Main user interface and application orchestration

**Key Features**:
- Training process management (start/stop/pause/resume)
- Real-time progress monitoring
- Video gallery for training recordings
- CUDA diagnostics and troubleshooting
- Settings and configuration management

**Tabs**:
- **Training Processes**: Active training management
- **🧪 ML Dashboard**: Experiment tracking and analysis
- **🎬 Video Gallery**: Training video playback
- **⚙️ Settings**: Configuration and preferences

### 2. ML Experiment Tracking System

#### Database Layer (`ml_database.py`)
- SQLite database with thread-safe operations
- Experiment runs, training metrics, and configuration storage
- Efficient querying and data export capabilities

#### Metrics Collection (`ml_collector.py`)
- Real-time log parsing from training processes
- System resource monitoring (CPU, GPU, memory)
- Automatic experiment creation and data collection

#### Visualization (`ml_dashboard.py`, `ml_plotting.py`)
- Interactive training curves (rewards, losses, learning dynamics)
- Multi-run comparison and analysis
- Real-time updates with auto-refresh

### 3. CUDA Diagnostics System (`cuda_diagnostics.py`)
**Purpose**: Intelligent GPU error handling and system analysis

**Features**:
- Comprehensive system diagnostics (GPU, memory, drivers)
- Error pattern recognition with specific solutions
- Configuration recommendations based on hardware
- User-friendly error messages with actionable steps

### 4. Process Management (`process_manager.py`)
**Purpose**: Training process lifecycle management

**Capabilities**:
- Subprocess creation and monitoring
- Status tracking (running, paused, stopped, failed)
- Resource usage monitoring
- Enhanced error detection and reporting

### 5. Training Pipeline (`training/train.py`)
**Purpose**: Core ML training execution

**Features**:
- PPO and DQN algorithm support
- Atari environment integration
- Progress logging and milestone tracking
- Enhanced CUDA error handling

## Data Flow

```
User Input (Desktop UI)
    ↓
Process Manager (Subprocess Creation)
    ↓
Training Script (ML Execution)
    ↓
Log Output (Real-time Streaming)
    ↓
Metrics Collector (Parsing & Analysis)
    ↓
SQLite Database (Persistent Storage)
    ↓
ML Dashboard (Visualization & Analysis)
```

## File Structure

```
tools/retro_ml_desktop/
├── main_simple.py           # Main desktop application
├── ml_database.py           # SQLite database operations
├── ml_collector.py          # Real-time metrics collection
├── ml_dashboard.py          # Experiment tracking UI
├── ml_plotting.py           # Interactive visualization
├── cuda_diagnostics.py     # GPU diagnostics and error handling
├── process_manager.py       # Training process management
└── video_player.py          # Video gallery functionality

training/
├── train.py                 # Core training script
├── callbacks.py             # Training callbacks
└── ml_analytics_callback.py # Analytics integration

conf/
├── config.yaml              # Training configuration
└── config.py                # Configuration management
```

## Key Improvements Over Previous Architecture

### 1. Simplified Deployment
- **Before**: Docker containers with complex orchestration
- **After**: Single desktop application with direct process management

### 2. Enhanced User Experience
- **Before**: Web-based UI requiring server setup
- **After**: Native desktop application with immediate startup

### 3. Professional ML Tracking
- **Before**: Basic progress monitoring
- **After**: Comprehensive experiment tracking with database persistence

### 4. Intelligent Error Handling
- **Before**: Cryptic CUDA errors with no guidance
- **After**: User-friendly diagnostics with actionable solutions

### 5. Real-time Visualization
- **Before**: Static progress displays
- **After**: Interactive charts with multi-run comparison

## Configuration

### Training Parameters
- Configurable via YAML files and UI dialogs
- Algorithm selection (PPO, DQN)
- Environment settings (game selection, preprocessing)
- Hardware optimization (batch size, parallel environments)

### System Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended for large batch sizes)
- **Storage**: SSD recommended for database and video storage
- **OS**: Windows 10/11 (cross-platform compatible)

## Future Enhancements

### Planned Features
- Advanced hyperparameter optimization
- Distributed training support
- Cloud integration for large-scale experiments
- Advanced statistical analysis tools
- Automated report generation

### Technical Debt
- Migrate remaining legacy documentation
- Enhance cross-platform compatibility
- Implement comprehensive test suite
- Add configuration validation

## Getting Started

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Launch Application**: `python -m tools.retro_ml_desktop.main_simple`
3. **Configure Training**: Use the Start Training dialog
4. **Monitor Progress**: Switch to ML Dashboard tab
5. **Troubleshoot Issues**: Use CUDA Diagnostics button

## Support

For issues and troubleshooting:
1. Check CUDA Diagnostics for hardware issues
2. Review training logs in the ML Dashboard
3. Consult error messages for specific solutions
4. Check system requirements and dependencies
