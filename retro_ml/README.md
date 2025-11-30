# Retro ML Core Package

## Overview

The `retro_ml` package provides the core functionality for the Retro ML Desktop application. This is a Phase 0 implementation focusing on backend components without UI dependencies.

## Package Structure

```
retro_ml/
├── core/
│   ├── experiments/     # Experiment configuration
│   │   └── config.py    # ExperimentConfig (single source of truth)
│   ├── engine/          # Training engine
│   │   └── training_engine.py  # TrainingEngine and run_experiment()
│   ├── metrics/         # Metrics collection
│   │   └── event_bus.py # MetricEventBus (pub/sub pattern)
│   └── artifacts/       # Artifact writers
│       └── writers.py   # MetricsWriter, LogWriter, ModelWriter
└── ui/                  # UI components (Phase 1)
```

## Core Components

### ExperimentConfig

Single source of truth for all experiment parameters.

```python
from retro_ml import ExperimentConfig, RunType

config = ExperimentConfig(
    game_id="BreakoutNoFrameskip-v4",
    algorithm="PPO",
    run_type=RunType.QUICK,
    seed=42,
    output_dir="D:/RetroML/experiments"
)
```

**Features:**
- Pydantic-based validation
- Automatic timestep calculation from run_type
- Path generation for artifacts
- JSON serialization/deserialization

### TrainingEngine

Core training orchestration.

```python
from retro_ml import run_experiment

result = run_experiment(config)

print(f"Success: {result.success}")
print(f"Training time: {result.training_time:.2f}s")
print(f"Metrics: {result.metrics_path}")
```

**Features:**
- Environment creation and wrapping
- Model initialization (PPO, A2C, DQN)
- Training loop execution
- Automatic artifact generation
- GPU detection and usage

### MetricEventBus

Publish/subscribe pattern for metrics collection.

```python
from retro_ml import MetricEventBus, MetricEvent, EventType

bus = MetricEventBus()

# Subscribe to events
def on_step(event):
    print(f"Step {event.data['step']}")

bus.subscribe(EventType.TRAINING_STEP, on_step)

# Emit events
bus.emit(MetricEvent(
    event_type=EventType.TRAINING_STEP,
    experiment_id="exp-123",
    data={"step": 1, "reward": 10.5}
))
```

**Features:**
- Decouples metric producers from consumers
- Event history with filtering
- Error handling in callbacks
- Multiple subscribers per event type

### Artifact Writers

Automatic artifact generation.

**MetricsWriter:**
- Saves metrics to JSON and CSV
- Subscribes to metric events
- Accumulates metrics in memory

**LogWriter:**
- Saves event log to file
- Subscribes to all event types
- Timestamped entries

**ModelWriter:**
- Saves model checkpoints
- Compatible with Stable-Baselines3

## Quick Start

```python
from retro_ml import ExperimentConfig, run_experiment, RunType

# Create configuration
config = ExperimentConfig(
    game_id="BreakoutNoFrameskip-v4",
    run_type=RunType.QUICK,
    seed=42
)

# Run experiment
result = run_experiment(config)

# Check results
if result.success:
    print(f"Training completed in {result.training_time:.2f}s")
    print(f"Metrics saved to: {result.metrics_path}")
    print(f"Model saved to: {result.model_path}")
else:
    print(f"Training failed: {result.error}")
```

## Artifacts Generated

Each experiment produces:

1. **metrics.json** - Training metrics in JSON format
2. **metrics.csv** - Training metrics in CSV format
3. **model.zip** - Trained model checkpoint
4. **training.log** - Python logging output
5. **events.log** - Event bus log

## Configuration Options

### Run Types

- `QUICK`: ~10,000 timesteps (5-10 minutes)
- `SHORT`: ~500,000 timesteps (1-2 hours)
- `MEDIUM`: ~2,000,000 timesteps (4-6 hours)
- `LONG`: ~10,000,000 timesteps (10+ hours)
- `CUSTOM`: User-defined timesteps

### Algorithms

- `PPO`: Proximal Policy Optimization (default)
- `A2C`: Advantage Actor-Critic
- `DQN`: Deep Q-Network

### GPU Usage

GPU is automatically detected and used if available. Set `use_gpu=False` to force CPU usage.

## Testing

Run unit tests:
```bash
pytest -m unit
```

Run integration tests:
```bash
pytest -m integration
```

Run all tests with coverage:
```bash
pytest --cov=retro_ml
```

## Phase 0 Exit Criteria

- ✅ ExperimentConfig with validation
- ✅ TrainingEngine with PPO support
- ✅ MetricEventBus with pub/sub
- ✅ Artifact writers (metrics, logs, models)
- ✅ Unit tests (>80% coverage)
- ✅ Integration tests (quick run)
- ✅ Same seed → same metrics (within tolerance)

## Next Steps (Phase 1)

- Desktop UI with CustomTkinter
- Training wizard (Simple Mode)
- Dashboard with live metrics
- Videos tab with management
- One-click installer

