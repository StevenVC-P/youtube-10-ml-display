# UI Migration Audit - tools/retro_ml_desktop/

## Overview

This document categorizes all files in `tools/retro_ml_desktop/` to identify:
1. **Duplicates** - Code that duplicates `retro_ml` package functionality (MIGRATE)
2. **UI Components** - CustomTkinter widgets and UI-specific code (KEEP)
3. **Utilities** - Helper functions and utilities (EVALUATE)

---

## 🔴 DUPLICATES - Migrate to retro_ml Package

These files duplicate functionality already in `retro_ml/` and should be replaced:

### 1. `experiment_manager.py` (505 lines)
**Status:** DUPLICATE - Replace with `retro_ml.ExperimentConfig`

**Current:**
- `ExperimentConfig` (dataclass) - Hyperparameter configuration
- `ExperimentLineage` - Lineage tracking
- `Experiment` - Experiment metadata
- `ExperimentManager` - CRUD operations

**Migration:**
- Replace `ExperimentConfig` with `retro_ml.ExperimentConfig` (Pydantic)
- Keep `ExperimentManager` but adapt to use `retro_ml.ExperimentConfig`
- Keep `Experiment` and `ExperimentLineage` (database models)
- Add video-specific fields to `retro_ml.ExperimentConfig`

### 2. `metric_event_bus.py` (300 lines)
**Status:** DUPLICATE - Replace with `retro_ml.MetricEventBus`

**Current:**
- `MetricEventBus` class with pub/sub pattern
- String-based event types (e.g., 'training.progress')
- Event history tracking

**Migration:**
- Replace with `retro_ml.MetricEventBus`
- Map string event types to `retro_ml.EventType` enum
- Update all subscribers to use new API

### 3. `process_manager.py` (1423 lines)
**Status:** PARTIAL DUPLICATE - Integrate with `retro_ml.TrainingEngine`

**Current:**
- Direct Python process management
- Training orchestration
- Log streaming
- Progress parsing

**Migration:**
- Use `retro_ml.run_experiment()` for training
- Keep process management for UI (subprocess handling, log streaming)
- Remove duplicate training logic
- Integrate with `retro_ml.MetricEventBus` for events

---

## 🟢 UI COMPONENTS - Keep (UI-Specific)

These files are UI-specific and should remain in `tools/retro_ml_desktop/`:

### Main Application
- `main.py` (609 lines) - Main application entry point
- `main_simple.py` - Simplified UI variant
- `launcher.py` - Application launcher

### Dashboard & Tabs
- `ml_dashboard.py` - Main dashboard UI
- `analytics_tab.py` - Analytics visualization tab
- `setup_wizard.py` - Training setup wizard

### Widgets
- `widgets/live_progress_widget.py` - Live progress display
- `widgets/recent_activity_widget.py` - Recent activity widget
- `widgets/resource_monitor_widget.py` - Resource monitoring widget

### UI Utilities
- `chart_annotations.py` - Chart annotation tools
- `chart_state.py` - Chart state management
- `close_confirmation_dialog.py` - Close confirmation dialog
- `enhanced_navigation.py` - Navigation enhancements
- `video_player.py` - Video playback widget

---

## 🟡 UTILITIES - Evaluate Case-by-Case

These files provide utility functions - evaluate whether to keep, migrate, or integrate:

### Keep in tools/retro_ml_desktop/
- `gpu_detector.py` - GPU detection (UI-specific, more detailed than retro_ml)
- `cuda_diagnostics.py` - CUDA diagnostics for UI
- `docker_manager.py` - Docker container management (UI feature)
- `monitor.py` - System monitoring for UI
- `config_manager.py` - UI configuration management
- `dependency_installer.py` - Dependency installation wizard

### Database & Storage
- `ml_database.py` - SQLite database for experiments (KEEP - UI persistence)
- `storage_cleaner.py` - Storage cleanup utilities (KEEP - UI feature)
- `ram_cleaner.py` - RAM cleanup utilities (KEEP - UI feature)

### Analysis & Export
- `ml_metrics.py` - Metrics calculation (EVALUATE - may overlap with retro_ml)
- `ml_plotting.py` - Plotting utilities (KEEP - UI-specific)
- `ml_collector.py` - Metrics collection (EVALUATE - may overlap)
- `enhanced_export.py` - Export functionality (KEEP - UI feature)
- `export_service.py` - Export service (KEEP - UI feature)
- `hyperparameter_analyzer.py` - Hyperparameter analysis (KEEP - UI feature)
- `statistical_analyzer.py` - Statistical analysis (KEEP - UI feature)
- `convergence_detector.py` - Convergence detection (KEEP - UI feature)
- `sample_efficiency.py` - Sample efficiency analysis (KEEP - UI feature)

### Resource Management
- `resource_selector.py` - Resource selection UI (KEEP - UI feature)

---

## Migration Priority

### Phase 1: Core Replacements (HIGH PRIORITY)
1. ✅ Replace `experiment_manager.ExperimentConfig` with `retro_ml.ExperimentConfig`
2. ✅ Replace `metric_event_bus.MetricEventBus` with `retro_ml.MetricEventBus`
3. ✅ Update `process_manager.py` to use `retro_ml.run_experiment()`

### Phase 2: Update UI Imports (MEDIUM PRIORITY)
4. Update `main.py` imports
5. Update `ml_dashboard.py` imports
6. Update `setup_wizard.py` imports
7. Update all widget imports

### Phase 3: Integration Testing (HIGH PRIORITY)
8. Create integration tests for UI + retro_ml
9. Test all UI functionality
10. Verify event bus integration

---

## API Mapping

### ExperimentConfig Migration

**OLD (tools/retro_ml_desktop/experiment_manager.py):**
```python
from tools.retro_ml_desktop.experiment_manager import ExperimentConfig

config = ExperimentConfig(
    algorithm="PPO",
    total_timesteps=1_000_000,
    learning_rate=3e-4,
    # ... many hyperparameters
)
```

**NEW (retro_ml package):**
```python
from retro_ml import ExperimentConfig, RunType

config = ExperimentConfig(
    game_id="BreakoutNoFrameskip-v4",
    algorithm="PPO",
    run_type=RunType.SHORT,
    seed=42,
    # Hyperparameters with defaults
    learning_rate=3e-4
)
```

### MetricEventBus Migration

**OLD:**
```python
from tools.retro_ml_desktop.metric_event_bus import get_event_bus

bus = get_event_bus()
bus.subscribe('training.progress', callback)
bus.publish('training.progress', {'progress_pct': 50})
```

**NEW:**
```python
from retro_ml import MetricEventBus, MetricEvent, EventType

bus = MetricEventBus()
bus.subscribe(EventType.TRAINING_STEP, callback)
bus.emit(MetricEvent(
    event_type=EventType.TRAINING_STEP,
    experiment_id="exp_123",
    data={'progress_pct': 50}
))
```

### Training Execution Migration

**OLD:**
```python
from tools.retro_ml_desktop.process_manager import ProcessManager

manager = ProcessManager(project_root)
process_id = manager.create_process(name, game, algorithm, preset)
manager.start_process(process_id)
```

**NEW:**
```python
from retro_ml import ExperimentConfig, run_experiment, RunType

config = ExperimentConfig(
    game_id=game,
    algorithm=algorithm,
    run_type=RunType.SHORT
)
result = run_experiment(config)
```

---

## Next Steps

1. ✅ Complete this audit
2. Extend `retro_ml.ExperimentConfig` with video-specific fields
3. Create adapter layer for backward compatibility
4. Update imports file-by-file
5. Add integration tests
6. Verify UI functionality


