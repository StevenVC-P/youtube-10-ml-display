# Test Matrix — Pytest File Generation Guide

## Overview

This document provides a minimal test matrix table to guide the generation of actual pytest files. Each row represents a test file to be created, with columns indicating the module under test, test type, priority, and key test cases.

---

## Phase 0 — Core Engine Stabilization

### Test Matrix

| Test File | Module Under Test | Type | Priority | Key Test Cases | Estimated Tests |
|-----------|------------------|------|----------|----------------|-----------------|
| `test_experiment_config.py` | `src/core/experiment_config.py` | Unit | P0 | Validation, serialization, deserialization, defaults, invalid inputs | 15-20 |
| `test_training_pipeline.py` | `src/core/training_pipeline.py` | Unit | P0 | Initialization, start/stop, pause/resume, error handling | 12-15 |
| `test_metric_bus.py` | `src/core/metric_bus.py` | Unit | P0 | Emit, subscribe, unsubscribe, multiple subscribers, event types | 10-12 |
| `test_artifact_manager.py` | `src/core/artifact_manager.py` | Unit | P0 | Create artifacts, link to experiment, cleanup, error handling | 10-12 |
| `test_video_generator.py` | `src/core/video_generator.py` | Unit | P0 | Short run, quality settings, naming, error handling | 8-10 |
| `test_quick_run.py` | End-to-end | Integration | P0 | Quick run produces all artifacts, schema validation, determinism | 5-8 |
| `test_metrics_snapshots.py` | Metrics output | Regression | P1 | Same seed → same metrics, schema stability | 3-5 |

**Total Estimated Tests:** 63-82

---

## Phase 1 — Tier 1 "Hobbyist" Release

### Test Matrix

| Test File | Module Under Test | Type | Priority | Key Test Cases | Estimated Tests |
|-----------|------------------|------|----------|----------------|-----------------|
| `test_training_presets.py` | `src/core/presets.py` | Unit | P0 | Short/medium/long presets, validation, time conversion | 8-10 |
| `test_gpu_detection.py` | `src/core/gpu_utils.py` | Unit | P0 | Detect GPU, fallback to CPU, VRAM check, multi-GPU | 6-8 |
| `test_video_naming.py` | `src/core/video_generator.py` | Unit | P1 | Auto-naming logic, collision handling, special characters | 5-7 |
| `test_artifact_linking.py` | `src/core/artifact_manager.py` | Unit | P1 | Link models/videos/logs to experiment ID, retrieval | 5-7 |
| `test_full_training_flow.py` | End-to-end | Integration | P0 | Full flow with video, pause/resume, artifact verification | 5-8 |
| `test_dashboard.py` | `src/ui/dashboard.py` | UI | P1 | Dashboard loads, displays metrics, collapsible sections | 5-7 |
| `test_training_wizard.py` | `src/ui/training_wizard.py` | UI | P0 | Wizard opens, accepts input, starts training, validation | 8-10 |
| `test_videos_tab.py` | `src/ui/videos_tab.py` | UI | P1 | Display videos, rename, delete, open, sort | 6-8 |
| `test_installer.py` | Installer | System | P0 | Install on clean VM, first run, PyTorch download, GPU detect | 4-6 |

**Total Estimated Tests:** 52-71  
**Cumulative Total:** 115-153

---

## Phase 2 — Tier 2 "Student/Education"

### Test Matrix

| Test File | Module Under Test | Type | Priority | Key Test Cases | Estimated Tests |
|-----------|------------------|------|----------|----------------|-----------------|
| `test_algorithm_ppo.py` | `src/algorithms/ppo.py` | Unit | P0 | Initialization, training step, hyperparameters, validation | 10-12 |
| `test_algorithm_a2c.py` | `src/algorithms/a2c.py` | Unit | P0 | Initialization, training step, hyperparameters, validation | 10-12 |
| `test_algorithm_dqn.py` | `src/algorithms/dqn.py` | Unit | P0 | Initialization, training step, hyperparameters, validation | 10-12 |
| `test_hyperparameter_validation.py` | `src/core/hyperparameters.py` | Unit | P0 | Safe ranges, validation, defaults, algorithm-specific | 12-15 |
| `test_seed_control.py` | `src/core/seed_manager.py` | Unit | P0 | Manual seed, random generation, determinism, display | 8-10 |
| `test_model_save_load.py` | `src/core/model_manager.py` | Unit | P0 | Save model, load model, continue training, metadata | 10-12 |
| `test_config_export_import.py` | `src/core/experiment_config.py` | Unit | P1 | Export to JSON, import from JSON, validation | 6-8 |
| `test_export_metrics_csv.py` | `src/export/csv_exporter.py` | Unit | P0 | Export to CSV, schema, Excel compatibility | 5-7 |
| `test_export_metrics_json.py` | `src/export/json_exporter.py` | Unit | P0 | Export to JSON, schema, metadata inclusion | 5-7 |
| `test_export_metrics_png.py` | `src/export/png_exporter.py` | Unit | P1 | Export charts to PNG, quality, resolution | 4-6 |
| `test_algorithm_comparison.py` | End-to-end | Integration | P1 | Side-by-side comparison, overlay charts, diff configs | 5-7 |
| `test_reimport_metrics.py` | End-to-end | Integration | P0 | Export → reimport in Python, schema validation | 3-5 |
| `test_reproducibility.py` | End-to-end | Regression | P0 | Same config + seed → similar curves, tolerance check | 5-7 |
| `test_hyperparameter_panel.py` | `src/ui/hyperparameter_panel.py` | UI | P0 | Display params, validation, tooltips, reset to defaults | 8-10 |
| `test_model_browser.py` | `src/ui/model_browser.py` | UI | P1 | Display models, metadata, load/delete operations | 6-8 |
| `test_onboarding.py` | `src/ui/onboarding.py` | UI | P1 | Tutorial flow, tooltips, sample experiment | 5-7 |

**Total Estimated Tests:** 112-145  
**Cumulative Total:** 227-298

---

## Phase 3 — Tier 3 "Research"

### Test Matrix

| Test File | Module Under Test | Type | Priority | Key Test Cases | Estimated Tests |
|-----------|------------------|------|----------|----------------|-----------------|
| `test_deterministic_pipeline.py` | `src/core/deterministic.py` | Unit | P0 | Seed control, CUDA determinism, verification | 10-12 |
| `test_batch_queue_manager.py` | `src/core/queue_manager.py` | Unit | P0 | Add to queue, priority, pause/resume, scheduling | 12-15 |
| `test_checkpoint_manager.py` | `src/core/checkpoint_manager.py` | Unit | P0 | Save checkpoint, versioning, tagging, cleanup | 12-15 |
| `test_checkpoint_resume.py` | `src/core/checkpoint_manager.py` | Unit | P0 | Resume from checkpoint, identical results, comparison | 10-12 |
| `test_dataset_export_hdf5.py` | `src/export/dataset_exporter.py` | Unit | P0 | Export to HDF5, schema, SARS tuples | 8-10 |
| `test_dataset_export_parquet.py` | `src/export/dataset_exporter.py` | Unit | P1 | Export to Parquet, schema, compression | 6-8 |
| `test_dataset_schema.py` | `src/export/dataset_exporter.py` | Unit | P0 | Schema validation, quality checks, statistics | 8-10 |
| `test_experiment_lineage.py` | `src/core/lineage_graph.py` | Unit | P1 | Parent/child relationships, graph traversal | 8-10 |
| `test_multi_seed_aggregation.py` | `src/core/multi_seed.py` | Unit | P0 | Run multiple seeds, aggregate stats, variance | 10-12 |
| `test_advanced_metrics.py` | `src/core/metrics.py` | Unit | P0 | Entropy, KL divergence, loss curves, FPS, gradients | 15-18 |
| `test_multi_config_sweep.py` | End-to-end | Integration | P0 | Grid search, random search, sweep templates | 8-10 |
| `test_dataset_external_load.py` | End-to-end | Integration | P0 | Export dataset → load in external Python script | 4-6 |
| `test_checkpoint_resume_e2e.py` | End-to-end | Integration | P0 | Resume from checkpoint → identical results | 5-7 |
| `test_queue_10_experiments.py` | End-to-end | Integration | P0 | Queue 10 experiments, resource scheduling, completion | 3-5 |
| `test_checkpoint_regression.py` | Checkpoint output | Regression | P0 | Checkpoint resume produces identical results | 4-6 |
| `test_multi_seed_regression.py` | Multi-seed output | Regression | P0 | Multi-seed aggregation matches expected stats | 4-6 |
| `test_compare_experiments_ui.py` | `src/ui/compare_view.py` | UI | P0 | Select experiments, overlay charts, diff configs | 8-10 |
| `test_checkpoint_browser_ui.py` | `src/ui/checkpoint_browser.py` | UI | P1 | Display checkpoints, timeline, operations | 6-8 |
| `test_dataset_browser_ui.py` | `src/ui/dataset_browser.py` | UI | P1 | Display datasets, statistics, quality checks | 6-8 |
| `test_visualization_panel.py` | `src/ui/visualization_panel.py` | UI | P0 | Interactive charts, diagnostics, export | 10-12 |

**Total Estimated Tests:** 157-200  
**Cumulative Total:** 384-498

---

## Phase 4 — Tier 4 "Enterprise/Institution"

### Test Matrix

| Test File | Module Under Test | Type | Priority | Key Test Cases | Estimated Tests |
|-----------|------------------|------|----------|----------------|-----------------|
| `test_user_authentication.py` | `src/auth/authentication.py` | Unit | P0 | Login, logout, session management, SSO | 12-15 |
| `test_role_permissions.py` | `src/auth/permissions.py` | Unit | P0 | Admin/researcher/viewer roles, access control | 15-18 |
| `test_license_enforcement.py` | `src/license/enforcement.py` | Unit | P0 | Floating licenses, node-locked, offline activation | 12-15 |
| `test_experiment_db_repository.py` | `src/db/experiment_repository.py` | Unit | P0 | CRUD operations, search, filter, tags | 15-18 |
| `test_audit_log.py` | `src/audit/audit_log.py` | Unit | P0 | Capture actions, query logs, compliance | 10-12 |
| `test_data_retention.py` | `src/db/retention_policy.py` | Unit | P1 | Retention policies, cleanup, archival | 8-10 |
| `test_multi_user_access.py` | End-to-end | Integration | P0 | Concurrent users, data isolation, conflicts | 10-12 |
| `test_distributed_run.py` | End-to-end | Integration | P0 | Multi-GPU/multi-node simulation, job distribution | 8-10 |
| `test_experiment_sharing.py` | End-to-end | Integration | P0 | Share experiments, permissions, collaboration | 8-10 |
| `test_export_pack.py` | End-to-end | Integration | P0 | Create export pack, reproducibility bundle | 6-8 |
| `test_backup_restore.py` | End-to-end | Integration | P1 | Backup database, restore, integrity check | 5-7 |
| `test_security_permissions.py` | Security | Security | P0 | Role enforcement, data isolation, access control | 12-15 |
| `test_audit_completeness.py` | Security | Security | P0 | All critical actions logged, query audit trail | 8-10 |
| `test_encryption.py` | Security | Security | P1 | Encryption at rest, in transit, key management | 6-8 |
| `test_load_100_users.py` | System | Load | P0 | 100+ concurrent users, performance, stability | 5-7 |
| `test_load_1000_experiments.py` | System | Load | P0 | 1000+ experiments, search performance, scaling | 5-7 |
| `test_compliance_audit.py` | Compliance | Compliance | P0 | Audit log completeness, retention enforcement | 6-8 |
| `test_license_compliance.py` | Compliance | Compliance | P0 | License limits enforced, reporting accurate | 6-8 |

**Total Estimated Tests:** 157-198  
**Cumulative Total:** 541-696

---

## Test File Template

### Unit Test Template

```python
"""
Unit tests for [Module Name]

Tests cover:
- [Test area 1]
- [Test area 2]
- [Test area 3]
"""

import pytest
from unittest.mock import Mock, patch
from src.core.module_name import ClassName


class TestClassName:
    """Test suite for ClassName"""
    
    @pytest.fixture
    def instance(self):
        """Create a test instance"""
        return ClassName()
    
    def test_initialization(self, instance):
        """Test that instance initializes correctly"""
        assert instance is not None
        assert instance.property == expected_value
    
    def test_method_with_valid_input(self, instance):
        """Test method with valid input"""
        result = instance.method(valid_input)
        assert result == expected_output
    
    def test_method_with_invalid_input(self, instance):
        """Test method raises error with invalid input"""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
    
    @pytest.mark.parametrize("input,expected", [
        (input1, output1),
        (input2, output2),
        (input3, output3),
    ])
    def test_method_parametrized(self, instance, input, expected):
        """Test method with multiple inputs"""
        result = instance.method(input)
        assert result == expected
```

### Integration Test Template

```python
"""
Integration tests for [Feature Name]

Tests cover end-to-end workflows:
- [Workflow 1]
- [Workflow 2]
"""

import pytest
import tempfile
from pathlib import Path
from src.core.experiment_config import ExperimentConfig
from src.core.training_pipeline import TrainingPipeline


@pytest.mark.integration
class TestFeatureIntegration:
    """Integration tests for feature"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Create test experiment config"""
        return ExperimentConfig(
            game="BreakoutNoFrameskip-v4",
            duration_hours=0.01,  # Very short for testing
            output_dir=temp_dir,
        )
    
    def test_end_to_end_workflow(self, config, temp_dir):
        """Test complete workflow from config to artifacts"""
        # Run training
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        # Verify artifacts created
        assert (temp_dir / "metrics.json").exists()
        assert (temp_dir / "model.pth").exists()
        assert (temp_dir / "video.mp4").exists()
        
        # Verify artifact contents
        metrics = load_metrics(temp_dir / "metrics.json")
        assert "reward" in metrics
        assert "timesteps" in metrics
```

---

## Priority Definitions

- **P0 (Critical):** Must pass before phase completion. Blocks release.
- **P1 (High):** Should pass before phase completion. Can be deferred with justification.
- **P2 (Medium):** Nice to have. Can be deferred to next phase.
- **P3 (Low):** Future enhancement. Not required for current phase.

---

## Test Execution Strategy

### Phase 0
1. Implement all P0 unit tests first
2. Implement P0 integration tests
3. Implement P1 regression tests
4. Achieve 80%+ coverage before moving to Phase 1

### Phase 1
1. Implement all P0 tests (unit + integration + UI)
2. Implement P1 tests
3. Run installer tests manually
4. Achieve 85%+ coverage before moving to Phase 2

### Phase 2
1. Implement all P0 tests
2. Implement P1 tests
3. Run regression tests against golden outputs
4. Achieve 88%+ coverage before moving to Phase 3

### Phase 3
1. Implement all P0 tests
2. Implement P1 tests
3. Run performance and regression tests
4. Achieve 90%+ coverage before moving to Phase 4

### Phase 4
1. Implement all P0 tests
2. Implement security and load tests
3. Run compliance tests
4. Achieve 92%+ coverage before release

---

## Next Steps

1. **Create test infrastructure:**
   - Set up `tests/` directory structure
   - Create `conftest.py` with common fixtures
   - Configure `pytest.ini`

2. **Generate Phase 0 test files:**
   - Use templates above
   - Start with `test_experiment_config.py`
   - Implement one test file at a time

3. **Run tests continuously:**
   - Run tests after each implementation
   - Fix failures immediately
   - Track coverage metrics

4. **Iterate and improve:**
   - Add more test cases as bugs are found
   - Refactor tests for clarity
   - Update matrix as requirements change
