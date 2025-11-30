# Retro ML Desktop — Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for Retro ML Desktop across all development phases. Testing is organized into layers, with each phase deepening coverage while reusing the same structure.

---

## Core Test Types

### 1. Unit Tests (Engine)

**Target:** Training pipeline, config parsing, metric bus, artifact writers  
**Tools:** pytest, pytest-cov, pytest-mock  
**Characteristics:**

- Use mock or tiny Atari environments
- Very short runs (100-1000 steps)
- Deterministic seeds
- Fast execution (< 1 second per test)
- High coverage target (>80%)

**Example Test Areas:**

- `ExperimentConfig` validation and serialization
- `MetricEventBus` pub/sub functionality
- `ArtifactManager` file operations
- `TrainingPipeline` orchestration logic
- Algorithm parameter validation

### 2. Integration Tests (End-to-End Headless)

**Target:** "Run experiment from config → artifacts on disk"  
**Tools:** pytest, pytest-timeout  
**Characteristics:**

- Inputs: experiment JSON/spec
- Outputs: metrics file, model, video stub, logs
- Run in CI on CPU-only config with short episodes
- Medium execution time (< 5 minutes per test)
- Validates entire pipeline without UI

**Example Test Scenarios:**

- `test_quick_run_produces_artifacts()`
- `test_experiment_with_custom_config()`
- `test_training_resume_from_checkpoint()`
- `test_multi_seed_run()`

### 3. UI Smoke Tests

**Target:** Main windows open, key flows don't crash  
**Tools:** pytest, pytest-qt (for CustomTkinter), or manual scripts  
**Characteristics:**

- Minimal automation initially
- Script that starts app, opens main views, triggers Quick Run
- Validates UI doesn't crash on basic operations
- Can be manual checklist until automation is mature

**Example Test Scenarios:**

- App launches without errors
- Dashboard loads and displays default state
- Training wizard opens and accepts input
- Videos tab displays and allows operations
- Settings panel saves preferences

### 4. Installer / VM Tests

**Target:** Installation and first-run experience  
**Tools:** VirtualBox, PowerShell scripts, manual testing  
**Characteristics:**

- Run installer in clean Windows VM
- Validate installation, launch, and Quick Run
- Eventually script with VM snapshots
- Manual first, automated later

**Validation Checklist:**

- ✅ Installer runs without errors
- ✅ App launches after installation
- ✅ PyTorch downloads on first run (if hybrid installer)
- ✅ Quick Run completes successfully
- ✅ Artifacts appear in expected folder
- ✅ GPU detected correctly (if available)

### 5. Performance Smoke Tests

**Target:** Resource usage and responsiveness  
**Tools:** pytest, memory_profiler, psutil  
**Characteristics:**

- Short "stress" scenarios
- Run several Quick Runs back-to-back
- Monitor memory, CPU, GPU usage
- Ensure app stays responsive

**Example Test Scenarios:**

- `test_memory_no_unbounded_growth()`
- `test_concurrent_training_sessions()`
- `test_app_responsiveness_during_training()`
- `test_gpu_memory_cleanup_after_training()`

### 6. Regression Snapshot Tests

**Target:** Prevent unintended changes to outputs  
**Tools:** pytest, pytest-regressions  
**Characteristics:**

- Keep "golden" metrics outputs and config files
- After changes, run same configs → compare key values within tolerance
- Detect breaking changes early

**Example Test Scenarios:**

- `test_metrics_match_golden_snapshot()`
- `test_config_schema_backward_compatible()`
- `test_model_output_deterministic()`

---

## Phase-by-Phase Testing Requirements

### Phase 0 – Core Engine Stabilization

#### Required Unit Tests

- ✅ `test_experiment_config_validation()`
- ✅ `test_experiment_config_serialization()`
- ✅ `test_training_pipeline_initialization()`
- ✅ `test_metric_bus_emit_and_subscribe()`
- ✅ `test_artifact_writer_creates_files()`
- ✅ `test_artifact_writer_handles_errors()`
- ✅ `test_video_generator_short_run()`
- ✅ `test_experiment_id_uniqueness()`

#### Required Integration Tests

- ✅ `test_run_quick_experiment_produces_artifacts()`
- ✅ `test_artifacts_schema_valid()`
- ✅ `test_same_seed_produces_same_metrics()`
- ✅ `test_training_pipeline_end_to_end()`

#### Required Regression Tests

- ✅ Metrics schema snapshot
- ✅ Config schema snapshot
- ✅ Deterministic output snapshot (same seed)

#### Coverage Target

- **Minimum:** 80% line coverage
- **Target:** 90% line coverage for core modules

---

### Phase 1 – Tier 1 "Hobbyist" Release

#### Additional Unit Tests

- ✅ `test_algorithm_selection_ppo()`
- ✅ `test_algorithm_selection_a2c()`
- ✅ `test_algorithm_selection_dqn()`
- ✅ `test_hyperparameter_validation()`
- ✅ `test_hyperparameter_safe_ranges()`
- ✅ `test_seed_control_manual()`
- ✅ `test_seed_control_random_generation()`
- ✅ `test_model_save_and_load()`
- ✅ `test_config_export_import()`

#### Additional Integration Tests

- ✅ `test_export_metrics_csv()`
- ✅ `test_export_metrics_json()`
- ✅ `test_export_metrics_png()`
- ✅ `test_reimport_exported_metrics_in_python()`
- ✅ `test_algorithm_comparison_side_by_side()`

#### Required Regression Tests

- ✅ Same config + seed → compare reward curve vs golden version
- ✅ Hyperparameter changes produce expected behavior differences
- ✅ Exported CSV/JSON schema matches specification

#### Additional UI Tests

- ✅ Hyperparameter panel validation
- ✅ Model browser displays metadata
- ✅ Onboarding tutorial completion
- ✅ Algorithm selection updates UI correctly

#### Coverage Target

- **Minimum:** 85% line coverage
- **Target:** 92% line coverage for core modules

---

### Phase 3 – Tier 3 "Research"

#### Additional Unit Tests

- ✅ `test_deterministic_seed_control()`
- ✅ `test_batch_queue_manager_add()`
- ✅ `test_batch_queue_manager_priority()`
- ✅ `test_batch_queue_manager_pause_resume()`
- ✅ `test_checkpoint_versioning()`
- ✅ `test_checkpoint_resume()`
- ✅ `test_checkpoint_comparison()`
- ✅ `test_dataset_export_hdf5()`
- ✅ `test_dataset_export_parquet()`
- ✅ `test_dataset_schema_validation()`
- ✅ `test_experiment_lineage_graph()`
- ✅ `test_multi_seed_aggregation()`

#### Additional Integration Tests

- ✅ `test_multi_config_sweep()`
- ✅ `test_grid_search_hyperparameters()`
- ✅ `test_dataset_export_and_external_load()`
- ✅ `test_checkpoint_resume_produces_identical_results()`
- ✅ `test_experiment_queue_handles_10_experiments()`

#### Required Regression Tests

- ✅ Checkpoint resume produces identical results
- ✅ Multi-seed runs show proper statistical aggregation
- ✅ Dataset export schema validation
- ✅ Deterministic mode produces bit-exact results (when possible)

#### Additional UI Tests

- ✅ Compare-experiments view loads multiple runs
- ✅ Checkpoint browser displays timeline
- ✅ Dataset browser shows statistics
- ✅ Visualization panel renders charts correctly

#### Coverage Target

- **Minimum:** 88% line coverage
- **Target:** 94% line coverage for core modules

---

### Phase 4 – Tier 4 "Enterprise/Institution"

#### Additional Unit Tests

- ✅ `test_user_authentication()`
- ✅ `test_role_based_permissions_admin()`
- ✅ `test_role_based_permissions_researcher()`
- ✅ `test_role_based_permissions_viewer()`
- ✅ `test_license_enforcement_floating()`
- ✅ `test_license_enforcement_node_locked()`
- ✅ `test_license_offline_activation()`
- ✅ `test_experiment_db_repository_crud()`
- ✅ `test_experiment_db_search()`
- ✅ `test_audit_log_capture()`
- ✅ `test_data_retention_policy()`

#### Additional Integration Tests

- ✅ `test_multi_user_concurrent_access()`
- ✅ `test_distributed_run_simulation()`
- ✅ `test_experiment_sharing_permissions()`
- ✅ `test_export_pack_creation()`
- ✅ `test_backup_and_restore()`

#### Required Security Tests

- ✅ User with limited role cannot see other users' private experiments
- ✅ User with limited role cannot modify other users' experiments
- ✅ Audit log captures all critical actions
- ✅ Data isolation between users/projects
- ✅ Encryption at rest and in transit

#### Required Load Tests

- ✅ 100+ concurrent users
- ✅ 1000+ experiments in database
- ✅ Search performance with large dataset
- ✅ Distributed training job scheduling

#### Required Compliance Tests

- ✅ Audit log completeness
- ✅ Data retention policy enforcement
- ✅ Export controls work correctly
- ✅ License compliance reporting

#### Coverage Target

- **Minimum:** 90% line coverage
- **Target:** 95% line coverage for core modules

---

## Test Infrastructure

### Directory Structure

```
tests/
├── unit/
│   ├── test_experiment_config.py
│   ├── test_training_pipeline.py
│   ├── test_metric_bus.py
│   ├── test_artifact_manager.py
│   ├── test_video_generator.py
│   └── ...
├── integration/
│   ├── test_quick_run.py
│   ├── test_full_training_flow.py
│   ├── test_export_import.py
│   └── ...
├── ui/
│   ├── test_dashboard.py
│   ├── test_training_wizard.py
│   ├── test_videos_tab.py
│   └── ...
├── regression/
│   ├── test_metrics_snapshots.py
│   ├── test_config_snapshots.py
│   └── ...
├── performance/
│   ├── test_memory_usage.py
│   ├── test_concurrent_sessions.py
│   └── ...
├── fixtures/
│   ├── configs/
│   ├── golden_outputs/
│   └── mock_data/
└── conftest.py
```

### pytest Configuration (pytest.ini)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --verbose
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
    --timeout=300
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (slower, end-to-end)
    ui: UI tests (require display)
    slow: Slow tests (> 1 minute)
    gpu: Tests requiring GPU
    regression: Regression snapshot tests
    performance: Performance and load tests
```

### Continuous Integration (CI)

#### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements-dev.txt
      - run: pytest -m unit

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements-dev.txt
      - run: pytest -m integration

  regression-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements-dev.txt
      - run: pytest -m regression
```

---

## Testing Best Practices

### 1. Test Naming Convention

- Use descriptive names: `test_<component>_<scenario>_<expected_result>()`
- Example: `test_experiment_config_invalid_seed_raises_error()`

### 2. Test Organization

- One test file per module/component
- Group related tests in classes
- Use fixtures for common setup/teardown

### 3. Test Data Management

- Store test configs in `tests/fixtures/configs/`
- Store golden outputs in `tests/fixtures/golden_outputs/`
- Use `pytest.fixture` for reusable test data

### 4. Mocking Strategy

- Mock external dependencies (GPU, file system when appropriate)
- Use real implementations for integration tests
- Mock time-consuming operations in unit tests

### 5. Assertion Strategy

- Use specific assertions (`assert x == y`, not `assert x`)
- Include helpful error messages
- Use `pytest.approx()` for floating-point comparisons

### 6. Test Isolation

- Each test should be independent
- Clean up resources in teardown
- Use temporary directories for file operations

### 7. Performance Considerations

- Keep unit tests fast (< 1 second)
- Mark slow tests with `@pytest.mark.slow`
- Run slow tests separately in CI

---

## Test Execution

### Run All Tests

```bash
pytest
```

### Run Specific Test Types

```bash
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests
pytest -m gpu           # GPU tests only (requires GPU)
```

### Run with Coverage

```bash
pytest --cov=src --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/unit/test_experiment_config.py
```

### Run Specific Test Function

```bash
pytest tests/unit/test_experiment_config.py::test_config_validation
```

### Run in Parallel (faster)

```bash
pytest -n auto  # Requires pytest-xdist
```

---

## Debugging Failed Tests

### Verbose Output

```bash
pytest -vv
```

### Show Print Statements

```bash
pytest -s
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Run Last Failed Tests Only

```bash
pytest --lf
```

---

## Test Metrics and Reporting

### Coverage Reports

- HTML report: `htmlcov/index.html`
- Terminal report: Shows missing lines
- Fail build if coverage < 80%

### Test Execution Time

- Track slow tests
- Optimize or mark as `@pytest.mark.slow`
- Monitor CI execution time

### Flaky Test Detection

- Re-run failed tests automatically
- Track flaky tests in separate report
- Fix or quarantine flaky tests

---

## Next Steps

1. **Immediate:**

   - Set up pytest infrastructure
   - Create `conftest.py` with common fixtures
   - Implement first unit tests for `ExperimentConfig`

2. **Short-term:**

   - Implement integration test framework
   - Set up CI pipeline
   - Create test data fixtures

3. **Medium-term:**

   - Implement UI test framework
   - Create installer test scripts
   - Set up performance monitoring

4. **Long-term:**
   - Expand test coverage to all modules
   - Implement automated regression testing
   - Create comprehensive test documentation

---

### Phase 2 – Tier 2 "Student/Education"

#### Additional Unit Tests
