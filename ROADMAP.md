# Retro ML Desktop — Development Roadmap

## Overview

This roadmap outlines the phased development approach for Retro ML Desktop, aligned with the four-tier product vision. Each phase has clear scope, deliverables, and exit criteria.

---

## Phase 0 – Core Engine Stabilization ✅ COMPLETE

**Timeline:** ✅ Completed 2025-11-29
**Focus:** Backend-only, no UI dependencies
**Status:** All exit criteria met, 98% test coverage

### Scope

#### Core Components

- **Experiment config object** — Single source of truth for all experiment parameters
- **Clean training pipeline** — `env → algorithm → metrics → artifacts`
- **Metric event bus** — Wired end-to-end with pub/sub pattern
- **Video generator** — Stable on short runs (1-2 hours)
- **Experiment ID system** — Unique run IDs with artifact linking (models, logs, videos)

#### Key Deliverables

1. `ExperimentConfig` class with validation
2. `TrainingPipeline` orchestrator
3. `MetricEventBus` with core event types
4. `VideoGenerator` with configurable quality/length
5. `ArtifactManager` for organizing outputs

### Exit Criteria

✅ Can run a "Quick Run" (short training) from Python CLI only  
✅ Produces: `metrics.json`, `model.pth`, `video.mp4`  
✅ Same seed → same metrics within tolerance (±5%)  
✅ All artifacts properly linked to experiment ID  
✅ No memory leaks in 10-hour training runs

### Testing Requirements

- Unit tests for all core components (>80% coverage)
- Integration test: `run_quick_experiment(config)` → validate artifacts
- Regression snapshots for metrics schema
- Memory profiling on extended runs

---

## Phase 1 – Tier 1 "Hobbyist" Release 🚀 IN PROGRESS

**Timeline:** 4-8 weeks (Started 2025-11-30)
**Focus:** Minimal viable desktop application
**Status:** Sprint 1 starting - Dashboard & UI Polish
**Branch:** `feature/phase1-hobbyist-release`
**Planning Doc:** `docs/PHASE1_HOBBYIST_RELEASE.md`

### Scope

#### UI Components

- **Dashboard** (minimal) with collapsible sections:
  - Training Controls
  - Live Metrics Chart
  - GPU Status
  - Recent Activity
- **Training Wizard** (Simple Mode):
  - Game selection
  - Preset selection (Short/Medium/Long)
  - Fillable time inputs (hours/minutes)
  - Start/Stop/Pause controls
- **Tabs:**
  - Dashboard (main view)
  - Experiments (read-only list)
  - Videos (manage generated videos)
  - Settings (basic preferences)

#### Features

- **One-click training presets:**
  - Short: 1-2 hours
  - Medium: 4-6 hours
  - Long: 10+ hours
- **Automatic video creation** after training completes
- **Videos tab management:**
  - Thumbnail previews
  - Rename/Delete/Open operations
  - Sort by date/name/size
- **GPU detection and utilization display**

### Exit Criteria

✅ User can: install → run Quick Start → see training progress → watch video  
✅ No crashes in normal flow on clean Windows VM  
✅ Logs written for every experiment (config + run metadata)  
✅ Videos auto-generated with meaningful names  
✅ GPU properly detected and utilized (NVIDIA only)  
✅ Installer size < 100MB (downloads PyTorch on first run)

### Testing Requirements

- UI smoke tests (app starts, dashboard loads, training wizard works)
- Installer test in VirtualBox (fresh Windows VM)
- Manual test script for Videos tab operations
- End-to-end test: install → train → video → cleanup

---

## Phase 2 – Tier 2 "Student/Education"

**Timeline:** 8-12 weeks  
**Focus:** Educational features and reproducibility

### Scope

#### Algorithm Support

- **Multiple algorithms:**
  - PPO (Proximal Policy Optimization)
  - A2C (Advantage Actor-Critic)
  - DQN (Deep Q-Network)
- **Algorithm-specific presets** (tuned defaults)
- **Algorithm comparison mode** (side-by-side)

#### Hyperparameter Control

- **Simple hyperparameter panel:**
  - Learning rate
  - Batch size
  - Discount factor (gamma)
  - Entropy coefficient
  - Number of environments
- **Tooltips and explanations** for each parameter
- **Validation and safe ranges**
- **Reset to defaults** button

#### Reproducibility

- **Seed control:**
  - Manual seed entry
  - Random seed generation
  - Seed displayed in results
- **Deterministic mode toggle**
- **Config export/import**

#### Data Export

- **Export metrics to:**
  - CSV (timestep, reward, loss, etc.)
  - JSON (full experiment metadata)
  - PNG (reward curves, loss plots)
- **Batch export** (multiple experiments)

#### Model Management

- **Model save/load:**
  - Save trained models
  - Load and continue training
  - Load for evaluation only
- **Model browser** with metadata

#### Onboarding

- **First-run tutorial:**
  - "Train your first model" walkthrough
  - Interactive tooltips
  - Sample experiment with known-good results
- **Help documentation** (embedded)

### Exit Criteria

✅ Instructor can define experiment, run it, export metrics, re-run with same seed  
✅ Two runs with same config + seed → similar curves (±10% variance)  
✅ Exported files open cleanly in Excel / Python  
✅ All three algorithms produce valid results  
✅ Hyperparameter changes reflected in training behavior  
✅ Onboarding tutorial completion rate > 70%

### Testing Requirements

- Unit tests for algorithm selection, hyperparameter validation, seed handling
- Integration tests: export metrics → re-import in Python script
- Regression tests: same config + seed → compare vs golden version
- UI tests: hyperparameter panel, model browser

---

## Phase 3 – Tier 3 "Research"

**Timeline:** 12-20 weeks  
**Focus:** Advanced experimentation and reproducibility

### Scope

#### Deterministic Training

- **Fully deterministic pipeline:**
  - Controlled random seeds (PyTorch, NumPy, environment)
  - Deterministic CUDA operations
  - Reproducibility verification tools
- **Environment versioning** (lock Gym/Gymnasium versions)
- **Dependency snapshot** (exact package versions)

#### Batch Experimentation

- **Multi-seed runs:**
  - Run same config with N different seeds
  - Automatic seed generation
  - Aggregate statistics (mean, std, min, max)
- **Experiment sweeps:**
  - Grid search over hyperparameters
  - Random search support
  - Sweep templates
- **Experiment queue:**
  - Add multiple experiments to queue
  - Priority ordering
  - Pause/resume queue
  - Resource-aware scheduling

#### Advanced Metrics

- **Comprehensive tracking:**
  - Policy entropy (exploration measure)
  - KL divergence (policy change magnitude)
  - Loss curves (policy, value, total)
  - FPS (frames per second)
  - Update statistics (gradient norms, parameter changes)
  - Value function accuracy
  - Advantage estimates
- **Custom metric definitions**
- **TensorBoard export**

#### Checkpoint Management

- **Versioned checkpoints:**
  - Save at regular intervals
  - Tag important checkpoints
  - Checkpoint browser with timeline
- **Checkpoint operations:**
  - Resume from any checkpoint
  - Compare checkpoints
  - Export checkpoint bundles
- **Automatic cleanup** (configurable retention)

#### Offline RL Support

- **Dataset export:**
  - Replay buffer dumps
  - SARS tuples (State, Action, Reward, Next State)
  - Episode trajectories
  - Format options (HDF5, Parquet, custom)
- **Dataset browser** with statistics

#### Visualization

- **Interactive charts:**
  - Reward curves with confidence intervals
  - Loss landscapes
  - Hyperparameter sensitivity plots
- **Diagnostic tools:**
  - Gradient flow visualization
  - Activation distributions
  - Weight histograms
- **Export visualizations** (PNG, SVG, HTML)

#### Experiment Comparison

- **Side-by-side comparison:**
  - Select 2+ experiments
  - Overlay reward curves
  - Diff configurations
  - Statistical significance tests
- **Comparison reports** with recommendations

### Exit Criteria

✅ Researcher can: define sweep → run batch → compare results → export datasets
✅ Same sweep rerun reproduces results within expected variance (±5%)
✅ Datasets pass schema validation and load in external code
✅ Checkpoint resume produces identical results from same point
✅ Multi-seed runs show proper statistical aggregation
✅ Queue handles 10+ experiments without issues

### Testing Requirements

- Unit tests for batch queue, checkpoint/lineage, dataset export schema
- Integration tests: multi-config sweep, dataset export + external loader
- UI tests: compare-experiments view, checkpoint browser
- Regression tests: checkpoint resume, multi-seed reproducibility

---

## Phase 4 – Tier 4 "Enterprise/Institution"

**Timeline:** 20-32 weeks
**Focus:** Multi-user, distributed, enterprise features

### Scope

#### Multi-User System

- **User accounts:**
  - Authentication (local or SSO)
  - Role-based permissions (admin, researcher, viewer)
  - User profiles and preferences
- **Team workspace:**
  - Shared experiment library
  - Access control (private, team, public)
  - Collaboration features (comments, annotations)
- **Project organization:**
  - Projects contain experiments
  - Project-level settings and resources

#### Central Database

- **Unified experiment database:**
  - All experiments across all users
  - Centralized storage (network share or cloud)
  - Backup and disaster recovery
- **Advanced search:**
  - Full-text search
  - Filter by any metadata field
  - Saved searches and views
  - Tag-based organization

#### Distributed Training

- **Multi-GPU support:**
  - Data parallelism
  - Model parallelism
  - Automatic GPU allocation
- **Multi-machine support:**
  - Cluster configuration
  - Job distribution
  - Fault tolerance
- **Compute management tab:**
  - Node/cluster monitoring
  - GPU utilization per node
  - Resource tracking and quotas

#### Licensing & Admin

- **License server:**
  - Floating licenses (concurrent users)
  - Node-locked licenses
  - Offline activation
- **Admin tools:**
  - User management dashboard
  - System health monitoring
  - Usage analytics
  - Batch operations
  - Instructor tools (assignment creation, grading)

#### Institutional Features

- **Export packs:**
  - Run bundles (complete experiment package)
  - Reproducibility packs (code + data + config + environment)
  - Publication supplements
  - Archival format
- **Compliance and security:**
  - Audit logging (all user actions)
  - Data retention policies
  - Export controls
  - Encryption at rest and in transit

### Exit Criteria

✅ Multiple users can share same experiment DB safely
✅ Admin can provision access, see runs, export bundles
✅ Distributed runs behave like single-node runs from UX standpoint
✅ License system enforces limits correctly
✅ Audit logs capture all critical actions
✅ System supports 100+ concurrent users
✅ Uptime > 99% in production deployment

### Testing Requirements

- Unit tests for user/role model, license enforcement, experiment DB repository
- Integration tests: multi-user access, distributed run simulation
- Security tests: permissions, role enforcement, data isolation
- Load tests: 100+ concurrent users, 1000+ experiments
- Compliance tests: audit log completeness, data retention

---

## Cross-Phase Considerations

### Backward Compatibility

- Each phase must support configs from previous phases
- Migration scripts for database schema changes
- Deprecation warnings for removed features (minimum 1 phase notice)

### Performance Targets

- **Phase 0-1:** Single GPU, local storage, <10GB disk usage
- **Phase 2:** Single GPU, local storage, <50GB disk usage
- **Phase 3:** Multi-GPU optional, network storage optional, <500GB disk usage
- **Phase 4:** Multi-GPU/multi-node, distributed storage, unlimited disk (quota-based)

### Documentation Requirements

- **Phase 0:** Developer docs (API reference)
- **Phase 1:** User guide (getting started, basic usage)
- **Phase 2:** Tutorial series (algorithm comparison, hyperparameter tuning)
- **Phase 3:** Research guide (reproducibility, benchmarking, dataset export)
- **Phase 4:** Admin guide (deployment, user management, troubleshooting)

### Support Model

- **Phase 0-1:** Community forum, GitHub issues
- **Phase 2:** Email support, video tutorials
- **Phase 3:** Priority support, research consultation
- **Phase 4:** Dedicated support team, SLA guarantees, on-site training

---

## Success Metrics

### Phase 0

- All unit tests pass (>80% coverage)
- Integration tests pass on CPU and GPU
- No memory leaks in 10-hour runs

### Phase 1

- Time to first successful training: < 5 minutes
- User retention: 30% return for second training
- Video generation success rate: > 95%
- Installer success rate: > 98% on clean Windows VMs

### Phase 2

- Educational adoption: Used in 5+ courses/tutorials
- Experiment completion rate: > 80%
- Reproducibility rate: > 90% (same seed → similar results)
- User-reported learning value: > 4/5 stars

### Phase 3

- Research citations: Mentioned in 10+ papers
- Reproducibility rate: > 95% (deterministic mode)
- Advanced feature usage: > 60% of users use sweeps or multi-seed
- Dataset export success rate: > 98%

### Phase 4

- Enterprise deployments: 5+ institutions
- Concurrent users: Support 100+ simultaneous users
- Uptime: 99.5% SLA
- License compliance: 100% (no violations)
- User satisfaction: > 4.5/5 stars

---

## Risk Mitigation

### Technical Risks

- **GPU compatibility issues:** Test on multiple NVIDIA GPU generations (Pascal, Turing, Ampere, Ada)
- **Memory leaks in long runs:** Continuous profiling, automated leak detection
- **Video generation failures:** Fallback to metrics-only mode, detailed error logging
- **Distributed training complexity:** Start with simple multi-process, defer true distributed to Phase 4

### Product Risks

- **Feature creep:** Strict adherence to phase scope, defer non-critical features
- **Poor UX:** User testing at end of each phase, iterate based on feedback
- **Performance degradation:** Benchmark suite run on every commit, performance budgets
- **Compatibility issues:** Automated testing on multiple Windows versions (10, 11)

### Business Risks

- **Low adoption:** Early beta program, community building, content marketing
- **Support burden:** Comprehensive documentation, self-service tools, community forum
- **Competition:** Focus on unique value props (ease of use, video generation, reproducibility)

---

## Next Steps

1. **Immediate (Week 1-2):**

   - Set up project structure for Phase 0
   - Implement `ExperimentConfig` class
   - Set up pytest infrastructure
   - Create initial test matrix

2. **Short-term (Week 3-4):**

   - Implement `TrainingPipeline` orchestrator
   - Implement `MetricEventBus`
   - Create integration test suite
   - Begin video generator implementation

3. **Medium-term (Month 2-3):**

   - Complete Phase 0 exit criteria
   - Begin UI prototyping for Phase 1
   - User research and UX design
   - Installer development

4. **Long-term (Month 4+):**
   - Phase 1 development and testing
   - Beta program launch
   - Community building
   - Plan Phase 2 features based on feedback
