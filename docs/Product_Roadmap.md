# Retro ML Desktop — Product Roadmap

## Overview

This document outlines the four-tier evolution of Retro ML Desktop, from a simple hobbyist tool to an enterprise-grade reinforcement learning platform. Each tier builds upon the previous, maintaining backward compatibility while expanding capabilities.

---

## Tier 1 — Hobbyist Edition

**Target Audience:** Casual users, ML enthusiasts, content creators  
**Value Proposition:** Simple, approachable RL training with no setup

### Core Requirements

#### Installation & Setup

- **One-click installer** with hybrid approach (small installer + PyTorch download on first run)
- **Built-in Atari games** (no ROM hunting required)
- **NVIDIA GPU-only** initial target (detect and utilize automatically)
- **Zero configuration** required to start first training

#### Training Interface

- **Simple training presets:**
  - Short (1-2 hours)
  - Medium (4-6 hours)
  - Long (10+ hours)
- **Fillable time inputs** (hours/minutes fields, not dropdowns)
- **Video length = training duration** (direct mapping)
- **Start/Stop/Pause controls** with clear visual feedback

#### Metrics & Monitoring

- **Basic metrics display:**
  - Current reward
  - Total timesteps
  - Training ETA
  - Episode count
- **Real-time updates** (refresh every 5-10 seconds)
- **Simple progress bar** showing completion percentage

#### Video Generation

- **Automatic video generation** after training completes
- **Time-lapse mode** (condensed training progress)
- **Final agent run** (best performance showcase)
- **Meaningful auto-naming** based on game, date, and duration
- **Single continuous video** matching requested target length

#### Dashboard

- **Simple, clean interface** with collapsible sections:
  - Training Controls
  - Current Metrics
  - GPU Status (basic)
  - Recent Activity
- **Videos tab** with:
  - Thumbnail previews
  - Rename functionality
  - Delete with confirmation
  - Open in default player
  - Sort by date/name/size

### Architecture Notes

#### Core Design Principles

- **Experiment IDs** used internally for all training sessions
- **Universal experiment config object** as single source of truth
- **Metric event bus** implemented early (pub/sub pattern)
- **D drive storage** for videos and training data (avoid OOM)
- **Unique run IDs** per session (prevent data corruption)

#### Key Components

```
experiment_config = {
    "id": "unique_run_id",
    "game": "BreakoutNoFrameskip-v4",
    "duration_hours": 2.0,
    "preset": "short",
    "created_at": "timestamp",
    "status": "running|completed|failed"
}
```

#### Event Bus Structure

```
MetricEventBus:
  - emit(event_type, data)
  - subscribe(event_type, callback)
  - events: reward_update, timestep_update, training_complete
```

---

## Tier 2 — Student / Education Edition

**Target Audience:** Students, educators, self-learners  
**Value Proposition:** Teach RL concepts with structured experiments

### Core Requirements

#### Algorithm Selection

- **Multiple algorithms available:**
  - PPO (Proximal Policy Optimization)
  - A2C (Advantage Actor-Critic)
  - DQN (Deep Q-Network)
- **Algorithm comparison mode** (side-by-side results)
- **Algorithm-specific presets** (tuned defaults per algorithm)

#### Hyperparameter Control

- **Simple form UI** for common hyperparameters:
  - Learning rate
  - Batch size
  - Discount factor (gamma)
  - Entropy coefficient
  - Number of environments
- **Tooltips and explanations** for each parameter
- **Validation and safe ranges** (prevent invalid configs)
- **Reset to defaults** button

#### Reproducibility

- **Seed control:**
  - Manual seed entry
  - Random seed generation
  - Seed displayed in results
- **Deterministic mode toggle** (when possible)
- **Config export** (save exact settings used)

#### Data Export

- **Export metrics to:**
  - CSV (timestep, reward, loss, etc.)
  - JSON (full experiment metadata)
  - PNG (reward curves, loss plots)
- **Batch export** (multiple experiments at once)
- **Export location selection**

#### Model Management

- **Basic save/load:**
  - Save trained models
  - Load and continue training
  - Load for evaluation only
- **Model metadata** (algorithm, game, training time)
- **Model browser** (list saved models with details)

#### Advanced Configuration

- **"Advanced" collapsible panel** exposing:
  - Network architecture options
  - Frame stacking settings
  - Reward scaling
  - Gradient clipping
  - Update frequency
- **Preset override warning** (notify when changing preset values)

#### Onboarding

- **First-run tutorial:**
  - "Train your first model" walkthrough
  - Interactive tooltips
  - Sample experiment with known-good results
- **Help documentation** (embedded or web-based)
- **Example experiments** (pre-configured scenarios)

### Architecture Notes

#### Expanded Config Object

```
experiment_config = {
    ...tier_1_fields,
    "algorithm": "PPO|A2C|DQN",
    "hyperparameters": {
        "learning_rate": 0.0003,
        "batch_size": 256,
        "gamma": 0.99,
        ...
    },
    "seed": 42,
    "advanced": {
        "network_arch": [64, 64],
        "frame_stack": 4,
        ...
    }
}
```

#### Dashboard Enhancements

- **Experiments table view:**
  - List all experiments (past and present)
  - Filter by game, algorithm, status
  - Sort by date, duration, best reward
  - Quick actions (view, delete, export)
- **Tabbed interface:**
  - Training (current session)
  - Experiments (history)
  - Videos (generated content)
  - Settings (global preferences)

---

## Tier 3 — Research Edition

**Target Audience:** Researchers, advanced practitioners, benchmarking enthusiasts  
**Value Proposition:** Serious experimentation, benchmarking, and reproducibility

### Core Requirements

#### Deterministic Training

- **Fully deterministic pipeline:**
  - Controlled random seeds (PyTorch, NumPy, environment)
  - Deterministic CUDA operations (when possible)
  - Reproducibility verification tools
- **Environment versioning** (lock Gym/Gymnasium versions)
- **Dependency snapshot** (exact package versions recorded)

#### Multi-Seed Experiments

- **Multi-seed runs:**
  - Run same config with N different seeds
  - Automatic seed generation
  - Aggregate statistics (mean, std, min, max)
- **Experiment sweeps:**
  - Grid search over hyperparameters
  - Random search support
  - Sweep templates (common patterns)

#### Batch Processing

- **Experiment queue:**
  - Add multiple experiments to queue
  - Priority ordering
  - Pause/resume queue
  - Resource-aware scheduling (don't overload GPU)
- **Queue management UI:**
  - Drag-and-drop reordering
  - Bulk actions (cancel, duplicate)
  - Queue persistence (survives app restart)

#### Checkpoint Management

- **Versioned checkpoints:**
  - Save at regular intervals
  - Tag important checkpoints
  - Checkpoint browser with timeline
- **Checkpoint operations:**
  - Resume from any checkpoint
  - Compare checkpoints
  - Export checkpoint bundles
- **Automatic cleanup** (configurable retention policy)

#### Offline RL Support

- **Dataset export:**
  - Replay buffer dumps
  - SARS tuples (State, Action, Reward, Next State)
  - Episode trajectories
  - Format options (HDF5, Parquet, custom)
- **Dataset browser:**
  - Preview dataset contents
  - Statistics and distributions
  - Quality checks (coverage, diversity)

#### Advanced Metrics

- **Comprehensive metric tracking:**
  - Policy entropy (exploration measure)
  - KL divergence (policy change magnitude)
  - Loss curves (policy loss, value loss, total loss)
  - FPS (frames per second, training speed)
  - Update statistics (gradient norms, parameter changes)
  - Value function accuracy
  - Advantage estimates
- **Custom metric definitions** (user-defined calculations)
- **Metric export** (all formats from Tier 2, plus TensorBoard)

#### Visualization Panel

- **Interactive charts:**
  - Reward curves with confidence intervals
  - Loss landscapes
  - Hyperparameter sensitivity plots
  - Learning rate schedules
- **Diagnostic tools:**
  - Gradient flow visualization
  - Activation distributions
  - Weight histograms
  - Attention maps (if applicable)
- **Real-time plotting** (updates during training)
- **Export visualizations** (high-res PNG, SVG, interactive HTML)

#### Experiment Comparison

- **Side-by-side comparison:**
  - Select 2+ experiments
  - Overlay reward curves
  - Diff configurations
  - Statistical significance tests
- **Comparison reports:**
  - Auto-generated summary
  - Best performer identification
  - Recommendation engine (suggest next experiments)

### Architecture Notes

#### Experiment Lineage Graph

```
ExperimentNode:
  - id: unique identifier
  - parent_id: forked from (if applicable)
  - children: list of derived experiments
  - config: full configuration
  - results: metrics and artifacts
  - relationships: "baseline", "ablation", "sweep_member"
```

#### Storage System

- **Hierarchical storage:**
  ```
  D:/RetroML/
    experiments/
      {experiment_id}/
        config.json
        checkpoints/
        metrics/
        videos/
        datasets/
    models/
      {model_id}/
    cache/
  ```
- **Metadata database** (SQLite or similar)
- **Efficient retrieval** (indexed by game, algorithm, date, tags)

#### Plugin Architecture

- **Custom algorithm support:**
  - Algorithm interface/base class
  - Registration system
  - UI auto-generation from algorithm spec
- **Plugin discovery** (scan plugins directory)
- **Sandboxed execution** (safety considerations)

---

## Tier 4 — Enterprise / Institution Edition

**Target Audience:** Universities, research labs, R&D teams, institutions
**Value Proposition:** Scalable RL platform for teams and organizations

### Core Requirements

#### Distributed Training

- **Multi-GPU support:**
  - Data parallelism (multiple environments per GPU)
  - Model parallelism (large models split across GPUs)
  - Automatic GPU allocation
  - GPU affinity control
- **Multi-machine support:**
  - Cluster configuration
  - Job distribution
  - Fault tolerance (handle node failures)
  - Network optimization (reduce communication overhead)
- **Hybrid mode** (local + cloud resources)

#### Compute Management

- **Compute tab:**
  - Node/cluster monitoring
  - GPU utilization per node
  - VRAM usage per node
  - Network bandwidth
  - Job allocation visualization
- **Resource tracking:**
  - Available vs. allocated resources
  - Queue wait time estimates
  - Cost tracking (if cloud resources)
- **Resource limits:**
  - Per-user quotas
  - Per-project budgets
  - Priority tiers

#### Team Workspace

- **User accounts:**
  - Authentication (local or SSO)
  - Role-based permissions (admin, researcher, viewer)
  - User profiles and preferences
- **Shared experiments:**
  - Team experiment library
  - Access control (private, team, public)
  - Collaboration features (comments, annotations)
- **Project organization:**
  - Projects contain experiments
  - Project-level settings and resources
  - Project templates

#### Central Experiment Database

- **Unified database:**
  - All experiments across all users
  - Centralized storage (network share or cloud)
  - Backup and disaster recovery
- **Advanced search:**
  - Full-text search
  - Filter by any metadata field
  - Saved searches and views
  - Tag-based organization
- **Experiment discovery:**
  - Trending experiments
  - Top performers
  - Recently completed
  - Recommended based on user activity

#### Licensing & Activation

- **License server:**
  - Floating licenses (concurrent users)
  - Node-locked licenses (specific machines)
  - License pool management
- **Offline activation:**
  - Generate activation codes
  - Manual license file import
  - Grace period for temporary disconnection
- **License compliance:**
  - Usage reporting
  - Audit logs
  - Renewal notifications

#### Institutional Features

- **Export packs:**
  - Run bundles (complete experiment package)
  - Reproducibility packs (code + data + config + environment)
  - Publication supplements (figures + tables + raw data)
  - Archival format (long-term preservation)
- **Admin tools:**
  - User management dashboard
  - System health monitoring
  - Usage analytics
  - Batch operations (bulk user creation, etc.)
  - Instructor tools (assignment creation, grading support)
- **Compliance and security:**
  - Audit logging (all user actions)
  - Data retention policies
  - Export controls (restrict sensitive data)
  - Encryption at rest and in transit

#### Dataset Management

- **Large-scale storage:**
  - Tiered storage (hot/warm/cold)
  - Compression and deduplication
  - Storage quotas per user/project
- **Dataset catalog:**
  - Browse all datasets
  - Dataset versioning
  - Dataset lineage (which experiments produced it)
  - Dataset sharing and permissions
- **Dataset operations:**
  - Merge datasets
  - Split datasets (train/val/test)
  - Transform datasets (preprocessing pipelines)
  - Dataset quality metrics

### Architecture Notes

#### Service Layer Architecture

```
Services:
  - AuthService (authentication, authorization)
  - ExperimentService (CRUD, search, lineage)
  - StorageService (files, datasets, models)
  - ComputeService (resource allocation, job scheduling)
  - MetricsService (collection, aggregation, querying)
  - LicenseService (validation, enforcement)
  - AuditService (logging, compliance)
```

#### Dashboard Evolution

- **System command center:**
  - Multi-user activity feed
  - System-wide metrics (total experiments, GPU hours, etc.)
  - Alerts and notifications
  - Quick actions (launch experiment, view reports)
- **Customizable layouts:**
  - Drag-and-drop widgets
  - Save custom views
  - Role-specific defaults
- **Responsive design** (desktop, tablet, mobile monitoring)

#### Multi-User Support

- **Session management:**
  - Multiple concurrent sessions per user
  - Session isolation (experiments don't interfere)
  - Session persistence (resume after disconnect)
- **Collaboration features:**
  - Real-time updates (see others' experiments)
  - Shared dashboards (team view)
  - Notifications (experiment complete, errors, etc.)

---

## Cross-Tier Considerations

### Backward Compatibility

- **Config migration:** Each tier can import configs from previous tiers
- **UI progressive disclosure:** Advanced features hidden by default, revealed as needed
- **Upgrade path:** Clear migration guides between tiers

### Performance & Scalability

- **Tier 1:** Single GPU, single user, local storage
- **Tier 2:** Single GPU, single user, local storage, more data
- **Tier 3:** Single/multi GPU, single user, local/network storage, large datasets
- **Tier 4:** Multi-GPU, multi-user, distributed storage, enterprise scale

### Data Retention

- **Tier 1:** Keep recent experiments (e.g., last 30 days or 50 experiments)
- **Tier 2:** User-controlled retention, manual cleanup
- **Tier 3:** Automatic archival, configurable retention policies
- **Tier 4:** Institutional policies, compliance-driven retention

### Support & Documentation

- **Tier 1:** Basic docs, community forum
- **Tier 2:** Comprehensive docs, video tutorials, email support
- **Tier 3:** API docs, research papers, priority support
- **Tier 4:** Dedicated support team, on-site training, SLA guarantees

---

## Implementation Priorities

### Phase 1: Foundation (Tier 1)

1. Core experiment config system
2. Metric event bus
3. Basic training loop with PPO
4. Simple desktop UI (Electron or similar)
5. Video generation pipeline
6. GPU detection and utilization

### Phase 2: Education (Tier 2)

1. Multi-algorithm support (A2C, DQN)
2. Hyperparameter UI
3. Experiment history and table view
4. Export functionality
5. Model save/load
6. Onboarding flow

### Phase 3: Research (Tier 3)

1. Deterministic training pipeline
2. Multi-seed and sweep support
3. Experiment queue
4. Advanced metrics and visualization
5. Checkpoint management
6. Offline RL dataset export

### Phase 4: Enterprise (Tier 4)

1. Distributed training infrastructure
2. User authentication and authorization
3. Central database and storage
4. License server
5. Admin tools and dashboards
6. Institutional export and compliance features

---

## Success Metrics

### Tier 1

- Time to first successful training: < 5 minutes
- User retention: 30% return for second training
- Video generation success rate: > 95%

### Tier 2

- Educational adoption: Used in 10+ courses/tutorials
- Experiment completion rate: > 80%
- User-reported learning value: > 4/5 stars

### Tier 3

- Research citations: Mentioned in 50+ papers
- Reproducibility rate: > 90% (same config → same results)
- Advanced feature usage: > 60% of users use sweeps or multi-seed

### Tier 4

- Enterprise deployments: 10+ institutions
- Concurrent users: Support 100+ simultaneous users
- Uptime: 99.5% SLA
- License compliance: 100% (no violations)

---

## Conclusion

This roadmap provides a clear evolution path from a simple hobbyist tool to an enterprise-grade RL platform. Each tier builds on the previous, maintaining a consistent architecture while expanding capabilities. The key to success is:

1. **Start simple** (Tier 1) and validate core value proposition
2. **Expand thoughtfully** (Tier 2) based on user feedback
3. **Enable research** (Tier 3) with advanced features
4. **Scale to enterprise** (Tier 4) when demand justifies investment

By following this roadmap, Retro ML Desktop can serve users across the entire spectrum of RL training needs.
