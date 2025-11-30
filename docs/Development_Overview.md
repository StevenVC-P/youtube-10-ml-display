# Retro ML Desktop — Development Overview

## Quick Reference

This document provides a high-level overview of the development documentation structure and how to use it.

---

## Documentation Structure

### 1. **Product_Roadmap.md** (Product Vision)
**Location:** `docs/Product_Roadmap.md`  
**Purpose:** Long-term product vision across four tiers

**Contents:**
- Tier 1: Hobbyist Edition (simple, approachable)
- Tier 2: Student/Education Edition (structured learning)
- Tier 3: Research Edition (advanced experimentation)
- Tier 4: Enterprise/Institution Edition (scalable platform)

**Use this when:**
- Understanding the overall product vision
- Planning features for future tiers
- Communicating with stakeholders
- Making architectural decisions that affect multiple tiers

---

### 2. **ROADMAP.md** (Development Phases)
**Location:** `ROADMAP.md`  
**Purpose:** Phased development plan with concrete deliverables

**Contents:**
- Phase 0: Core Engine Stabilization (backend-only)
- Phase 1: Tier 1 "Hobbyist" Release (minimal viable product)
- Phase 2: Tier 2 "Student/Education" (educational features)
- Phase 3: Tier 3 "Research" (advanced experimentation)
- Phase 4: Tier 4 "Enterprise/Institution" (multi-user, distributed)

**Each phase includes:**
- Scope and deliverables
- Exit criteria (what must work before moving on)
- Testing requirements
- Success metrics

**Use this when:**
- Planning sprint work
- Determining what to build next
- Checking if a phase is complete
- Understanding dependencies between features

---

### 3. **TESTING.md** (Testing Strategy)
**Location:** `TESTING.md`  
**Purpose:** Comprehensive testing approach across all phases

**Contents:**
- Core test types (unit, integration, UI, installer, performance, regression)
- Phase-by-phase testing requirements
- Test infrastructure setup
- Testing best practices
- Test execution commands

**Use this when:**
- Writing new tests
- Setting up test infrastructure
- Understanding coverage requirements
- Debugging test failures
- Planning test automation

---

### 4. **TEST_MATRIX.md** (Test Implementation Guide)
**Location:** `TEST_MATRIX.md`  
**Purpose:** Detailed test matrix for generating pytest files

**Contents:**
- Test file matrix for each phase
- Estimated test counts
- Test file templates (unit, integration)
- Priority definitions (P0, P1, P2, P3)
- Test execution strategy

**Use this when:**
- Creating new test files
- Estimating testing effort
- Prioritizing test implementation
- Generating pytest files from templates

---

## Development Workflow

### Starting a New Phase

1. **Review ROADMAP.md** for the phase
   - Understand scope and deliverables
   - Review exit criteria
   - Check dependencies on previous phases

2. **Review TESTING.md** for the phase
   - Understand testing requirements
   - Review coverage targets
   - Plan test infrastructure needs

3. **Review TEST_MATRIX.md** for the phase
   - Identify test files to create
   - Prioritize P0 tests
   - Use templates to generate test files

4. **Implement features and tests in parallel**
   - Write tests first (TDD) or alongside features
   - Run tests continuously
   - Track coverage metrics

5. **Verify exit criteria**
   - All P0 tests passing
   - Coverage targets met
   - Manual testing complete (if applicable)
   - Success metrics achieved

---

## Current Status

### Phase 0: Core Engine Stabilization
**Status:** 🚧 In Progress  
**Focus:** Backend-only, no UI

**Immediate Tasks:**
1. Set up project structure
2. Implement `ExperimentConfig` class
3. Set up pytest infrastructure
4. Create initial test files

**Next Milestone:** Quick Run from Python CLI

---

## Key Architectural Decisions

### 1. Experiment Config as Single Source of Truth
- All experiment parameters stored in `ExperimentConfig` object
- Serializable to JSON for persistence
- Validated on creation
- Versioned for backward compatibility

### 2. Metric Event Bus
- Pub/sub pattern for metrics
- Decouples metric producers from consumers
- Enables real-time UI updates
- Supports multiple subscribers (UI, file writers, TensorBoard)

### 3. Unique Experiment IDs
- Every training session has unique ID
- Prevents data corruption from concurrent runs
- Enables artifact linking (models, logs, videos)
- Supports experiment lineage tracking

### 4. D Drive Storage
- Use D drive for storage-intensive operations
- Avoids OOM issues on C drive
- Configurable storage location
- Automatic cleanup policies

### 5. GPU-Only Initial Target
- Focus on NVIDIA GPUs initially
- Simplifies development and testing
- Expand to AMD/CPU later if needed

---

## Success Metrics Summary

### Phase 0
- ✅ All unit tests pass (>80% coverage)
- ✅ Integration tests pass on CPU and GPU
- ✅ No memory leaks in 10-hour runs

### Phase 1
- ✅ Time to first successful training: < 5 minutes
- ✅ User retention: 30% return for second training
- ✅ Video generation success rate: > 95%

### Phase 2
- ✅ Educational adoption: Used in 5+ courses/tutorials
- ✅ Reproducibility rate: > 90% (same seed → similar results)
- ✅ User-reported learning value: > 4/5 stars

### Phase 3
- ✅ Research citations: Mentioned in 10+ papers
- ✅ Reproducibility rate: > 95% (deterministic mode)
- ✅ Advanced feature usage: > 60% of users use sweeps or multi-seed

### Phase 4
- ✅ Enterprise deployments: 5+ institutions
- ✅ Concurrent users: Support 100+ simultaneous users
- ✅ Uptime: 99.5% SLA

---

## Common Questions

### Q: Which document should I read first?
**A:** Start with this document (Development_Overview.md), then ROADMAP.md for your current phase.

### Q: How do I know what to build next?
**A:** Check ROADMAP.md for your current phase's scope and deliverables.

### Q: How do I know what tests to write?
**A:** Check TEST_MATRIX.md for your current phase, prioritize P0 tests.

### Q: How do I know if a phase is complete?
**A:** Check ROADMAP.md exit criteria for your phase. All must be met.

### Q: What's the difference between Product_Roadmap.md and ROADMAP.md?
**A:** Product_Roadmap.md is the long-term product vision (what we're building). ROADMAP.md is the development plan (how we're building it).

### Q: Do I need to implement all tiers?
**A:** No. Start with Phase 0 and Phase 1 (Tier 1). Later phases/tiers are optional based on market demand.

### Q: Can I skip tests?
**A:** No. P0 tests are required for phase completion. P1 tests should be implemented unless there's a strong justification to defer.

### Q: How do I handle backward compatibility?
**A:** Each phase must support configs from previous phases. Use migration scripts for breaking changes.

---

## Quick Links

- **Product Vision:** [Product_Roadmap.md](Product_Roadmap.md)
- **Development Plan:** [ROADMAP.md](../ROADMAP.md)
- **Testing Strategy:** [TESTING.md](../TESTING.md)
- **Test Matrix:** [TEST_MATRIX.md](../TEST_MATRIX.md)
- **Current Sprint:** See GitHub Projects (when set up)
- **Issue Tracker:** See GitHub Issues (when set up)

---

## Getting Help

### Technical Questions
- Review relevant documentation first
- Check existing code for patterns
- Ask in team chat (when set up)

### Product Questions
- Review Product_Roadmap.md
- Check with product owner
- Refer to user research (when available)

### Testing Questions
- Review TESTING.md and TEST_MATRIX.md
- Check existing test files for examples
- Run `pytest --help` for pytest options

---

## Next Steps

1. **Set up development environment:**
   - Install Python 3.10+
   - Install dependencies: `pip install -r requirements-dev.txt`
   - Install PyTorch (GPU version)
   - Verify GPU detection

2. **Set up testing infrastructure:**
   - Create `tests/` directory structure
   - Create `conftest.py` with common fixtures
   - Configure `pytest.ini`
   - Run initial test: `pytest --version`

3. **Start Phase 0 development:**
   - Implement `ExperimentConfig` class
   - Write unit tests for `ExperimentConfig`
   - Implement `TrainingPipeline` orchestrator
   - Write integration test for quick run

4. **Track progress:**
   - Update this document with current status
   - Mark completed tasks in ROADMAP.md
   - Track coverage metrics
   - Document decisions and learnings
