# Phase 1: Hobbyist Release - Implementation Plan

**Status:** 🚀 IN PROGRESS  
**Started:** 2025-11-30  
**Target Completion:** 4-8 weeks  
**Branch:** `feature/phase1-hobbyist-release`

---

## 🎯 Phase 1 Goals

Create a **minimal viable desktop application** that allows hobbyist users to:

- Install the application easily (< 5 minutes)
- Run their first training session (< 5 minutes)
- See live training progress
- Watch generated videos of AI gameplay
- Manage their experiments and videos

---

## 📋 Exit Criteria

- [x] ✅ Phase 0 Complete - Core engine stabilized
- [x] ✅ Phase 1 (UI Migration) Complete - UI using retro_ml package
- [x] ✅ Phase 2 (Integration Testing) Complete - Adapter layer tested
- [ ] User can: install → run Quick Start → see training progress → watch video
- [ ] No crashes in normal flow on clean Windows VM
- [ ] Logs written for every experiment (config + run metadata)
- [ ] Videos auto-generated with meaningful names
- [ ] GPU properly detected and utilized (NVIDIA only)
- [ ] Installer size < 100MB (downloads PyTorch on first run)

---

## 🗓️ Sprint Breakdown

### Sprint 1: Dashboard & UI Polish (Week 1-2)

**Goal:** Improve the existing dashboard for better user experience

**Tasks:**

1. **Dashboard Layout Improvements**

   - [ ] Add collapsible sections for Training Controls, Metrics, GPU Status
   - [ ] Improve visual hierarchy and spacing
   - [ ] Add dark theme polish (consistent colors, better contrast)
   - [ ] Add status indicators (running, paused, stopped, completed)

2. **Live Metrics Enhancements**

   - [ ] Improve chart performance (reduce lag during training)
   - [ ] Add chart legends and tooltips
   - [ ] Add zoom/pan controls
   - [ ] Add export chart as PNG feature

3. **GPU Status Display**

   - [ ] Show GPU name and VRAM capacity
   - [ ] Show real-time GPU utilization %
   - [ ] Show VRAM usage (Used/Total)
   - [ ] Show temperature (if available)
   - [ ] Add warning indicators for high usage/temperature

4. **Recent Activity Feed**
   - [ ] Show last 10 experiments with status
   - [ ] Show quick actions (view, resume, delete)
   - [ ] Add timestamps and duration
   - [ ] Add filtering (all, running, completed, failed)

**Deliverables:**

- Polished dashboard with collapsible sections
- Real-time GPU monitoring
- Improved metrics visualization
- Recent activity feed

---

### Sprint 2: Training Wizard & Presets (Week 3-4)

**Goal:** Create a simple, guided training experience

**Tasks:**

1. **Training Wizard UI**

   - [ ] Create wizard dialog/window
   - [ ] Add step-by-step navigation (Game → Preset → Settings → Confirm)
   - [ ] Add progress indicator
   - [ ] Add "Quick Start" shortcut (skip wizard, use defaults)

2. **Game Selection**

   - [ ] Create game selection grid with thumbnails
   - [ ] Add game descriptions and difficulty ratings
   - [ ] Add search/filter functionality
   - [ ] Highlight recommended games for beginners

3. **Preset System**

   - [ ] Implement preset selection (Quick/Short/Medium/Long)
   - [ ] Add preset descriptions with expected duration
   - [ ] Add custom preset option
   - [ ] Show estimated time and resource requirements

4. **Time Input Controls**

   - [ ] Replace dropdown with fillable hours/minutes fields
   - [ ] Add validation (min/max limits)
   - [ ] Show estimated timesteps based on time
   - [ ] Add "recommended" time suggestions

5. **Training Controls**
   - [ ] Improve Start/Stop/Pause/Resume buttons
   - [ ] Add confirmation dialogs for destructive actions
   - [ ] Add progress bar with time remaining
   - [ ] Add cancel training option

**Deliverables:**

- User-friendly training wizard
- Game selection interface
- Preset system with time controls
- Improved training controls

---

### Sprint 3: Video Management (Week 5-6)

**Goal:** Create a dedicated Videos tab for managing generated videos

**Tasks:**

1. **Videos Tab UI**

   - [ ] Create Videos tab in main window
   - [ ] Add grid/list view toggle
   - [ ] Add sort options (date, name, size, duration)
   - [ ] Add filter options (game, date range)

2. **Video Thumbnails**

   - [ ] Generate thumbnails from videos (first frame or middle frame)
   - [ ] Cache thumbnails for performance
   - [ ] Add hover preview (play short clip)
   - [ ] Add video metadata overlay (duration, size, date)

3. **Video Operations**

   - [ ] Add Rename functionality
   - [ ] Add Delete functionality (with confirmation)
   - [ ] Add Open in default player
   - [ ] Add Open containing folder
   - [ ] Add Copy path to clipboard

4. **Video Generation Settings**
   - [ ] Add video quality settings (resolution, FPS, codec)
   - [ ] Add milestone percentage configuration
   - [ ] Add clip duration settings
   - [ ] Add auto-generation toggle

**Deliverables:**

- Functional Videos tab
- Thumbnail generation and caching
- Video management operations
- Video generation settings

---

### Sprint 4: Installer & Distribution (Week 7-8)

**Goal:** Create a professional installer for easy distribution

**Tasks:**

1. **Installer Development**

   - [ ] Choose installer framework (Inno Setup, NSIS, or PyInstaller + custom)
   - [ ] Create installer script
   - [ ] Add license agreement screen
   - [ ] Add installation directory selection
   - [ ] Add desktop shortcut option
   - [ ] Add Start Menu entry

2. **First-Run Experience**

   - [ ] Create welcome screen
   - [ ] Add GPU detection and validation
   - [ ] Add PyTorch download progress indicator
   - [ ] Add dependency installation (FFmpeg, etc.)
   - [ ] Add quick tutorial/walkthrough option

3. **Application Bundling**

   - [ ] Bundle Python runtime
   - [ ] Bundle core dependencies (except PyTorch)
   - [ ] Bundle FFmpeg binaries
   - [ ] Optimize bundle size (< 100MB)
   - [ ] Test on clean Windows VM

4. **Auto-Update System** (Optional)

   - [ ] Add version checking
   - [ ] Add update notification
   - [ ] Add auto-download and install
   - [ ] Add release notes display

5. **Uninstaller**
   - [ ] Create uninstaller script
   - [ ] Add option to keep user data
   - [ ] Clean up registry entries
   - [ ] Clean up shortcuts

**Deliverables:**

- Professional Windows installer (< 100MB)
- First-run setup wizard
- Auto-update system (optional)
- Clean uninstaller

---

## 📊 Success Metrics

- [ ] Time to first successful training: < 5 minutes
- [ ] User retention: 30% return for second training
- [ ] Video generation success rate: > 95%
- [ ] Installer success rate: > 98% on clean Windows VMs
- [ ] Zero crashes in normal flow
- [ ] GPU detection rate: 100% for NVIDIA GPUs

---

## 🔧 Technical Requirements

### Performance

- Dashboard updates: < 100ms latency
- Chart rendering: 60 FPS during training
- Video thumbnail generation: < 2 seconds per video
- Application startup: < 3 seconds

### Compatibility

- Windows 10 (64-bit) minimum
- Windows 11 (64-bit) recommended
- NVIDIA GPU (Pascal generation or newer)
- 8GB RAM minimum, 16GB recommended
- 50GB free disk space

### Dependencies

- Python 3.10+
- PyTorch 2.0+ (downloaded on first run)
- Stable-Baselines3
- Gymnasium
- CustomTkinter
- FFmpeg (bundled)

---

## 📝 Notes

- Focus on **simplicity** - this is for hobbyists, not researchers
- **No advanced features** - save those for Phase 2 (Student/Education)
- **Stability over features** - better to have fewer features that work perfectly
- **User testing** - get feedback from 3-5 users before finalizing
