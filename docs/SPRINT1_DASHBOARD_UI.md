# Sprint 1: Dashboard & UI Polish

**Duration:** Week 1-2 (2025-11-30 → 2025-12-13)  
**Status:** 🚀 STARTING  
**Branch:** `feature/phase1-hobbyist-release`

---

## 🎯 Sprint Goal

Improve the existing dashboard for better user experience with collapsible sections, enhanced metrics visualization, real-time GPU monitoring, and a recent activity feed.

---

## 📋 Task Breakdown

### 1. Dashboard Layout Improvements

**Priority:** HIGH  
**Estimated Time:** 2-3 days

#### Tasks:
- [ ] **1.1 Add Collapsible Sections**
  - [ ] Create CollapsibleFrame widget (CustomTkinter)
  - [ ] Implement expand/collapse animation
  - [ ] Add section headers with icons
  - [ ] Save/restore section states in user preferences
  - **Files:** `tools/retro_ml_desktop/widgets/collapsible_frame.py` (new)

- [ ] **1.2 Reorganize Dashboard Layout**
  - [ ] Group related controls into sections:
    - Training Controls (Start/Stop/Pause/Resume)
    - Live Metrics (Charts)
    - GPU Status (Utilization, VRAM, Temperature)
    - Recent Activity (Last 10 experiments)
  - [ ] Improve spacing and padding
  - [ ] Add visual separators between sections
  - **Files:** `tools/retro_ml_desktop/main_window.py`

- [ ] **1.3 Dark Theme Polish**
  - [ ] Audit all colors for consistency
  - [ ] Improve contrast for better readability
  - [ ] Add hover states for interactive elements
  - [ ] Add focus indicators for accessibility
  - **Files:** `tools/retro_ml_desktop/theme.py` (new or update existing)

- [ ] **1.4 Status Indicators**
  - [ ] Create status badge widget (Running, Paused, Stopped, Completed, Failed)
  - [ ] Add color coding (green, yellow, red, blue, gray)
  - [ ] Add pulsing animation for "Running" status
  - [ ] Add status icons
  - **Files:** `tools/retro_ml_desktop/widgets/status_badge.py` (new)

---

### 2. Live Metrics Enhancements

**Priority:** HIGH  
**Estimated Time:** 3-4 days

#### Tasks:
- [ ] **2.1 Improve Chart Performance**
  - [ ] Profile current chart rendering performance
  - [ ] Implement data downsampling for large datasets (> 1000 points)
  - [ ] Use line simplification algorithms (Douglas-Peucker)
  - [ ] Add frame rate limiting (max 30 FPS updates)
  - [ ] Implement incremental updates (only new data)
  - **Files:** `tools/retro_ml_desktop/ml_plotting.py`

- [ ] **2.2 Add Chart Legends and Tooltips**
  - [ ] Improve legend positioning (draggable, auto-hide)
  - [ ] Add interactive tooltips on hover (show exact values)
  - [ ] Add crosshair cursor for precise reading
  - [ ] Add data point highlighting on hover
  - **Files:** `tools/retro_ml_desktop/ml_plotting.py`

- [ ] **2.3 Add Zoom/Pan Controls**
  - [ ] Enable matplotlib navigation toolbar
  - [ ] Add custom zoom buttons (Zoom In, Zoom Out, Reset)
  - [ ] Add pan mode toggle
  - [ ] Add "Fit to Data" button
  - [ ] Save/restore zoom state per experiment
  - **Files:** `tools/retro_ml_desktop/ml_plotting.py`

- [ ] **2.4 Add Export Chart Feature**
  - [ ] Add "Export as PNG" button
  - [ ] Add file dialog for save location
  - [ ] Add resolution options (1080p, 4K, custom)
  - [ ] Add transparent background option
  - [ ] Add timestamp to filename
  - **Files:** `tools/retro_ml_desktop/ml_plotting.py`

---

### 3. GPU Status Display

**Priority:** HIGH  
**Estimated Time:** 2-3 days

#### Tasks:
- [ ] **3.1 GPU Detection and Info**
  - [ ] Detect GPU name and model
  - [ ] Detect VRAM capacity (total)
  - [ ] Detect CUDA version
  - [ ] Detect driver version
  - [ ] Handle multiple GPUs (show all, highlight active)
  - **Files:** `tools/retro_ml_desktop/gpu_monitor.py` (new)

- [ ] **3.2 Real-Time GPU Metrics**
  - [ ] Show GPU utilization % (update every 1 second)
  - [ ] Show VRAM usage (Used/Total in GB)
  - [ ] Show GPU temperature (°C)
  - [ ] Show GPU power usage (W) if available
  - [ ] Show GPU clock speed (MHz) if available
  - **Files:** `tools/retro_ml_desktop/gpu_monitor.py`

- [ ] **3.3 GPU Status Widget**
  - [ ] Create GPU status card widget
  - [ ] Add progress bars for utilization and VRAM
  - [ ] Add color coding (green < 70%, yellow 70-90%, red > 90%)
  - [ ] Add warning indicators for high temp (> 80°C)
  - [ ] Add "No GPU Detected" fallback message
  - **Files:** `tools/retro_ml_desktop/widgets/gpu_status_card.py` (new)

- [ ] **3.4 GPU Monitoring Thread**
  - [ ] Create background thread for GPU monitoring
  - [ ] Update metrics every 1 second
  - [ ] Publish metrics to event bus
  - [ ] Handle GPU disconnection gracefully
  - [ ] Stop thread when application closes
  - **Files:** `tools/retro_ml_desktop/gpu_monitor.py`

---

### 4. Recent Activity Feed

**Priority:** MEDIUM  
**Estimated Time:** 2-3 days

#### Tasks:
- [ ] **4.1 Activity Feed Widget**
  - [ ] Create scrollable activity feed widget
  - [ ] Show last 10 experiments
  - [ ] Display: name, game, status, start time, duration
  - [ ] Add experiment icons/thumbnails
  - [ ] Add status badges
  - **Files:** `tools/retro_ml_desktop/widgets/activity_feed.py` (new)

- [ ] **4.2 Quick Actions**
  - [ ] Add "View Details" button (opens experiment details)
  - [ ] Add "Resume" button (for paused/stopped experiments)
  - [ ] Add "Delete" button (with confirmation)
  - [ ] Add "Open Video" button (if video exists)
  - [ ] Add context menu (right-click)
  - **Files:** `tools/retro_ml_desktop/widgets/activity_feed.py`

- [ ] **4.3 Timestamps and Duration**
  - [ ] Show relative timestamps ("2 hours ago", "Yesterday")
  - [ ] Show absolute timestamps on hover
  - [ ] Show duration (HH:MM:SS format)
  - [ ] Show progress % for running experiments
  - **Files:** `tools/retro_ml_desktop/widgets/activity_feed.py`

- [ ] **4.4 Filtering**
  - [ ] Add filter dropdown (All, Running, Completed, Failed)
  - [ ] Add search box (filter by name or game)
  - [ ] Add sort options (newest first, oldest first, name)
  - [ ] Save filter preferences
  - **Files:** `tools/retro_ml_desktop/widgets/activity_feed.py`

---

## 🧪 Testing Checklist

- [ ] Dashboard loads without errors
- [ ] All collapsible sections expand/collapse smoothly
- [ ] Charts render at 30+ FPS during training
- [ ] GPU metrics update every second
- [ ] Activity feed shows correct experiments
- [ ] All quick actions work correctly
- [ ] Export chart produces valid PNG files
- [ ] No memory leaks during extended use (4+ hours)
- [ ] UI remains responsive during training

---

## 📊 Success Criteria

- [ ] Dashboard startup time < 2 seconds
- [ ] Chart update latency < 100ms
- [ ] GPU metrics accuracy > 95%
- [ ] Zero UI freezes during training
- [ ] User can collapse/expand all sections
- [ ] All widgets follow dark theme consistently

---

## 📝 Notes

- Focus on **performance** - dashboard must remain responsive during training
- Use **threading** for GPU monitoring to avoid blocking UI
- Implement **graceful degradation** - if GPU monitoring fails, show fallback message
- Add **error handling** for all user actions
- Test on **multiple GPU models** (if available)

