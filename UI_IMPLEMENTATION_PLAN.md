# UI Implementation Plan - Dashboard-First Architecture

## 📋 Strategic Decisions Summary

Based on your input, here's the confirmed direction:

### **Core Philosophy:**
- **Dashboard-first design** - Central hub for all activity
- **Power user architecture** - Built for scalability, exposed simply
- **Modular components** - Training Monitor, Export Service, Metric Bus
- **Experiment-centric** - Everything organized around experiment tracking

### **Navigation Structure:**
**Final Tabs:** Dashboard | Experiments | Models | Datasets | Videos | Settings

**Phase 1 (v1.0):** Dashboard | Training | Videos | Settings
**Phase 2 (v1.1):** + Experiments tab (advanced experiment management)
**Phase 3 (v2.0):** + Models | Datasets (full MLOps features)

---

## 🏗️ Architecture Overview

### **New Core Systems:**

#### **1. Experiment Config System**
```
Experiment {
  id: str
  name: str
  game: str
  algorithm: str (PPO/DQN)
  preset: str (quick/standard/epic)
  config: {
    total_timesteps: int
    video_length_hours: float
    learning_rate: float
    n_steps: int
    batch_size: int
    # ... full hyperparameters
  }
  lineage: {
    parent_experiment_id: str (for continue training)
    checkpoint_source: str
  }
  artifacts: {
    videos: [Video]
    models: [Model]
    logs: [LogFile]
  }
  metadata: {
    created: datetime
    status: str (running/completed/failed)
    tags: [str]
  }
}
```

**Implementation:**
- Create `experiment_manager.py` - Handles experiment CRUD
- Use existing `ml_database.py` - Extend with experiment tables
- Presets map to full configs (user sees "Quick", system uses full config)

---

#### **2. Metric Event Bus**
```
MetricEventBus {
  channels: {
    'training.progress': [subscribers]
    'training.complete': [subscribers]
    'video.generated': [subscribers]
    'experiment.status': [subscribers]
  }

  publish(event_type, data)
  subscribe(event_type, callback)
}
```

**Events:**
- `training.progress` - {experiment_id, timestep, progress_pct, metrics}
- `training.complete` - {experiment_id, final_metrics}
- `video.generated` - {experiment_id, video_path, video_type, duration}
- `experiment.status` - {experiment_id, status, message}

**Implementation:**
- Create `metric_event_bus.py` - Pub/sub pattern
- `process_manager.py` publishes events
- UI components subscribe (Dashboard, Training Monitor)
- Enables real-time updates without polling

---

#### **3. Export Service**
```
ExportService {
  # Architecture
  exporters: {
    'video.copy': VideoCopyExporter
    'video.youtube': YouTubeExporter (future)
    'video.twitter': TwitterExporter (future)
    'model.share': ModelExporter (future)
  }

  # Current (v1.0)
  export_video_to_location(video_path, dest_path)
  copy_video_path_to_clipboard(video_path)
  open_video_location(video_path)
  rename_video(video_path, new_name)

  # Future (v1.1+)
  export_for_youtube(video_path, metadata)
  export_for_twitter(video_path, compression_opts)
  export_model_checkpoint(model_id, format)
}
```

**Implementation:**
- Create `export_service.py` - Unified export interface
- Start with basic file operations (copy, rename, open folder)
- Architecture supports future exporters (YouTube, Twitter, etc.)

---

#### **4. Video Artifact System**
```
VideoArtifact {
  id: str
  experiment_id: str (link to parent experiment)
  path: str
  type: str (milestone/hour/evaluation)
  metadata: {
    duration: float
    size_mb: float
    created: datetime
    tags: [str] (e.g., ['10h_epic', 'breakout', 'best_score'])
    thumbnail_path: str (future)
  }
  metrics: {
    avg_score: float
    max_score: float
    episode_count: int
  }
}
```

**Implementation:**
- Extend `ml_database.py` with `video_artifacts` table
- `process_manager.py` registers videos on generation
- Videos linked to experiments via `experiment_id` foreign key

---

## 🎨 UI Redesign - Phase 1 (v1.0)

### **Tab 1: Dashboard (Home Screen)**

**Purpose:** Central hub showing recent activity, quick actions, system status

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Dashboard                                   [Quick Start]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📊 Recent Activity                  🎮 Quick Start          │
│  ┌───────────────────────────┐      ┌──────────────────────┐│
│  │ Training: Breakout #42    │      │ 🎯 Breakout          ││
│  │ Status: Running (45%)     │      │ ⏱️  1 Hour            ││
│  │ Time: 1h 23m / 3h est.    │      │ 🤖 PPO (Recommended) ││
│  │ [View Details] [Stop]     │      │                      ││
│  └───────────────────────────┘      │ [Start Training Now] ││
│                                      └──────────────────────┘│
│  🎬 Latest Videos                    📈 Recent Experiments   │
│  ┌───────────────────────────┐      ┌──────────────────────┐│
│  │ 🆕 Breakout_Epic_10h.mp4  │      │ Breakout #42 (Active)││
│  │    2.1 GB • 10h • Today   │      │ Breakout #41 (Done)  ││
│  │    Score: 234             │      │ Pong #15 (Done)      ││
│  │ [▶️ Play] [📁 Open Folder]│      │ [View All →]         ││
│  └───────────────────────────┘      └──────────────────────┘│
│                                                               │
│  ⚙️ System Status                                            │
│  ├─ CPU: 45% │ Memory: 8.2/16 GB │ GPU: CUDA Available ✅   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Recent Activity Widget** - Shows active training with live progress
- **Quick Start Panel** - One-click training with smart defaults
- **Latest Videos Widget** - Most recent video with quick actions
- **Recent Experiments Widget** - Last 3 experiments with status
- **System Status Bar** - CPU/Memory/GPU at a glance

**Empty State** (No training yet):
```
┌─────────────────────────────────────────┐
│   👋 Welcome to Retro ML Trainer        │
│                                         │
│   Get started in 3 easy steps:          │
│   1. Pick a game                        │
│   2. Click "Start Training"             │
│   3. Watch your AI learn!               │
│                                         │
│   [🚀 Start Your First Training]        │
└─────────────────────────────────────────┘
```

**Implementation Priority:** ⭐⭐⭐ CRITICAL

---

### **Tab 2: Training (Replaces "Training Processes")**

**Purpose:** Monitor and manage active/recent training sessions

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  🎮 Training Sessions                          [New Training]│
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Active Training                                             │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 🎯 Breakout Epic #42                    [Stop] [Pause]   ││
│  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░░░░  68%          ││
│  │                                                           ││
│  │ Time: 6h 45m / 10h estimated     │  Score: 187 → 234     ││
│  │ Episodes: 1,234 completed         │  GPU: CUDA ✅         ││
│  │                                                           ││
│  │ 📊 Live Metrics                   │  🎬 Videos Generated  ││
│  │ Reward: ▲ 189.5 (↑23%)           │  • 10% milestone ✅   ││
│  │ Loss: ▼ 0.042 (↓15%)             │  • 20% milestone ✅   ││
│  │ Timesteps: 2.5M / 10M            │  • 30% milestone ✅   ││
│  │                                   │  • 40% milestone ⏳   ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
│  Recent Experiments                                          │
│  ┌────────────────────────────────────────────────┐          │
│  │ Name          │ Game     │ Status   │ Duration │ Actions ││
│  ├────────────────────────────────────────────────┤          │
│  │ Breakout #41  │ Breakout │ ✅ Done  │ 4h       │ [Resume]││
│  │ Pong #15      │ Pong     │ ✅ Done  │ 1h       │ [Resume]││
│  │ Breakout #40  │ Breakout │ ⚠️ Failed│ 0.5h     │ [Retry] ││
│  └────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Training Monitor Panel** - Modular component showing live progress
- **Progress Bar** - Visual 0-100% with time estimates
- **Live Metrics** - Simplified view (reward, loss, timesteps)
- **Video Generation Tracker** - Shows which milestone videos are ready
- **Recent Experiments Table** - Quick access to resume/retry

**Implementation Priority:** ⭐⭐⭐ CRITICAL

---

### **Tab 3: Videos (Improved "Video Gallery")**

**Purpose:** Browse, watch, and export training videos

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  🎬 My Videos                    [Refresh] Filter: All ▼     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  📁 Experiments  │  🎞️ All Videos  │  ⭐ Favorites          │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 🎮 Thumbnail │  │ 🎮 Thumbnail │  │ 🎮 Thumbnail │      │
│  │ [Preview]    │  │ [Preview]    │  │ [Preview]    │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ Breakout_10h │  │ Pong_4h_Best │  │ Epic_Series  │      │
│  │ 10h • 2.1 GB │  │ 4h • 1.2 GB  │  │ 10h • 2.5 GB │      │
│  │ Today        │  │ Yesterday    │  │ Last Week    │      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│  │ ▶️ Play       │  │ ▶️ Play       │  │ ▶️ Play       │      │
│  │ 📁 Export     │  │ 📁 Export     │  │ 📁 Export     │      │
│  │ ✏️ Rename     │  │ ✏️ Rename     │  │ ✏️ Rename     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  Selected: Breakout_10h.mp4                                  │
│  [▶️ Play Video] [📋 Copy Path] [📁 Open Location] [🗑️ Delete]│
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Thumbnail Grid** - Visual card layout (implement in v1.1, list view in v1.0)
- **Metadata Tags** - Filter by game, duration, experiment
- **Quick Actions** - Play, Export, Rename per video
- **Batch Actions** - Multi-select for bulk operations (v1.1)
- **"Open Location"** - Opens folder for SELECTED video specifically

**Implementation Priority:** ⭐⭐ HIGH

---

### **Tab 4: Settings**

**Purpose:** System configuration, advanced features, help

**Layout:**
```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ Settings                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  General  │  Storage  │  Advanced  │  About                  │
│  ─────────                                                    │
│                                                               │
│  🎨 Appearance                                               │
│    Theme: [Dark ▼]  Color: [Blue ▼]                         │
│                                                               │
│  📍 Default Paths                                            │
│    Videos: [Documents/ML_Videos]    [Change]                │
│    Models: [Documents/ML_Models]    [Change]                │
│    Database: [AppData/ml_experiments.db]  [Change]          │
│                                                               │
│  🔔 Notifications                                            │
│    ☑️ Notify when training completes                         │
│    ☑️ Notify when video is generated                         │
│    ☐ Desktop notifications                                   │
│                                                               │
│  🎮 Training Defaults                                        │
│    Default Game: [Breakout ▼]                               │
│    Default Algorithm: [PPO ▼]                               │
│    Default Video Length: [4 hours ▼]                        │
│                                                               │
│  🔧 Advanced Settings                                        │
│    [Open ML Dashboard] (Technical charts)                   │
│    [Resource Configuration] (CPU/GPU allocation)            │
│    [CUDA Diagnostics]                                       │
│    [Install Atari ROMs]                                     │
│    [Storage Cleanup]                                        │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Features:**
- **Reorganized** - General settings front and center
- **Advanced Features** - ML Dashboard, Resource Config moved here
- **Notifications** - Toggle desktop notifications (implement event bus first)
- **Defaults** - Set preferred game/algorithm for Quick Start
- **Help & About** - Version, links, documentation

**Implementation Priority:** ⭐ MEDIUM

---

## 🔨 Implementation Phases

### **Phase 1: Foundation (Week 1-2)** ⭐⭐⭐

**Goal:** Build new core systems without breaking existing functionality

#### **Week 1: Backend Systems**
1. **Experiment Manager** (`experiment_manager.py`)
   - Create Experiment class/schema
   - CRUD operations for experiments
   - Extend `ml_database.py` with experiments table
   - Migration: Map existing training runs to experiments

2. **Metric Event Bus** (`metric_event_bus.py`)
   - Pub/sub implementation
   - Core event types (training.progress, training.complete, video.generated)
   - Integrate with `process_manager.py` (publisher)
   - Add unit tests

3. **Export Service** (`export_service.py`)
   - Basic file operations (copy, rename, open_folder)
   - Copy path to clipboard function
   - Video-specific export methods
   - Future-proof architecture for YouTube/Twitter exporters

#### **Week 2: Database Schema & Migration**
4. **Database Updates**
   - Create `experiments` table
   - Create `video_artifacts` table with experiment_id FK
   - Add metadata columns (tags, lineage)
   - Write migration script for existing data

5. **Video Artifact Integration**
   - Update video generation to create VideoArtifact records
   - Link videos to experiments
   - Add metadata tagging

**Acceptance Criteria:**
- ✅ Experiment manager can create/read/update/delete experiments
- ✅ Metric event bus publishes and delivers events
- ✅ Export service can copy video, open folder, copy path
- ✅ Database has new schema with migrated data
- ✅ Existing functionality still works (no breaking changes)

---

### **Phase 2: Dashboard UI (Week 3-4)** ⭐⭐⭐

**Goal:** Replace "Training Processes" with "Dashboard" as default view

#### **Week 3: Dashboard Components**
1. **Dashboard Tab** (new)
   - Create `dashboard_tab.py`
   - Empty state for new users
   - Recent Activity widget (shows active training)
   - Quick Start panel
   - Latest Videos widget
   - System Status bar

2. **Training Monitor Panel** (modular component)
   - Create `training_monitor.py` (reusable component)
   - Progress bar with % and time remaining
   - Live metrics display (subscribes to metric event bus)
   - Video generation tracker
   - Can be embedded in Dashboard or Training tab

#### **Week 4: Dashboard Integration**
3. **Quick Start Dialog**
   - Simplified training dialog with presets
   - "Quick" (30m), "Standard" (1h), "Epic" (4h/10h) buttons
   - Hides algorithm selection (defaults to PPO)
   - Uses experiment config under the hood

4. **Dashboard Logic**
   - Subscribe to metric event bus for real-time updates
   - Fetch latest experiment from database
   - Fetch latest video from database
   - Auto-refresh on events

5. **Tab Reordering**
   - Dashboard as first tab (default view)
   - Rename "Training Processes" → "Training"
   - Update navigation

**Acceptance Criteria:**
- ✅ Dashboard tab is default view
- ✅ Empty state shows welcome message
- ✅ Active training shows in Recent Activity with live updates
- ✅ Quick Start works with one click
- ✅ Latest video appears automatically
- ✅ Training Monitor shows progress, time remaining, live metrics

---

### **Phase 3: Training Tab Redesign (Week 5)** ⭐⭐

**Goal:** Improve training management and monitoring

1. **Training Tab Overhaul**
   - Embed Training Monitor panel (reuse from Dashboard)
   - Recent Experiments table (replaces old process list)
   - [Resume] button for completed experiments (continue training)
   - [Retry] button for failed experiments
   - Remove technical jargon (PID → Session ID)

2. **New Training Dialog Enhancement**
   - Add "Estimated time" calculation
   - Show recommended settings
   - Keep "Resume Training" option (experiment lineage)
   - Add tooltips for all options

**Acceptance Criteria:**
- ✅ Training tab shows modular Training Monitor
- ✅ Recent experiments table with Resume/Retry actions
- ✅ Estimated time shown in training dialog
- ✅ Resume Training creates child experiment with lineage

---

### **Phase 4: Videos Tab Redesign (Week 6)** ⭐⭐

**Goal:** Improve video discovery and export

1. **Videos Tab Improvements**
   - Add "Open Location" button (opens folder for SELECTED video)
   - Add "Copy Path" button
   - Add "Rename Video" dialog
   - Add metadata tags display
   - Link to parent experiment
   - Sort by newest first (default)

2. **Export Integration**
   - Wire up Export Service
   - Test copy path to clipboard
   - Test open folder for specific video
   - Test rename video

**Acceptance Criteria:**
- ✅ Can open folder for specific selected video
- ✅ Can copy video path to clipboard
- ✅ Can rename video in-app
- ✅ Videos show parent experiment link
- ✅ Videos sorted by date (newest first) by default

---

### **Phase 5: Notifications & Polish (Week 7)** ⭐

**Goal:** Add notifications and UI polish

1. **Desktop Notifications**
   - Windows toast notifications for training complete
   - Badge on Videos tab when new video generated
   - In-app notification popup when training complete

2. **UI Polish**
   - Add tooltips throughout
   - Improve error messages
   - Add loading states
   - Smooth animations for progress updates
   - Test all workflows end-to-end

3. **Onboarding Popup** (light)
   - Show on first launch after setup wizard
   - 3 tips: "Start training", "Check Dashboard", "Find videos in Videos tab"
   - "Don't show again" checkbox

**Acceptance Criteria:**
- ✅ Desktop notifications work on Windows
- ✅ Badge appears on Videos tab when new video ready
- ✅ Tooltips on all key UI elements
- ✅ Onboarding popup shows for new users
- ✅ All workflows tested and working

---

## 📐 Architecture Diagrams

### **System Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                         UI Layer                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Dashboard │  │ Training │  │  Videos  │  │ Settings │   │
│  │   Tab    │  │   Tab    │  │   Tab    │  │   Tab    │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │               │             │          │
│       └─────────────┴───────────────┴─────────────┘          │
│                         │                                    │
├─────────────────────────┼────────────────────────────────────┤
│                    Service Layer                             │
│         ┌───────────────┴───────────────┐                   │
│         │    Metric Event Bus           │                   │
│         │  (Pub/Sub Coordinator)        │                   │
│         └───┬───────────┬───────────┬───┘                   │
│             │           │           │                        │
│      ┌──────▼────┐ ┌───▼─────┐ ┌──▼────────┐              │
│      │Experiment │ │ Process │ │  Export   │              │
│      │  Manager  │ │ Manager │ │  Service  │              │
│      └──────┬────┘ └───┬─────┘ └──┬────────┘              │
│             │           │           │                        │
├─────────────┼───────────┼───────────┼────────────────────────┤
│                    Data Layer                                │
│         ┌───▼───────────▼───────────▼───┐                   │
│         │       MetricsDatabase          │                   │
│         │  ┌──────────┐  ┌────────────┐ │                   │
│         │  │experiments│  │video_      │ │                   │
│         │  │          │  │artifacts   │ │                   │
│         │  └──────────┘  └────────────┘ │                   │
│         └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### **Metric Event Flow:**

```
[Training Process]
      │
      │ publishes events
      ▼
[Metric Event Bus] ───────────────┐
      │                           │
      │ notifies                  │ notifies
      ▼                           ▼
[Dashboard Tab]            [Training Tab]
      │                           │
      │ updates UI                │ updates UI
      ▼                           ▼
[Recent Activity Widget]   [Training Monitor Panel]
```

---

## 📝 File Structure Changes

### **New Files to Create:**

```
tools/retro_ml_desktop/
├── experiment_manager.py       # NEW: Experiment CRUD operations
├── metric_event_bus.py         # NEW: Pub/sub event system
├── export_service.py           # NEW: Unified export interface
├── training_monitor.py         # NEW: Modular training progress component
├── dashboard_tab.py            # NEW: Dashboard UI
├── quick_start_dialog.py       # NEW: Simplified training dialog
└── notification_service.py     # NEW: Desktop notifications (Phase 5)
```

### **Files to Modify:**

```
tools/retro_ml_desktop/
├── main_simple.py              # MOD: Add Dashboard tab, reorder tabs
├── ml_database.py              # MOD: Add experiments, video_artifacts tables
├── process_manager.py          # MOD: Publish events to metric bus
└── video_player.py             # MOD: Add export buttons (copy path, open folder)
```

### **Files to Keep (Minimal Changes):**

```
tools/retro_ml_desktop/
├── ml_dashboard.py             # KEEP: Move to Settings as advanced feature
├── cuda_diagnostics.py         # KEEP: No changes
├── setup_wizard.py             # KEEP: No changes
└── config_manager.py           # KEEP: Minor additions for defaults
```

---

## 🎯 Success Metrics

### **User Experience Improvements:**

| Metric | Before | Target (v1.0) |
|--------|--------|---------------|
| Time to first training | ~5 min (complex dialog) | <30 sec (Quick Start) |
| Time to find latest video | ~2 min (hunt in gallery) | <5 sec (Dashboard widget) |
| Training status clarity | ⚠️ Technical logs | ✅ Progress bar + time remaining |
| Video export difficulty | ❌ Manual file search | ✅ One-click "Open Location" |
| Empty state guidance | ❌ Confusing technical tab | ✅ Welcome message with CTA |

### **Technical Improvements:**

| Metric | Before | Target (v1.0) |
|--------|--------|---------------|
| Real-time updates | ❌ Manual refresh | ✅ Event-driven updates |
| Code modularity | ⚠️ Monolithic UI | ✅ Reusable components |
| Architecture scalability | ⚠️ Limited | ✅ Experiment-centric, extensible |
| Database organization | ⚠️ Flat structure | ✅ Normalized with relationships |

---

## 🚀 Quick Start Implementation Guide

### **To Start (Week 1, Day 1):**

1. **Create Experiment Manager**
   ```bash
   cd tools/retro_ml_desktop
   # Create experiment_manager.py with Experiment class
   # Add experiments table to ml_database.py
   ```

2. **Create Metric Event Bus**
   ```bash
   # Create metric_event_bus.py with pub/sub pattern
   # Add unit tests
   ```

3. **Create Export Service**
   ```bash
   # Create export_service.py with basic methods
   ```

4. **Update Database Schema**
   ```bash
   # Extend ml_database.py
   # Write migration script
   ```

### **Testing Strategy:**

- **Unit Tests:** Experiment Manager, Event Bus, Export Service
- **Integration Tests:** Event flow from Process Manager → Event Bus → UI
- **UI Tests:** Dashboard interactions, Quick Start workflow
- **E2E Tests:** Complete workflow: Quick Start → Monitor → Watch Video → Export

---

## 📋 Checklist for Phase 1 Completion

- [ ] Experiment Manager implemented with CRUD operations
- [ ] Metric Event Bus working with pub/sub
- [ ] Export Service has basic file operations
- [ ] Database schema updated with migrations
- [ ] Existing functionality still works
- [ ] Unit tests passing
- [ ] Documentation updated

---

## 🔮 Future Roadmap (v1.1+)

### **v1.1 - Enhanced Experiments** (Phase 2+)
- Full "Experiments" tab with advanced filtering
- Experiment comparison (side-by-side metrics)
- Export wizard for YouTube/Twitter
- Thumbnail previews in video gallery
- Batch video operations

### **v2.0 - Full MLOps** (Phase 3)
- "Models" tab - Model versioning and management
- "Datasets" tab - Training data management
- Model comparison tools
- Hyperparameter search
- Remote training support
- Multi-user roles (student/instructor)

---

**Document Version:** 1.0
**Date:** 2025-01-26
**Status:** Ready for Implementation
**Next Step:** Begin Phase 1, Week 1 tasks
