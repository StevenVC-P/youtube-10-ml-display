# Dashboard Overview Improvements Plan

**Created:** 2025-11-30  
**Priority:** Medium (after 10-hour video completion)  
**Status:** Planning

---

## 🐛 Current Issues

### Issue 1: Multiple "Running" Runs Displayed

**Problem:** Dashboard shows 9 runs total, with 5 marked as "running", causing confusion about which run is actually active.

**Evidence (from screenshot 2025-11-30 ~03:30):**

- run-54d8ae4e: 51.7% ✅ (actual current 10-hour run)
- run-e237d255: 12.6% ❌ (old run, should be completed)
- run-7624ae0c: 12.6% ❌ (old run, should be completed)
- run-1211d6e4: 32.8% ❌ (old stale run)
- run-ff7a56a4: 12.6% ❌ (old run, should be completed)
- run-974d17a6: 9.4% (shows as "Stopped" but still in list)
- run-6b6f5b95: 14.0% (shows as "Stopped" but still in list)
- run-f6c4cc8a: 12.6% ✅ (correctly marked as Completed)
- run-3bd82f59: 100.0% ✅ (correctly marked as Completed)

**User Impact:**

- Progress appears to "jump" between different runs when UI refreshes/highlights different rows
- Chart on right may be switching between different Breakout runs
- Unclear which run is the current active training
- Cluttered UI with old/completed runs (9 total, only 1 actually active)

---

### Issue 2: Status Inconsistency

**Problem:** Runs marked as "stopped" in database still show in the active runs list.

**Evidence:**

- run-974d17a6 and run-6b6f5b95 show "● Stopped" status but appear in the dashboard
- Should be filtered out or moved to a separate "Recent Runs" section

---

### Issue 3: Completed Runs Not Auto-Updated

**Problem:** Runs that exceed 100% progress are not automatically marked as "completed".

**Evidence:**

- 3 runs at 100.76% still marked as "running"
- Should auto-transition to "completed" status when progress >= 100%

---

### Issue 4: No Visual Distinction for Active Run

**Problem:** All runs look the same - no clear indicator of which is actively training.

**User Impact:**

- Hard to quickly identify the current training session
- Need to scan progress % to guess which is active

---

## ✅ Proposed Solutions

### Solution 1: Auto-Cleanup of Completed Runs

**Implementation:**

- Add background task to check for runs with progress >= 100%
- Auto-update status from "running" → "completed"
- Move completed runs to a separate "Completed" tab or section

**Code Changes:**

- `tools/retro_ml_desktop/experiment_manager.py`: Add `auto_update_completed_runs()` method
- `tools/retro_ml_desktop/main_simple.py`: Call cleanup on dashboard refresh

**Priority:** High

---

### Solution 2: Filter Dashboard by Status

**Implementation:**

- Add filter dropdown: "Active Only" | "All" | "Completed" | "Stopped"
- Default to "Active Only" to show only truly running experiments
- Store filter preference in user settings

**Code Changes:**

- `tools/retro_ml_desktop/main_simple.py`: Add filter UI component
- Update dashboard query to filter by selected status

**Priority:** High

---

### Solution 3: Visual Highlighting for Active Run

**Implementation:**

- Add visual indicator for the most recent active run:
  - Green pulsing dot or border
  - "ACTIVE NOW" badge
  - Highlight row with different background color
- Sort runs by start_time DESC so newest is on top

**Code Changes:**

- `tools/retro_ml_desktop/main_simple.py`: Add conditional styling to table rows
- Use theme colors (SUCCESS green) for active run

**Priority:** Medium

---

### Solution 4: Separate "Recent Runs" Section

**Implementation:**

- Split dashboard into two sections:
  - **Active Training** (status = "running", progress < 100%)
  - **Recent Runs** (last 10 stopped/completed runs)
- Collapsible sections to reduce clutter

**Code Changes:**

- `tools/retro_ml_desktop/main_simple.py`: Create two separate tables
- Use CollapsibleFrame widget for "Recent Runs"

**Priority:** Medium

---

### Solution 5: Progress Update Smoothing

**Implementation:**

- Instead of showing raw progress %, smooth updates over time
- Cache last 5 progress values and show moving average
- Prevents visual "jumping" when UI refreshes

**Code Changes:**

- `tools/retro_ml_desktop/main_simple.py`: Add progress smoothing logic
- Store recent values in memory, calculate rolling average

**Priority:** Low (nice-to-have)

---

### Solution 6: Auto-Refresh Indicator

**Implementation:**

- Add visual indicator when dashboard is refreshing
- Show "Last updated: X seconds ago"
- Prevent confusion about stale data

**Code Changes:**

- `tools/retro_ml_desktop/main_simple.py`: Add refresh timestamp label
- Add subtle loading indicator during refresh

**Priority:** Low

---

## 🎯 Implementation Phases

### Phase 1: Critical Fixes (1-2 hours)

**Goal:** Stop the confusion immediately

- [ ] Auto-update completed runs (Solution 1)
- [ ] Add "Active Only" filter (Solution 2)
- [ ] Sort by start_time DESC

**Deliverable:** Dashboard shows only truly active runs by default

---

### Phase 2: UX Improvements (2-3 hours)

**Goal:** Make it obvious which run is current

- [ ] Visual highlighting for active run (Solution 3)
- [ ] Separate "Recent Runs" section (Solution 4)
- [ ] Add refresh timestamp

**Deliverable:** Clear visual hierarchy, easy to identify current training

---

### Phase 3: Polish (1-2 hours)

**Goal:** Smooth out the experience

- [ ] Progress smoothing (Solution 5)
- [ ] Auto-refresh indicator (Solution 6)
- [ ] User preference persistence

**Deliverable:** Professional, polished dashboard experience

---

## 📊 Success Metrics

**Before:**

- 5+ runs showing as "running"
- Progress appears to jump (12% → 46% → 11%)
- User confusion about which run is active

**After:**

- Only 1 active run visible by default
- Stable progress display (no jumping)
- Clear visual indicator of current training
- Old runs cleanly organized in "Recent" section

---

## 🔧 Technical Notes

### Files to Modify:

1. `tools/retro_ml_desktop/main_simple.py` (Dashboard UI)
2. `tools/retro_ml_desktop/experiment_manager.py` (Status management)
3. `tools/retro_ml_desktop/ml_database.py` (Query optimization)

### Database Changes:

- No schema changes needed
- Add index on `status` column for faster filtering (optional)

### Testing:

- Test with multiple runs in different states
- Verify filter works correctly
- Check auto-cleanup doesn't affect active runs

---

## 📅 Timeline

**Start:** After 10-hour video is complete  
**Estimated Duration:** 4-7 hours total  
**Phases:** 3 phases over 1-2 days

---

## 💡 Additional Ideas (Future)

- Add "Archive" feature to hide old runs
- Export run history to CSV
- Search/filter by game, algorithm, date range
- Pin favorite runs to top
- Bulk actions (stop all, archive all completed)

---

**This plan will be executed after the current 10-hour training completes and the video is generated.**
