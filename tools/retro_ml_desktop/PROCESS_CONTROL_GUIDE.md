# Process Control Guide

## 🎮 **Training Process Control System**

Your ML training manager now has **robust process control** with improved pause, resume, and stop functionality.

### ✅ **What Was Fixed**

**❌ Previous Issues:**
- **Status Confusion**: Paused processes showed as "running" 
- **Child Process Issues**: Only main process was paused/resumed, not child processes
- **Poor Error Handling**: Unclear error messages and no user feedback
- **UI Problems**: No visual indicators for process states

**✅ Now Fixed:**
- **Accurate Status Tracking**: Processes maintain correct status (paused/running/stopped)
- **Complete Process Control**: Handles main process and all child processes
- **Rich User Feedback**: Clear confirmations, error messages, and status updates
- **Visual Status Indicators**: Color-coded buttons and emoji status displays

### 🎯 **Process Control Features**

#### **🛑 Stop Process**
- **Purpose**: Permanently terminate a training process
- **What it does**:
  - Resumes paused processes first (for clean shutdown)
  - Attempts graceful termination (SIGTERM)
  - Force kills if graceful shutdown fails (after 30 seconds)
  - Kills all child processes
  - Saves current training progress
  - Cleans up log streams and resources

**Usage:**
1. Select process in the list
2. Click "🛑 Stop Selected" (red button)
3. Confirm termination
4. Process status changes to "🔴 Stopped"

#### **⏸️ Pause Process**
- **Purpose**: Temporarily suspend training (keeps in memory)
- **What it does**:
  - Suspends main process and all child processes
  - Preserves memory state and training progress
  - Can be resumed later from exact same point
  - No data loss or checkpoint corruption

**Usage:**
1. Select running process
2. Click "⏸️ Pause Selected" (yellow button)
3. Confirm pause
4. Process status changes to "⏸️ Paused"

#### **▶️ Resume Process**
- **Purpose**: Continue paused training from where it left off
- **What it does**:
  - Resumes main process and all child processes
  - Continues training from exact same state
  - No progress loss or data corruption
  - Training metrics continue seamlessly

**Usage:**
1. Select paused process
2. Click "▶️ Resume Selected" (green button)
3. Confirm resume
4. Process status changes to "🟢 Running"

### 📊 **Visual Status Indicators**

**Process List Status Display:**
- **🟢 Running**: Process is actively training
- **⏸️ Paused**: Process is suspended but can be resumed
- **🔴 Stopped**: Process has been terminated
- **✅ Finished**: Process completed successfully

**Button Colors:**
- **🛑 Red Stop Button**: Permanent termination
- **⏸️ Yellow Pause Button**: Temporary suspension
- **▶️ Green Resume Button**: Continue paused training

### 🔧 **Technical Improvements**

#### **Smart Status Tracking**
```python
# Now correctly preserves paused status
if process_info.status != "paused":
    # Only update to running if not explicitly paused
    if proc.status() == psutil.STATUS_STOPPED:
        process_info.status = "paused"
    else:
        process_info.status = "running"
```

#### **Complete Process Control**
```python
# Handles all child processes
children = proc.children(recursive=True)
for child in children:
    child.suspend()  # or resume() or kill()
proc.suspend()  # Main process
```

#### **Robust Error Handling**
- **Process validation**: Checks if process exists and is in correct state
- **Graceful degradation**: Continues even if some child processes fail
- **User feedback**: Clear success/error messages with specific details
- **Logging**: Detailed logs for troubleshooting

### 🚀 **Best Practices**

#### **When to Use Each Control**

**🛑 Stop Process:**
- Training is complete or no longer needed
- Need to free up system resources immediately
- Process is stuck or unresponsive
- Want to start fresh with new configuration

**⏸️ Pause Process:**
- Need system resources temporarily (other tasks)
- System maintenance or restart required
- Want to check intermediate results
- Debugging or configuration changes needed

**▶️ Resume Process:**
- Ready to continue paused training
- System resources are available again
- Maintenance is complete
- Want to continue from exact same point

#### **Safe Operation Guidelines**

1. **Always use Pause instead of Stop** if you plan to continue training
2. **Wait for confirmation messages** before performing other operations
3. **Check Progress Tracking tab** to verify training continues after resume
4. **Monitor system resources** before resuming multiple processes
5. **Use Stop only when certain** you don't need to continue training

### 🔍 **Troubleshooting**

#### **Common Issues & Solutions**

**Problem**: "Failed to pause process"
- **Cause**: Process may have crashed or finished
- **Solution**: Check if process is still running, refresh process list

**Problem**: "Failed to resume process"  
- **Cause**: Process was terminated or system resources unavailable
- **Solution**: Check system memory/CPU, restart training if needed

**Problem**: Process shows "🟢 Running" but no progress
- **Cause**: Process may be stuck or waiting for resources
- **Solution**: Check Progress Tracking tab, consider pause/resume cycle

**Problem**: Resume doesn't continue training
- **Cause**: Training may have reached completion or hit an error
- **Solution**: Check logs tab for error messages, verify training configuration

### 📈 **Monitoring Process Health**

**Use Progress Tracking Tab:**
- **Real-time metrics**: Timesteps, rewards, FPS
- **File generation**: Videos and checkpoints being created
- **ETA estimates**: Time remaining for training completion

**Check Logs Tab:**
- **Training output**: Real-time training logs and metrics
- **Error messages**: Detailed error information if issues occur
- **System messages**: Process control confirmations and status changes

### 🎯 **Perfect for Your ML Workflow**

This enhanced process control system is **ideal for your 10-hour training sessions**:

1. **Start multiple training runs** with different configurations
2. **Pause training** when you need system resources for other tasks
3. **Resume seamlessly** when ready to continue
4. **Stop completed runs** to free up resources
5. **Monitor all processes** with real-time progress tracking

**Ready to use**: Your process control system now provides **complete, reliable control** over your ML training pipeline with **robust error handling** and **clear user feedback**! 🚀
