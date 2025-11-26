# Retro ML Trainer - User Guide

Welcome to Retro ML Trainer! This guide will help you get started with training AI agents to play classic Atari games.

## 🎮 What is Retro ML Trainer?

Retro ML Trainer is a desktop application that lets you:
- **Train AI agents** to play Atari games using machine learning
- **Watch progress** through automatically generated videos
- **Track performance** with detailed metrics and charts
- **Manage multiple training sessions** simultaneously

No coding required - everything is done through an easy-to-use interface!

---

## 📥 Installation

### Step 1: Download

You should have received a file called:
- `RetroMLTrainer-Setup-1.0.0.exe` (installer)

Or a zip file containing the portable version.

### Step 2: Install

**Using the Installer (Recommended):**

1. Double-click `RetroMLTrainer-Setup-1.0.0.exe`
2. If Windows shows a security warning:
   - Click "More info"
   - Click "Run anyway"
3. Follow the installation wizard
4. Choose where to install (default is fine)
5. Click "Install"

**Using the Portable Version:**

1. Extract the zip file to a folder (e.g., `C:\RetroMLTrainer`)
2. Open the folder
3. Double-click `RetroMLTrainer.exe`

### Step 3: First-Run Setup

When you first launch the app, a setup wizard will appear:

#### Welcome Screen
- Read the introduction
- Click "Next"

#### Installation Location
- Choose where to store the app's data
- Default: `C:\Users\YourName\RetroMLTrainer`
- Make sure you have at least 50GB free space
- Click "Next"

#### Storage Configuration
- Choose where to store:
  - **Models** - AI training checkpoints
  - **Videos** - Generated gameplay videos
  - **Database** - Training history
- Default locations are fine for most users
- Click "Next"

#### System Check
- The app will check your computer:
  - ✅ GPU detected? (faster training)
  - ✅ FFmpeg available? (for videos)
  - ✅ Atari ROMs installed?
- Don't worry if some items show ❌ - the app will still work
- Click "Next"

#### Atari ROM Installation
- **Important:** You must accept the license to use Atari games
- Check "I accept the Atari 2600 ROM license"
- Click "Install ROMs"
- Wait for download to complete
- Click "Next"

#### Finish
- Review your settings
- Click "Finish"
- The main application will launch!

---

## 🚀 Quick Start Guide

### Your First Training Session

1. **Open the Training Tab**
   - Click the "Training" tab at the top

2. **Choose a Game**
   - Select a game from the dropdown (try "Breakout" first)

3. **Select Algorithm**
   - Choose "PPO" (recommended for beginners)

4. **Set Training Duration**
   - Start with 1 hour for testing
   - Enter: Hours: 1, Minutes: 0

5. **Start Training**
   - Click "Start Training"
   - The app will begin training the AI

6. **Watch Progress**
   - Go to the "Processes" tab to see training status
   - Go to the "Videos" tab to see generated videos

---

## 📊 Understanding the Interface

### Main Tabs

#### 🎮 Training Tab
- **Start new training sessions**
- Configure game, algorithm, and duration
- Choose resource allocation (CPU/GPU)
- Set advanced options

#### 📈 Dashboard Tab
- **View training metrics**
- See reward progress over time
- Monitor episode length
- Track learning curves

#### 🎬 Videos Tab
- **Watch AI gameplay videos**
- Videos are automatically generated during training
- See progress from beginner to expert
- Compare different training runs

#### ⚙️ Processes Tab
- **Monitor active training sessions**
- See CPU/GPU usage
- View logs in real-time
- Stop/pause training

#### 📊 Analytics Tab
- **Deep dive into performance**
- Compare multiple training runs
- Export data for analysis
- View detailed statistics

---

## 🎯 Training Tips

### For Beginners

1. **Start Small**
   - Begin with 1-hour training sessions
   - Use simple games like Breakout or Pong
   - Stick with PPO algorithm

2. **Watch the Videos**
   - Videos show AI progress
   - Early videos: random movements
   - Later videos: strategic play

3. **Be Patient**
   - Training takes time
   - 1 hour = basic understanding
   - 10+ hours = competent play
   - 50+ hours = expert level

### For Advanced Users

1. **Experiment with Algorithms**
   - **PPO:** Best for most games, stable
   - **DQN:** Good for discrete actions
   - **A2C:** Faster but less stable

2. **Adjust Hyperparameters**
   - Learning rate: How fast AI learns
   - Batch size: Training efficiency
   - Gamma: Future reward importance

3. **Use GPU Acceleration**
   - Much faster training
   - Requires NVIDIA GPU with CUDA
   - Check System tab for GPU status

---

## 🎬 Video Generation

### Automatic Videos

The app automatically generates videos during training:

- **Milestone Videos** (90 seconds each)
  - At 0%, 1%, 5%, 10%, 25%, 50%, 75%, 90%, 100% progress
  - Shows AI improvement over time

- **Hour Videos** (30-60 minutes each)
  - One video per hour of training
  - Longer gameplay sessions

### Manual Video Generation

You can also generate videos after training:

1. Go to "Videos" tab
2. Click "Generate Videos from Training"
3. Select a training run
4. Choose video length
5. Click "Generate"

### Video Progress Tracking

While videos are being generated:
- They appear in the gallery with 🎬 icon
- Shows progress percentage
- Displays estimated time remaining
- Grayed out until complete

---

## 💾 Managing Storage

### Disk Space Usage

Training can use a lot of disk space:
- **Models:** ~1-5GB per training run
- **Videos:** ~500MB - 2GB per hour
- **Database:** ~10-100MB

### Cleaning Up

To free up space:

1. **Delete Old Videos**
   - Go to Videos tab
   - Select videos you don't need
   - Click "Delete Video"

2. **Remove Old Training Runs**
   - Go to Processes tab
   - Select completed runs
   - Click "Clean Up"

3. **Use Storage Cleaner**
   - Click "Storage" in the sidebar
   - Review disk usage
   - Delete unnecessary files

---

## ❓ Troubleshooting

### Training Won't Start

**Problem:** "Failed to start training"

**Solutions:**
- Check if another training is already running
- Verify game ROMs are installed
- Check available disk space
- Review error message in logs

### Videos Not Generating

**Problem:** No videos appear in gallery

**Solutions:**
- Check if FFmpeg is installed (System Check)
- Verify video directory has write permissions
- Check available disk space
- Look for errors in logs

### Slow Training

**Problem:** Training is very slow

**Solutions:**
- Check if GPU is being used (Processes tab)
- Close other applications
- Reduce training complexity
- Consider upgrading hardware

### Application Crashes

**Problem:** App closes unexpectedly

**Solutions:**
- Check logs in `logs/` directory
- Ensure enough RAM available
- Update graphics drivers
- Restart computer

---

## 🎓 Learning Resources

### Understanding the Metrics

- **Reward:** Points earned by AI (higher is better)
- **Episode Length:** How long AI survives
- **Loss:** Training error (lower is better)
- **FPS:** Training speed (frames per second)

### Game Difficulty

**Easy Games (Good for Learning):**
- Breakout
- Pong

**Medium Games:**
- Space Invaders
- Asteroids

**Hard Games:**
- Pac-Man
- Frogger

---

## 📞 Getting Help

If you encounter issues:

1. **Check this guide** - Most common issues are covered
2. **Review logs** - Located in `logs/` directory
3. **Check system requirements** - Ensure your PC meets minimum specs
4. **Contact support** - Reach out with error details

---

## 🎉 Have Fun!

Remember:
- Training AI is a learning process
- Experiment with different settings
- Watch the videos to see progress
- Share your results!

Enjoy watching your AI agents learn to play!

