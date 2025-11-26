# Retro ML Trainer - Distribution Guide

This guide explains how to build and distribute Retro ML Trainer as a standalone Windows application.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Application](#building-the-application)
3. [Distribution Options](#distribution-options)
4. [User Installation Guide](#user-installation-guide)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For Building

You need the following installed on your development machine:

1. **Python 3.11+** with all project dependencies
2. **PyInstaller** (will be auto-installed by build script)
3. **Inno Setup** (for creating installer) - Download from https://jrsoftware.org/isdl.php
4. **Internet connection** (for downloading FFmpeg)

### System Requirements for End Users

- **OS:** Windows 10/11 (64-bit)
- **RAM:** 2GB minimum (8GB+ recommended)
- **Disk Space:** 
  - 2GB for application
  - 50GB+ recommended for training data and videos
- **Optional:** NVIDIA GPU with CUDA support for faster training

---

## Building the Application

### Option 1: Complete Build (Recommended)

Build both the executable and installer in one step:

```batch
cd "D:\Python projects\ML project\Atari ML project"
build_scripts\build_all.bat
```

This will:
1. ✅ Install PyInstaller if needed
2. ✅ Download FFmpeg binaries
3. ✅ Build standalone executable
4. ✅ Create Windows installer (.exe)

**Output:**
- Executable: `dist/RetroMLTrainer/`
- Installer: `installer_output/RetroMLTrainer-Setup-1.0.0.exe`

### Option 2: Executable Only (For Testing)

Build just the executable without creating an installer:

```batch
build_scripts\build_exe_only.bat
```

**Output:**
- Executable: `dist/RetroMLTrainer/RetroMLTrainer.exe`

### Manual Build Steps

If you prefer manual control:

```batch
# 1. Install PyInstaller
pip install pyinstaller

# 2. Build executable
python build_scripts/build_executable.py

# 3. Create installer (requires Inno Setup)
iscc build_scripts/installer.iss
```

---

## Distribution Options

### Option A: Installer (Recommended for Non-Technical Users)

**File:** `installer_output/RetroMLTrainer-Setup-1.0.0.exe`

**Pros:**
- ✅ Professional installation experience
- ✅ Creates Start Menu shortcuts
- ✅ Handles uninstallation cleanly
- ✅ Checks system requirements
- ✅ Single file to distribute

**Size:** ~800MB - 1.2GB

**How to distribute:**
1. Upload to cloud storage (Dropbox, Google Drive, OneDrive)
2. Share download link with your friend
3. They download and run the installer

### Option B: Portable Folder (For Advanced Users)

**Folder:** `dist/RetroMLTrainer/`

**Pros:**
- ✅ No installation required
- ✅ Can run from USB drive
- ✅ Easy to move/copy

**Cons:**
- ❌ No shortcuts created
- ❌ Larger download (uncompressed)

**How to distribute:**
1. Zip the entire `dist/RetroMLTrainer/` folder
2. Upload to cloud storage
3. User extracts and runs `RetroMLTrainer.exe`

---

## User Installation Guide

### Using the Installer (Recommended)

1. **Download** the installer: `RetroMLTrainer-Setup-1.0.0.exe`

2. **Run** the installer
   - Double-click the downloaded file
   - Windows may show a security warning (click "More info" → "Run anyway")

3. **Follow** the installation wizard
   - Choose installation location (default: `C:\Program Files\Retro ML Trainer`)
   - Optionally create desktop shortcut
   - Click "Install"

4. **Launch** the application
   - From Start Menu: "Retro ML Trainer"
   - Or from desktop shortcut (if created)

5. **Complete** first-run setup wizard
   - Choose where to store training data
   - Configure storage paths
   - System capability check
   - Install Atari ROMs (requires license acceptance)
   - Initialize database

6. **Start training!**
   - Go to "Training" tab
   - Select a game (e.g., Breakout)
   - Click "Start Training"

### Using the Portable Version

1. **Download** and **extract** the zip file

2. **Navigate** to the extracted folder

3. **Run** `RetroMLTrainer.exe`

4. **Complete** first-run setup wizard (same as above)

---

## First-Run Setup Wizard

The setup wizard guides users through initial configuration:

### Step 1: Welcome
- Introduction to the application
- System requirements overview

### Step 2: Installation Location
- Choose where to install (default: `C:\Users\{username}\RetroMLTrainer`)
- Shows available disk space
- Recommends 50GB+ free space

### Step 3: Storage Configuration
- Configure paths for:
  - Models directory (training checkpoints)
  - Videos directory (generated videos)
  - Database directory (experiment tracking)
- Can use relative or absolute paths

### Step 4: System Check
- Detects GPU/CUDA availability
- Checks for FFmpeg
- Verifies Atari ROMs
- Shows warnings if components missing

### Step 5: Atari ROM Installation
- License acceptance required
- Auto-downloads ROMs using AutoROM
- Can skip and install manually later

### Step 6: Finish
- Shows installation summary
- Creates all configured directories
- Initializes database
- Launches main application

---

## Troubleshooting

### Build Issues

**Problem:** PyInstaller fails with "module not found"
```
Solution: Add missing module to hiddenimports in retro_ml_trainer.spec
```

**Problem:** FFmpeg download fails
```
Solution: Manually download FFmpeg from https://ffmpeg.org/download.html
         Place ffmpeg.exe in build_scripts/ffmpeg/
```

**Problem:** Build is too large (>2GB)
```
Solution: Using CPU-only PyTorch reduces size significantly
         Exclude unnecessary packages in retro_ml_trainer.spec
```

### Installation Issues

**Problem:** Windows Defender blocks installer
```
Solution: Click "More info" → "Run anyway"
         Or add exception in Windows Security
```

**Problem:** "Not enough disk space" error
```
Solution: Free up disk space or choose different installation location
```

**Problem:** ROM installation fails
```
Solution: Install manually: pip install autorom[accept-rom-license]
         Then run: AutoROM --accept-license
```

### Runtime Issues

**Problem:** Application won't start
```
Solution: Check Windows Event Viewer for error details
         Run from command line to see error messages
```

**Problem:** "Database locked" error
```
Solution: Close any other instances of the application
         Delete ml_experiments.db-wal and ml_experiments.db-shm files
```

**Problem:** Video generation fails
```
Solution: Ensure FFmpeg is bundled (check for ffmpeg.exe in app folder)
         Or install FFmpeg system-wide
```

**Problem:** Training is very slow
```
Solution: Check if GPU is detected in System Check
         If no GPU, training will use CPU (slower)
         Consider installing CUDA drivers for GPU acceleration
```

---

## Advanced Configuration

### Customizing the Build

Edit `retro_ml_trainer.spec` to:
- Add/remove bundled files
- Include custom icons
- Exclude unnecessary packages
- Adjust compression settings

### Customizing the Installer

Edit `build_scripts/installer.iss` to:
- Change app name/version
- Modify installation defaults
- Add custom installation steps
- Include additional files

### Creating a Portable Version

To create a truly portable version:

1. Build executable as normal
2. Copy `dist/RetroMLTrainer/` to USB drive
3. Create `portable.txt` file in root (signals portable mode)
4. Application will store all data in its own directory

---

## Distribution Checklist

Before distributing to your friend:

- [ ] Build completed successfully
- [ ] Tested executable on clean Windows VM
- [ ] First-run wizard works correctly
- [ ] Training starts and runs
- [ ] Video generation works
- [ ] Database initializes properly
- [ ] All features tested
- [ ] README included
- [ ] User guide provided
- [ ] Support contact information included

---

## File Size Optimization

To reduce distribution size:

1. **Use CPU-only PyTorch** (~500MB smaller)
2. **Exclude test files** (already done in spec)
3. **Use UPX compression** (already enabled)
4. **Remove unnecessary packages**
5. **Consider 7-Zip** for better compression than ZIP

Expected sizes:
- **Executable folder:** ~1.5GB - 2GB
- **Installer:** ~800MB - 1.2GB (compressed)
- **Zipped folder:** ~600MB - 900MB

---

## Support

For issues or questions:
- Check troubleshooting section above
- Review application logs in `logs/` directory
- Contact developer with error details

---

## License

Include appropriate license information for:
- Your application
- Bundled dependencies (PyTorch, Gymnasium, etc.)
- Atari ROMs (user accepts license during setup)
- FFmpeg (LGPL/GPL depending on build)

