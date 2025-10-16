# 🎮 ML Training System - Command Reference

## Quick Start

```powershell
# Load custom commands
. .\scripts\ml_commands.ps1

# Train any game for any duration
Train-Game breakout 5h
Train-Game tetris 30min -Test

# Generate videos
Make-Video space_invaders 2h

# Train entire systems
Train-System atari -Hours 8 -Epic 1
```

---

## 📋 Available Commands

### 🎮 **Core Training Commands**

#### **1. Epic Training System (Recommended)**
```powershell
# Basic syntax
python epic_training/scripts/train_epic_continuous.py --game <GAME> --epic <NUM> [OPTIONS]

# Examples
python epic_training/scripts/train_epic_continuous.py --game breakout --epic 1 --hours 10
python epic_training/scripts/train_epic_continuous.py --game tetris --epic 1 --test
python epic_training/scripts/train_epic_continuous.py --game space_invaders --epic 2 --hours 5

# Parameters:
#   --game: breakout, pong, space_invaders, asteroids, pacman, frogger, tetris
#   --epic: 1=from_scratch, 2=advanced_mastery, 3=perfect_play
#   --hours: Training duration (default: 10)
#   --test: Quick test mode (~1 minute)
```

#### **2. Custom Training Commands (New!)**
```powershell
# Train specific game with flexible duration
python scripts/custom_training_commands.py train --game <GAME> --hours <NUM> [OPTIONS]

# Generate videos of specific lengths
python scripts/custom_training_commands.py video --game <GAME> --length <DURATION> [OPTIONS]

# Batch training multiple games
python scripts/custom_training_commands.py batch --games "game1,game2" --hours <NUM> [OPTIONS]

# Train entire gaming system
python scripts/custom_training_commands.py system-train --system <SYSTEM> --hours <NUM> [OPTIONS]
```

#### **3. PowerShell Shortcuts (Easiest!)**
```powershell
# Source the commands first
. .\scripts\ml_commands.ps1

# Then use simple functions
Train-Game breakout 5h
Train-Game tetris 30min -Test
Make-Video space_invaders 2h -Epic 2
Train-System atari -Hours 8 -Epic 1
```

---

## 🎯 **Practical Examples**

### **Training Different Games**

```powershell
# Atari Games
Train-Game breakout 10h          # Full 10-hour Breakout training
Train-Game pong 5h -Epic 2       # 5-hour Pong advanced training
Train-Game space_invaders 2h     # 2-hour Space Invaders training

# Gameboy Games  
Train-Game tetris 8h             # 8-hour Tetris training
Train-Game tetris 30min -Test    # Quick Tetris test

# Quick tests (1 minute each)
Quick-Train breakout
Quick-Train tetris
```

### **Video Generation**

```powershell
# Different video lengths
Make-Video breakout 30min        # 30-minute Breakout video
Make-Video tetris 2h             # 2-hour Tetris video
Make-Video space_invaders 90m    # 90-minute Space Invaders video

# Different epic levels
Make-Video breakout 1h -Epic 1   # Beginner level video
Make-Video breakout 1h -Epic 3   # Expert level video

# Different quality settings
Make-Video tetris 30min -Quality high
```

### **Batch Operations**

```powershell
# Train multiple games
Train-Batch "breakout,tetris,pong" 2h -Epic 1
Train-Batch "space_invaders,asteroids" 1h -Test

# Train entire gaming systems
Train-System atari -Hours 5 -Epic 1      # All Atari games, 5h each
Train-System gameboy -Hours 3 -Test      # All Gameboy games, test mode
```

### **Complete Learning Journeys**

```powershell
# Preset journeys (3 epics each)
Start-BreakoutJourney    # Epic 1 → Epic 2 → Epic 3 (30 hours total)
Start-TetrisJourney      # Epic 1 → Epic 2 → Epic 3 (30 hours total)

# Manual journey
Train-Game breakout 10h -Epic 1
Train-Game breakout 10h -Epic 2  
Train-Game breakout 10h -Epic 3
```

---

## 🎮 **Supported Games & Systems**

### **Atari System** (6 games)
- `breakout` - Classic brick-breaking game
- `pong` - Original paddle tennis game  
- `space_invaders` - Alien shooting game
- `asteroids` - Space rock destruction
- `pacman` - Maze navigation and pellet collection
- `frogger` - Road and river crossing

### **Gameboy System** (1 game, more coming)
- `tetris` - Block puzzle game ✅ **Fully Working**
- `super_mario_land` - Platform adventure (🚧 Coming Soon)
- `kirbys_dream_land` - Platform adventure (🚧 Coming Soon)

---

## ⚙️ **Configuration & Customization**

### **Duration Formats**
```powershell
# All these are valid duration formats:
Train-Game breakout 5h           # 5 hours
Train-Game tetris 30min          # 30 minutes  
Train-Game pong 90m              # 90 minutes
Train-Game asteroids 2.5h        # 2.5 hours
Train-Game frogger 120min        # 120 minutes
```

### **Epic Types**
- **Epic 1** (`from_scratch`): Complete learning from zero to competent play
- **Epic 2** (`advanced_mastery`): Building on competent play to expert level  
- **Epic 3** (`perfect_play`): Expert level to flawless execution

### **Test Mode**
```powershell
# Add -Test flag for quick 1-minute tests
Train-Game tetris 10h -Test      # Actually runs ~1 minute
Quick-Train breakout             # Dedicated quick test function
```

---

## 🔧 **Utility Commands**

### **Status & Monitoring**
```powershell
Check-Epic-Status               # Check current training status
Get-Process | Where-Object {$_.ProcessName -like "*python*"}  # Check running processes
```

### **Model Evaluation**
```powershell
# Evaluate trained models
Eval-Model "models\checkpoints\latest.zip" -Episodes 5
Eval-Model "games\breakout\epic_001_from_scratch\models\final\epic_001_final_model.zip" -Deterministic
```

### **Testing & Demos**
```powershell
Demo-AllGames                   # Quick test all supported games
Quick-Train <game>              # 1-minute test of specific game
```

---

## 📁 **File Structure & Outputs**

### **Training Outputs**
```
games/
├── breakout/
│   ├── epic_001_from_scratch/
│   │   ├── models/checkpoints/     # Training checkpoints
│   │   ├── models/final/           # Final trained model
│   │   ├── videos/individual_hours/ # Hourly progress videos
│   │   └── logs/tensorboard/       # Training logs
│   ├── epic_002_advanced_mastery/
│   └── epic_003_perfect_play/
└── tetris/
    └── epic_001_from_scratch/
        ├── models/
        ├── videos/
        └── logs/
```

### **Video Outputs**
```
video/
├── milestones/          # Milestone videos (10%, 20%, etc.)
├── eval/               # Evaluation videos
└── render/             # Final rendered videos
```

---

## 🚀 **Advanced Usage**

### **Creating Custom Commands**

You can extend the system by modifying `scripts/custom_training_commands.py`:

```python
# Add new games to GAME_SYSTEMS
GAME_SYSTEMS = {
    "new_game": "new_system",
    # ... existing games
}

# Add new system configurations
SYSTEM_CONFIGS = {
    "new_system": {
        "base_config": "conf/config.yaml",
        "epic_script": "epic_training/scripts/train_epic_continuous.py", 
        "supported_games": ["new_game"]
    }
}
```

### **Environment Variables**
```powershell
# Set custom paths
$env:EPIC_DIR = "games\tetris\epic_001_from_scratch"
$env:PYTHONPATH = "."
```

---

## 💡 **Tips & Best Practices**

1. **Start with test mode** to verify everything works:
   ```powershell
   Train-Game tetris 10h -Test
   ```

2. **Use shorter durations** for initial experiments:
   ```powershell
   Train-Game breakout 1h
   ```

3. **Monitor training** with status checks:
   ```powershell
   Check-Epic-Status
   ```

4. **Generate videos** after training completes:
   ```powershell
   Make-Video breakout 30min -Epic 1
   ```

5. **Use batch training** for efficiency:
   ```powershell
   Train-Batch "breakout,tetris" 2h -Test
   ```

---

## 🆘 **Troubleshooting**

### **Common Issues**

1. **"Game not supported"** - Check available games with `Show-MLCommands`
2. **"Model not found"** - Train the model first before generating videos
3. **"Python not found"** - Ensure virtual environment is activated
4. **Memory errors** - Use test mode or shorter durations

### **Getting Help**
```powershell
Show-MLCommands                 # Show all available commands
python scripts/custom_training_commands.py --help
python epic_training/scripts/train_epic_continuous.py --help
```

---

This command system gives you complete flexibility to train any supported game for any duration and generate videos of any length across multiple gaming systems! 🎮✨
