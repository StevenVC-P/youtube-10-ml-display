# RetroML Trainer - Feature Roadmap

This document tracks the features and requirements for the RetroML Trainer desktop application.

## Core Requirements

### Installation & Setup
- [ ] **One-click installation**
  - No manual Python setup required
  - No Gymnasium installation needed
  - Bundled dependencies
  - Automatic environment setup

### Training Capabilities
- [ ] **Train models on included Atari games**
  - Support for PPO algorithm
  - Support for DQN algorithm
  - Pre-configured game environments
  - Pre-set training configurations with sensible defaults

### Video Generation
- [ ] **Automatic video generation**
  - Training visualization
  - Final "AI plays the game" output video
  - Automated render pipeline

### Metrics & Monitoring
- [ ] **Basic metrics display**
  - Episode reward tracking
  - Loss curve visualization
  - Training duration
  - FPS or steps/sec counter

### User Experience
- [ ] **Simple UI / Streamlined UX**
  - "Train → Render → Export" workflow
  - No technical experience required
  - Everything runs locally
  - Defaults tuned for fun results
  - Beginner-friendly interface

## Stretch Features (Optional)

### Enhanced Visualization
- [ ] **Basic overlay on videos**
  - Current score display
  - Timestep counter
  - Training speed indicator

### Training Insights
- [ ] **"Time-lapse" training video generator**
  - Show AI learning progression
  - Compress 10+ hours into minutes
  - Milestone comparison

### User Engagement
- [ ] **Leaderboard of user's best runs**
  - Track personal best scores per game
  - Compare training runs
  - Achievement system

### Model Management
- [ ] **Simple model loading**
  - "Replay your previous run" feature
  - Load and continue training
  - Export/import trained models
  - Model comparison tools

## Implementation Status

### Completed Features
- ✅ Core ML training pipeline (PPO algorithm)
- ✅ Multi-game support (6 Atari games)
- ✅ Video generation system
- ✅ Epic training sessions (10-hour sessions)
- ✅ Sequential training pipeline
- ✅ Automated checkpoint system
- ✅ Basic desktop UI

### In Progress
- 🔄 Professional installer with bundled dependencies
- 🔄 One-click installation experience
- 🔄 Simplified training workflow
- 🔄 Basic metrics dashboard

### Planned
- 📋 Enhanced video overlays
- 📋 Time-lapse video generation
- 📋 Personal leaderboard system
- 📋 Model replay functionality

## Target User Experience

The ideal user journey:
1. **Download** - Single installer file
2. **Install** - One-click installation, all dependencies bundled
3. **Launch** - Desktop application with friendly UI
4. **Select** - Choose a game and training duration
5. **Train** - Hit "Start Training" and watch progress
6. **Watch** - Automatic video generation of AI learning
7. **Share** - Export video and share results

## Technical Requirements

- **Platform**: Windows 10/11
- **GPU Support**: Optional CUDA acceleration
- **Python**: Bundled (user doesn't need to install)
- **Dependencies**: All included in installer
- **Storage**: ~2GB for installation, variable for models/videos

## Success Criteria

The application is ready for release when:
- ✅ Non-technical users can install without help
- ✅ Training starts with a single button click
- ✅ Videos are automatically generated
- ✅ Results are viewable within the application
- ✅ No command-line interaction required
- ✅ Error messages are clear and actionable
- ✅ Installation includes all necessary dependencies

## Notes

- Focus on user-friendliness over advanced features
- Prioritize "works out of the box" over configurability
- Default settings should produce fun, shareable results
- Keep technical complexity hidden from the user
