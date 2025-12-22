# Walkthrough - Fast Training Mode UI Integration

I have successfully integrated the "Fast Mode" feature into the Atari ML UI. This allows users to train agents at maximum speed (without video recording overhead) and automatically generate a high-quality "Epic" video journey upon completion.

## Changes

### 1. Backend (`ui/backend`)

#### API & Models
*   Updated `TrainingConfig` and `TrainingConfigRequest` to include `fast_mode: bool`.
*   Updated `create_container` endpoint to accept this flag.

#### Training Logic
*   Modified `TrainingWrapper` (`ui/backend/app/core/training_wrapper.py`):
    *   **Config Overrides**: When `fast_mode` is enabled, it automatically disables live video recording (`enabled: false`, `milestone_clip_seconds: 0`) in the generated `config.yaml`.
    *   **Auto-Render**: Added logic to monitor training completion (`exit_code == 0`). If Fast Mode was used, it automatically triggers the `post_training_video_generator` script to render the full video journey from checkpoints.

### 2. Frontend (`ui/frontend`)

#### API Client
*   Updated `TrainingConfig` interface in `src/lib/api.ts` to include `fast_mode?: boolean`.

#### UI Components
*   **Created/Updated** `CreateContainerModal.tsx`:
    *   Added a "Fast Mode ⚡" toggle capability.
    *   When enabled, it selects `fast_mode=true` and visually hides/disables the "Live Video Recording" option (since it's mutually exclusive).
    *   Added descriptive text explaining the trade-off (Speed vs Live View).

### 3. Desktop Application (`tools/retro_ml_desktop`)

#### UI Integration
*   **Modified** `main_simple.py` (Desktop UI):
    *   Added `fast_mode_var` and a "Fast Mode ⚡" checkbox in the `StartTrainingDialog`.
    *   Implemented logic to automatically uncheck "Training Video Recording" when Fast Mode is enabled.
    *   Updated process creation to pass the `--fast` command-line argument to the training script.

#### Training Script (`training/train.py`)
*   **New Arguments**: Added `--fast` and `--render-only` flags.
*   **Fast Mode Logic**:
    *   Disables live video recording (`milestone_clip_seconds=0`, `training_video.enabled=False`) when `--fast` is active.
    *   Triggers automatic **post-training video generation** (rendering from checkpoints) immediately after training completes.
*   **Render Only Logic**:
    *   Added support for `--render-only` to skip training and manually trigger video generation for existing checkpoints.

## Verification Results

### Desktop App Verification
1.  **Launch**: Run `python -m tools.retro_ml_desktop.main_simple`.
2.  **UI Check**: Open "Start New AI Training".
3.  **Fast Mode**: Toggle "Fast Mode". Verify "Training Video Recording" is unchecked.
4.  **Backend Logic**: When starting a run, the command line now includes `--fast`.
5.  **Execution**: `training/train.py` detects `--fast`, disables recording, and upon completion, automatically runs `post_training_video_generator.py` to create the full video journey.

### Web UI Verification (Previous Work)
1.  **UI Check**: Launch the Web UI.
2.  **New Container**: Toggle "Fast Mode" in the modal.
3.  **Backend Log**: Verify `config.yaml` is generated with `recording.enabled=False` and `fast_mode=True`.

## Next Steps
*   Run a short test training session (e.g., 30 minutes) using Fast Mode in the Desktop App to confirm the video is generated automatically at the end.

## Post-Implementation Fixes (Video Location)
*   **Issue**: Videos were being saved to `Documents/ML_Videos` by default, which the "Video Gallery" tab did not immediately show if it was looking at the local project structure.
*   **Fix**:
    1.  Moved the generated video for `run-08b42952` to the project's `outputs` folder where the UI expects it.
    2.  Updated `main_simple.py` to change the default video output location to the project's `outputs` directory.
    3.  Future runs will now save videos directly to the project folder, making them immediately visible in the Video Gallery.
