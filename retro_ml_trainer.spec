# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Retro ML Trainer

This creates a standalone Windows executable with all dependencies bundled.
"""

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Project root
project_root = Path('.').absolute()

# Find PyTorch installation to get CUDA DLLs
import torch
torch_lib_path = Path(torch.__file__).parent / 'lib'

# Collect all data files from various packages
datas = []

# CustomTkinter themes and assets
datas += collect_data_files('customtkinter')

# Gymnasium/ALE-Py data files
datas += collect_data_files('gymnasium')
datas += collect_data_files('ale_py')

# Add configuration files
datas += [
    ('conf/config.yaml', 'conf'),
    ('tools/retro_ml_desktop/training_presets.yaml', 'tools/retro_ml_desktop'),
]

# Add any documentation or README files you want to include
datas += [
    ('README.md', '.'),
]

# Collect hidden imports (modules that PyInstaller might miss)
hiddenimports = []

# Stable-Baselines3 and dependencies
hiddenimports += collect_submodules('stable_baselines3')
hiddenimports += collect_submodules('gymnasium')
hiddenimports += collect_submodules('ale_py')

# PyTorch (CPU-only to reduce size)
hiddenimports += collect_submodules('torch')

# Other ML/scientific packages
hiddenimports += [
    'numpy',
    'cv2',
    'PIL',
    'yaml',
    'matplotlib',
    'psutil',
    'sqlite3',
]

# CustomTkinter
hiddenimports += collect_submodules('customtkinter')

# Our own modules
hiddenimports += [
    'tools.retro_ml_desktop.config_manager',
    'tools.retro_ml_desktop.setup_wizard',
    'tools.retro_ml_desktop.main_simple',
    'tools.retro_ml_desktop.process_manager',
    'tools.retro_ml_desktop.monitor',
    'tools.retro_ml_desktop.ml_database',
    'tools.retro_ml_desktop.ml_collector',
    'tools.retro_ml_desktop.ml_dashboard',
    'tools.retro_ml_desktop.video_player',
    'training.train',
    'training.post_training_video_generator',
    'agents.algo_factory',
    'envs.make_env',
]

# Collect CUDA DLLs if available
binaries = []
print(f"🔍 Checking CUDA availability...")
print(f"   torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"   torch_lib_path = {torch_lib_path}")

if torch.cuda.is_available():
    print(f"✅ CUDA is available! Collecting CUDA DLLs...")
    # Add ALL DLLs from torch/lib directory
    if torch_lib_path.exists():
        for dll_file in torch_lib_path.glob('*.dll'):
            binaries.append((str(dll_file), 'torch/lib'))
            print(f"   📦 Adding: {dll_file.name}")

    print(f"✅ Total CUDA DLLs collected: {len(binaries)}")
else:
    print(f"❌ CUDA not available - skipping CUDA DLLs")

# Analysis
a = Analysis(
    ['tools/retro_ml_desktop/launcher.py'],  # Entry point
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter.test',
        'unittest',
        'test',
        'tests',
        'pytest',
        'IPython',
        'jupyter',
        'notebook',
        # DO NOT exclude CUDA - we want GPU support!
        # 'torch.cuda',
        # 'torch.backends.cudnn',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# PYZ (Python zip archive)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# EXE (executable)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RetroMLTrainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress with UPX
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # TODO: Add icon file if you have one
)

# COLLECT (collect all files into a folder)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RetroMLTrainer',
)

