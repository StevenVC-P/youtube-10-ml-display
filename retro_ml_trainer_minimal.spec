# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Retro ML Trainer - MINIMAL VERSION
This creates a small installer WITHOUT PyTorch (downloads on first run)
"""

import sys
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Project root
project_root = Path('.').absolute()

# Collect all data files from various packages
datas = []

# CustomTkinter themes and assets
datas += collect_data_files('customtkinter')

# Gymnasium/ALE-Py data files (but NOT the ROMs - those are downloaded separately)
datas += collect_data_files('gymnasium')
datas += collect_data_files('ale_py')

# Add configuration files and training scripts
datas += [
    ('conf/config.yaml', 'conf'),
    ('tools/retro_ml_desktop/training_presets.yaml', 'tools/retro_ml_desktop'),
    ('training/train.py', 'training'),
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

# CustomTkinter
hiddenimports += ['PIL._tkinter_finder']

# Database
hiddenimports += ['sqlite3']

# Video processing
hiddenimports += ['cv2', 'moviepy', 'imageio', 'imageio_ffmpeg']

# Other ML dependencies (but NOT torch - we'll download that separately)
hiddenimports += ['numpy', 'pandas', 'scipy']

# GPU detection modules
hiddenimports += ['tools.retro_ml_desktop.gpu_detector']

# Windows-specific modules for GPU detection
if sys.platform == 'win32':
    hiddenimports += ['winreg', 'wmi']

# Binaries to include
binaries = []

# NO CUDA DLLs - PyTorch will be downloaded on first run!
print("=" * 60)
print("BUILDING MINIMAL INSTALLER (WITHOUT PYTORCH)")
print("PyTorch will be downloaded on first run")
print("=" * 60)

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
        # EXCLUDE PYTORCH - we'll download it on first run!
        'torch',
        'torchvision',
        'torchaudio',
        'torch.cuda',
        'torch.backends.cudnn',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RetroMLTrainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Add your icon file here
)

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

# Print summary
print("\n" + "=" * 60)
print("BUILD COMPLETE - MINIMAL INSTALLER")
print("=" * 60)
print("This installer does NOT include PyTorch.")
print("PyTorch will be downloaded on first run (~2.5 GB for CUDA version).")
print("Expected installer size: ~100-150 MB")
print("=" * 60)

