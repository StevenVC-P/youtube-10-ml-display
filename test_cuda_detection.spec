# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for CUDA detection test
"""

import sys
import os
from pathlib import Path

# Find PyTorch installation to get CUDA DLLs
import torch
torch_lib_path = Path(torch.__file__).parent / 'lib'

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

a = Analysis(
    ['test_cuda_detection.py'],
    pathex=[],
    binaries=binaries,
    datas=[],
    hiddenimports=['torch', 'torch.cuda'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='test_cuda_detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console window for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='test_cuda_detection',
)

