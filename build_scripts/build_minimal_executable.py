"""
Build script for creating minimal Retro ML Trainer executable (without PyTorch).

This creates a small installer that downloads PyTorch on first run.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import urllib.request
import zipfile
import io

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

def check_pyinstaller():
    """Check if PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"[OK] PyInstaller {PyInstaller.__version__} found")
        return True
    except ImportError:
        print("[ERROR] PyInstaller not found")
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])
        return True

def download_ffmpeg():
    """Download FFmpeg if not already present."""
    ffmpeg_dir = Path('build_scripts/ffmpeg')
    ffmpeg_exe = ffmpeg_dir / 'bin' / 'ffmpeg.exe'
    
    if ffmpeg_exe.exists():
        print(f"[OK] FFmpeg already downloaded at {ffmpeg_exe}")
        return ffmpeg_dir

    print("[INFO] Downloading FFmpeg...")
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    # Create temp directory
    temp_dir = Path('build_scripts/temp')
    temp_dir.mkdir(exist_ok=True)
    
    # Download
    zip_path = temp_dir / 'ffmpeg.zip'
    print(f"   Downloading from {ffmpeg_url}")
    urllib.request.urlretrieve(ffmpeg_url, zip_path)
    
    # Extract
    print("   Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the extracted folder (it has a version number in the name)
    extracted_folders = [f for f in temp_dir.iterdir() if f.is_dir() and f.name.startswith('ffmpeg-')]
    if not extracted_folders:
        raise RuntimeError("Could not find extracted FFmpeg folder")
    
    extracted_folder = extracted_folders[0]
    
    # Move to ffmpeg_dir
    if ffmpeg_dir.exists():
        shutil.rmtree(ffmpeg_dir)
    shutil.move(str(extracted_folder), str(ffmpeg_dir))
    
    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"[OK] FFmpeg downloaded to {ffmpeg_dir}")
    return ffmpeg_dir

def build_executable():
    """Build the executable using PyInstaller."""
    print("\n" + "=" * 60)
    print("BUILDING MINIMAL RETRO ML TRAINER EXECUTABLE")
    print("(WITHOUT PyTorch - downloads on first run)")
    print("=" * 60 + "\n")
    
    # Check PyInstaller
    check_pyinstaller()
    
    # Download FFmpeg
    ffmpeg_dir = download_ffmpeg()
    
    # Run PyInstaller
    print("\n[INFO] Running PyInstaller...")
    spec_file = 'retro_ml_trainer_minimal.spec'
    
    cmd = [
        sys.executable,
        '-m', 'PyInstaller',
        spec_file,
        '--clean',  # Clean cache
        '--noconfirm',  # Overwrite without asking
    ]
    
    print(f"   Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("\n[ERROR] Build failed!")
        sys.exit(1)

    # Copy FFmpeg to dist folder
    dist_dir = Path('dist/RetroMLTrainer')
    ffmpeg_dest = dist_dir / 'ffmpeg'

    if ffmpeg_dest.exists():
        shutil.rmtree(ffmpeg_dest)

    print(f"\n[INFO] Copying FFmpeg to {ffmpeg_dest}")
    shutil.copytree(ffmpeg_dir, ffmpeg_dest)
    
    # Create README in dist folder
    readme_content = """# Retro ML Trainer - Minimal Installer

This is a minimal installer that downloads PyTorch on first run.

## First Run Setup

When you first launch the application, it will:
1. Detect your GPU (NVIDIA recommended)
2. Download PyTorch with CUDA support (~2.5 GB) or CPU version (~800 MB)
3. Install ML dependencies
4. Download Atari ROMs (with license acceptance)

This process takes 10-15 minutes depending on your internet speed.

## System Requirements

- Windows 10 or later (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- 10 GB free disk space
- NVIDIA GPU recommended (for GPU acceleration)
- Internet connection (for first-time setup)

## GPU Support

This application is optimized for NVIDIA GPUs with CUDA support.
If you have an NVIDIA GPU, make sure you have the latest drivers installed.

AMD and Intel GPUs are not currently supported for GPU acceleration,
but the application will work in CPU mode (slower training).

## Support

For issues or questions, please visit:
https://github.com/StevenVC-P/youtube-10-ml-display

## License

See LICENSE file for details.
"""
    
    readme_path = dist_dir / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("[SUCCESS] BUILD COMPLETE!")
    print("=" * 60)
    print(f"[INFO] Output directory: {dist_dir}")
    print(f"[INFO] Total size: {size_mb:.1f} MB")
    print("\n[INFO] Next steps:")
    print("   1. Test the executable: dist/RetroMLTrainer/RetroMLTrainer.exe")
    print("   2. Compile installer with Inno Setup (build_scripts/installer.iss)")
    print("   3. Test the installer on a clean Windows machine")
    print("\n[NOTE] This installer does NOT include PyTorch.")
    print("   PyTorch will be downloaded on first run (~2.5 GB for CUDA).")
    print("=" * 60)

if __name__ == '__main__':
    try:
        build_executable()
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

