"""
Build script for creating standalone executable with PyInstaller

This script:
1. Installs PyInstaller if needed
2. Runs PyInstaller with the spec file
3. Bundles FFmpeg binaries
4. Creates distribution folder
"""

import subprocess
import sys
import shutil
from pathlib import Path
import urllib.request
import zipfile
import os


def check_pyinstaller():
    """Check if PyInstaller is installed, install if not."""
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__} is installed")
        return True
    except ImportError:
        print("❌ PyInstaller not found")
        print("📥 Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✅ PyInstaller installed")
        return True


def download_ffmpeg():
    """Download FFmpeg binaries for Windows."""
    print("\n📥 Downloading FFmpeg...")
    
    ffmpeg_dir = Path("build_scripts/ffmpeg")
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
    if ffmpeg_exe.exists():
        print("✅ FFmpeg already downloaded")
        return ffmpeg_dir
    
    # FFmpeg download URL (essentials build)
    # Using a stable release from gyan.dev
    ffmpeg_url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    
    print(f"Downloading from: {ffmpeg_url}")
    print("This may take a few minutes...")
    
    zip_path = ffmpeg_dir / "ffmpeg.zip"
    
    try:
        urllib.request.urlretrieve(ffmpeg_url, zip_path)
        print("✅ Download complete")
        
        # Extract
        print("📦 Extracting FFmpeg...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        
        # Find the extracted ffmpeg.exe
        for root, dirs, files in os.walk(ffmpeg_dir):
            if 'ffmpeg.exe' in files:
                src = Path(root) / 'ffmpeg.exe'
                shutil.copy(src, ffmpeg_exe)
                print(f"✅ FFmpeg extracted to {ffmpeg_exe}")
                break
        
        # Cleanup
        zip_path.unlink()
        
        return ffmpeg_dir
    
    except Exception as e:
        print(f"❌ Error downloading FFmpeg: {e}")
        print("You can manually download FFmpeg from https://ffmpeg.org/download.html")
        print("and place ffmpeg.exe in build_scripts/ffmpeg/")
        return None


def build_executable():
    """Build the executable using PyInstaller."""
    print("\n🔨 Building executable with PyInstaller...")
    
    spec_file = Path("retro_ml_trainer.spec")
    
    if not spec_file.exists():
        print(f"❌ Spec file not found: {spec_file}")
        return False
    
    try:
        # Run PyInstaller
        subprocess.check_call([
            sys.executable, "-m", "PyInstaller",
            str(spec_file),
            "--clean",  # Clean cache
            "--noconfirm",  # Overwrite output directory
        ])
        
        print("✅ Executable built successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False


def bundle_ffmpeg(dist_dir: Path, ffmpeg_dir: Path):
    """Copy FFmpeg to distribution folder."""
    print("\n📦 Bundling FFmpeg...")
    
    ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
    if not ffmpeg_exe.exists():
        print("⚠️ FFmpeg not found, skipping...")
        return
    
    # Copy to dist folder
    dest = dist_dir / "RetroMLTrainer" / "ffmpeg.exe"
    shutil.copy(ffmpeg_exe, dest)
    print(f"✅ FFmpeg bundled to {dest}")


def create_distribution():
    """Create final distribution folder."""
    print("\n📦 Creating distribution package...")
    
    dist_dir = Path("dist")
    app_dir = dist_dir / "RetroMLTrainer"
    
    if not app_dir.exists():
        print(f"❌ Distribution folder not found: {app_dir}")
        return False
    
    # Create README for distribution
    readme_content = """# Retro ML Trainer

## Installation

1. Extract this folder to your desired location (e.g., C:\\RetroMLTrainer)
2. Run RetroMLTrainer.exe
3. Follow the setup wizard

## System Requirements

- Windows 10/11 (64-bit)
- 2GB RAM minimum (8GB+ recommended)
- 50GB+ free disk space (for training data and videos)
- Optional: NVIDIA GPU with CUDA support for faster training

## First Run

On first run, the setup wizard will guide you through:
- Choosing installation location
- Configuring storage paths
- Installing Atari game ROMs
- System capability detection

## Support

For issues or questions, please contact the developer.

## License

See LICENSE file for details.
"""
    
    readme_path = app_dir / "README.txt"
    readme_path.write_text(readme_content)
    print(f"✅ Created {readme_path}")
    
    # Calculate size
    total_size = sum(f.stat().st_size for f in app_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n✅ Distribution package created!")
    print(f"📁 Location: {app_dir}")
    print(f"📊 Size: {size_mb:.1f} MB")
    
    return True


def main():
    """Main build process."""
    print("=" * 60)
    print("🎮 Retro ML Trainer - Build Script")
    print("=" * 60)
    
    # Step 1: Check PyInstaller
    if not check_pyinstaller():
        return 1
    
    # Step 2: Download FFmpeg
    ffmpeg_dir = download_ffmpeg()
    
    # Step 3: Build executable
    if not build_executable():
        return 1
    
    # Step 4: Bundle FFmpeg
    if ffmpeg_dir:
        bundle_ffmpeg(Path("dist"), ffmpeg_dir)
    
    # Step 5: Create distribution
    if not create_distribution():
        return 1
    
    print("\n" + "=" * 60)
    print("✅ Build complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the executable in dist/RetroMLTrainer/")
    print("2. Run build_installer.bat to create Windows installer")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

