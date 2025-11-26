"""
Dependency installer for downloading PyTorch and other ML dependencies on first run.
"""
import os
import sys
import subprocess
import urllib.request
import json
import logging
from pathlib import Path
from typing import Callable, Optional

# Import robust GPU detector
from tools.retro_ml_desktop.gpu_detector import RobustGPUDetector

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class DependencyInstaller:
    """Handles downloading and installing ML dependencies on first run."""
    
    def __init__(self, install_dir: Optional[Path] = None):
        """
        Initialize the dependency installer.

        Args:
            install_dir: Directory to install dependencies to.
                        Defaults to AppData/Local/RetroMLTrainer/python_env
        """
        logger.info("Initializing DependencyInstaller...")

        if install_dir is None:
            # Install to AppData by default
            appdata = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~/.local')))
            self.install_dir = appdata / 'RetroMLTrainer' / 'python_env'
            logger.info(f"Using default install directory: {self.install_dir}")
        else:
            self.install_dir = Path(install_dir)
            logger.info(f"Using custom install directory: {self.install_dir}")

        self.install_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Install directory created/verified: {self.install_dir}")

        # Track installation status
        self.status_file = self.install_dir / 'installation_status.json'
        logger.info(f"Status file: {self.status_file}")
    
    def is_installed(self) -> bool:
        """Check if dependencies are already installed."""
        logger.info("Checking if PyTorch is already installed...")

        if not self.status_file.exists():
            logger.info(f"Status file does not exist: {self.status_file}")
            logger.info("PyTorch is NOT installed")
            return False

        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            is_installed = status.get('pytorch_installed', False)
            logger.info(f"Status file found. PyTorch installed: {is_installed}")
            return is_installed
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading status file: {e}")
            logger.info("Assuming PyTorch is NOT installed")
            return False
    
    def detect_gpu(self) -> dict:
        """
        Detect GPU capabilities using robust multi-method detection.

        Returns:
            Dictionary with GPU info: {'has_nvidia': bool, 'gpu_name': str, 'cuda_available': bool,
                                      'driver_version': str, 'detection_method': str}
        """
        logger.info("Detecting GPU using robust multi-method detector...")

        try:
            # Use the robust GPU detector
            detector = RobustGPUDetector()
            gpu_info_obj = detector.detect()

            # Convert to dictionary for backward compatibility
            gpu_info = {
                'has_nvidia': gpu_info_obj.has_nvidia,
                'gpu_name': gpu_info_obj.gpu_name,
                'cuda_available': gpu_info_obj.cuda_available,
                'driver_version': gpu_info_obj.driver_version,
                'cuda_version': gpu_info_obj.cuda_version,
                'detection_method': gpu_info_obj.detection_method,
                'gpu_names': gpu_info_obj.gpu_names,
            }

            # Log final result
            if gpu_info['has_nvidia']:
                logger.info(f"✅ NVIDIA GPU detected: {gpu_info['gpu_name']}")
                logger.info(f"   Detection method: {gpu_info['detection_method']}")
                if gpu_info['driver_version']:
                    logger.info(f"   Driver version: {gpu_info['driver_version']}")
            else:
                logger.info("❌ No NVIDIA GPU detected")

            return gpu_info

        except Exception as e:
            # If GPU detection completely fails (e.g., in restricted VM environments),
            # return a safe default indicating no GPU
            logger.warning(f"GPU detection failed with error: {e}")
            logger.info("Defaulting to CPU-only mode")
            return {
                'has_nvidia': False,
                'gpu_name': 'None',
                'cuda_available': False,
                'driver_version': None,
                'cuda_version': None,
                'detection_method': 'error_fallback',
                'gpu_names': [],
            }
    
    def install_pytorch_cuda(self, progress_callback: Optional[Callable[[str, int], None]] = None):
        """
        Install PyTorch with CUDA support.
        
        Args:
            progress_callback: Optional callback function(message: str, progress: int)
                             Called with status updates and progress percentage (0-100)
        """
        def update_progress(message: str, progress: int):
            if progress_callback:
                progress_callback(message, progress)
            else:
                print(f"[{progress}%] {message}")
        
        try:
            update_progress("Preparing to install PyTorch with CUDA support...", 0)
            
            # Get Python executable
            python_exe = sys.executable
            
            # Install PyTorch with CUDA 11.8 (compatible with most NVIDIA GPUs)
            update_progress("Downloading PyTorch with CUDA 11.8 (this may take 5-10 minutes)...", 10)
            
            # Use pip to install PyTorch
            install_cmd = [
                python_exe,
                '-m', 'pip',
                'install',
                'torch',
                'torchvision', 
                'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu118',
                '--no-cache-dir'  # Don't cache to save space
            ]
            
            # Run installation
            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor progress
            progress = 10
            for line in process.stdout:
                line = line.strip()
                if line:
                    # Update progress based on pip output
                    if 'Downloading' in line:
                        progress = min(progress + 2, 70)
                        update_progress(f"Downloading: {line[:60]}...", progress)
                    elif 'Installing' in line:
                        progress = min(progress + 5, 90)
                        update_progress(f"Installing: {line[:60]}...", progress)
            
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"PyTorch installation failed with code {process.returncode}")
            
            update_progress("PyTorch installation complete!", 95)
            
            # Mark as installed
            self._mark_installed('pytorch')
            
            update_progress("All dependencies installed successfully!", 100)
            
        except Exception as e:
            raise RuntimeError(f"Failed to install PyTorch: {str(e)}")
    
    def install_pytorch_cpu(self, progress_callback: Optional[Callable[[str, int], None]] = None):
        """
        Install PyTorch CPU-only version (fallback for non-NVIDIA systems).
        
        Args:
            progress_callback: Optional callback function(message: str, progress: int)
        """
        def update_progress(message: str, progress: int):
            if progress_callback:
                progress_callback(message, progress)
            else:
                print(f"[{progress}%] {message}")
        
        try:
            update_progress("Preparing to install PyTorch (CPU version)...", 0)
            
            python_exe = sys.executable
            
            update_progress("Downloading PyTorch CPU version...", 10)
            
            install_cmd = [
                python_exe,
                '-m', 'pip',
                'install',
                'torch',
                'torchvision',
                'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                '--no-cache-dir'
            ]
            
            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            progress = 10
            for line in process.stdout:
                line = line.strip()
                if line:
                    if 'Downloading' in line:
                        progress = min(progress + 2, 70)
                        update_progress(f"Downloading: {line[:60]}...", progress)
                    elif 'Installing' in line:
                        progress = min(progress + 5, 90)
                        update_progress(f"Installing: {line[:60]}...", progress)
            
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"PyTorch installation failed with code {process.returncode}")
            
            update_progress("PyTorch CPU installation complete!", 95)
            self._mark_installed('pytorch')
            update_progress("All dependencies installed successfully!", 100)
            
        except Exception as e:
            raise RuntimeError(f"Failed to install PyTorch: {str(e)}")
    
    def _mark_installed(self, component: str):
        """Mark a component as installed."""
        status = {}
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        status[f'{component}_installed'] = True
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def get_installation_size_estimate(self, include_cuda: bool = True) -> str:
        """
        Get estimated download size.
        
        Args:
            include_cuda: Whether to include CUDA version (vs CPU-only)
        
        Returns:
            Human-readable size estimate
        """
        if include_cuda:
            return "~2.5 GB"
        else:
            return "~800 MB"


if __name__ == '__main__':
    # Test the installer
    installer = DependencyInstaller()
    
    print("Detecting GPU...")
    gpu_info = installer.detect_gpu()
    print(f"GPU Info: {gpu_info}")
    
    if gpu_info['has_nvidia']:
        print(f"\nNVIDIA GPU detected: {gpu_info['gpu_name']}")
        print(f"Will install PyTorch with CUDA support")
        print(f"Estimated download: {installer.get_installation_size_estimate(True)}")
    else:
        print("\nNo NVIDIA GPU detected")
        print(f"Will install PyTorch CPU version")
        print(f"Estimated download: {installer.get_installation_size_estimate(False)}")

