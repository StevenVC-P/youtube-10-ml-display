"""
Robust GPU Detection System with Multiple Fallback Methods

This module provides comprehensive GPU detection that works in various environments,
including PyInstaller frozen executables and restricted environments.

Detection Methods (in order of priority):
1. nvidia-smi command-line tool
2. Windows Registry check for NVIDIA drivers
3. WMI (Windows Management Instrumentation) query
4. PyTorch CUDA detection (if torch is available)
5. CUDA DLL file check
6. Environment variables inspection
"""

import os
import sys
import subprocess
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GPUInfo:
    """Container for GPU detection results."""
    has_nvidia: bool = False
    gpu_name: str = "None"
    gpu_names: List[str] = None  # Multiple GPUs
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cuda_available: bool = False
    detection_method: str = "none"
    detection_methods_tried: List[str] = None
    is_frozen: bool = False  # Running in PyInstaller

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.gpu_names is None:
            self.gpu_names = []
        if self.detection_methods_tried is None:
            self.detection_methods_tried = []

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class RobustGPUDetector:
    """
    Multi-method GPU detection system with extensive fallbacks.

    Designed to work in various environments including PyInstaller frozen executables.
    """

    def __init__(self):
        """Initialize the GPU detector."""
        self.is_frozen = getattr(sys, 'frozen', False)
        logger.info(f"GPU Detector initialized (Frozen: {self.is_frozen})")

    def detect(self) -> GPUInfo:
        """
        Detect GPU using all available methods.

        Returns:
            GPUInfo object with detection results
        """
        logger.info("=" * 60)
        logger.info("STARTING ROBUST GPU DETECTION")
        logger.info("=" * 60)

        gpu_info = GPUInfo(is_frozen=self.is_frozen)

        # Try detection methods in order of reliability
        detection_methods = [
            ("nvidia-smi", self._detect_via_nvidia_smi),
            ("registry", self._detect_via_registry),
            ("wmi", self._detect_via_wmi),
            ("pytorch", self._detect_via_pytorch),
            ("cuda_dll", self._detect_via_cuda_dll),
            ("environment", self._detect_via_environment),
        ]

        for method_name, method_func in detection_methods:
            gpu_info.detection_methods_tried.append(method_name)
            logger.info(f"\n--- Trying detection method: {method_name} ---")

            try:
                result = method_func()
                if result and result.get('has_nvidia', False):
                    # Found NVIDIA GPU!
                    logger.info(f"✅ GPU detected via {method_name}!")
                    gpu_info.has_nvidia = True
                    gpu_info.detection_method = method_name

                    # Update GPU info with detected values
                    if 'gpu_name' in result:
                        gpu_info.gpu_name = result['gpu_name']
                    if 'gpu_names' in result:
                        gpu_info.gpu_names = result['gpu_names']
                    if 'driver_version' in result:
                        gpu_info.driver_version = result['driver_version']
                    if 'cuda_version' in result:
                        gpu_info.cuda_version = result['cuda_version']
                    if 'cuda_available' in result:
                        gpu_info.cuda_available = result['cuda_available']

                    # If we found a GPU, try to enhance with additional info from other methods
                    self._enhance_gpu_info(gpu_info, detection_methods)
                    break
                else:
                    logger.info(f"❌ No GPU detected via {method_name}")
            except KeyboardInterrupt:
                raise  # Allow user to interrupt
            except Exception as e:
                logger.warning(f"❌ Error in {method_name}: {e}")
                # Continue to next method - don't let any single method crash the detector

        # Final result
        logger.info("\n" + "=" * 60)
        if gpu_info.has_nvidia:
            logger.info("✅ NVIDIA GPU DETECTED!")
            logger.info(f"   GPU: {gpu_info.gpu_name}")
            logger.info(f"   Driver: {gpu_info.driver_version or 'Unknown'}")
            logger.info(f"   CUDA: {gpu_info.cuda_version or 'Unknown'}")
            logger.info(f"   Detection method: {gpu_info.detection_method}")
        else:
            logger.info("❌ NO NVIDIA GPU DETECTED")
            logger.info(f"   Methods tried: {', '.join(gpu_info.detection_methods_tried)}")
        logger.info("=" * 60)

        return gpu_info

    def _enhance_gpu_info(self, gpu_info: GPUInfo, all_methods: List[Tuple]) -> None:
        """Try to enhance GPU info with additional details from other methods."""
        logger.info("\n--- Enhancing GPU info with additional methods ---")

        for method_name, method_func in all_methods:
            if method_name == gpu_info.detection_method:
                continue  # Skip the method that already succeeded

            try:
                result = method_func()
                if result:
                    # Add missing information
                    if not gpu_info.driver_version and 'driver_version' in result:
                        gpu_info.driver_version = result['driver_version']
                        logger.info(f"   Added driver version from {method_name}")

                    if not gpu_info.cuda_version and 'cuda_version' in result:
                        gpu_info.cuda_version = result['cuda_version']
                        logger.info(f"   Added CUDA version from {method_name}")
            except Exception as e:
                logger.debug(f"Could not enhance with {method_name}: {e}")

    def _detect_via_nvidia_smi(self) -> Optional[dict]:
        """
        Detect GPU using nvidia-smi command.

        This is the most reliable method when nvidia-smi is available.
        """
        try:
            # Try to find nvidia-smi in common locations
            nvidia_smi_paths = [
                'nvidia-smi',  # In PATH
                r'C:\Windows\System32\nvidia-smi.exe',
                r'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe',
            ]

            # Add CUDA toolkit paths if environment variable exists
            if 'CUDA_PATH' in os.environ:
                cuda_path = Path(os.environ['CUDA_PATH'])
                nvidia_smi_paths.append(str(cuda_path / 'bin' / 'nvidia-smi.exe'))

            nvidia_smi_exe = None
            for path in nvidia_smi_paths:
                if Path(path).exists() if '\\' in path else True:
                    try:
                        # Test if this path works
                        result = subprocess.run(
                            [path, '--version'],
                            capture_output=True,
                            text=True,
                            timeout=2,
                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                        )
                        if result.returncode == 0:
                            nvidia_smi_exe = path
                            logger.info(f"Found nvidia-smi at: {path}")
                            break
                    except:
                        continue

            if not nvidia_smi_exe:
                logger.info("nvidia-smi not found in any common location")
                return None

            # Get GPU names
            result = subprocess.run(
                [nvidia_smi_exe, '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            if result.returncode != 0 or not result.stdout.strip():
                logger.info("nvidia-smi query returned no GPU information")
                return None

            gpu_names = [name.strip() for name in result.stdout.strip().split('\n') if name.strip()]

            # Get driver version
            driver_result = subprocess.run(
                [nvidia_smi_exe, '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            driver_version = None
            if driver_result.returncode == 0 and driver_result.stdout.strip():
                driver_version = driver_result.stdout.strip().split('\n')[0]

            # Get CUDA version
            cuda_result = subprocess.run(
                [nvidia_smi_exe, '--query-gpu=cuda_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            cuda_version = None
            if cuda_result.returncode == 0 and cuda_result.stdout.strip():
                cuda_version = cuda_result.stdout.strip().split('\n')[0]

            return {
                'has_nvidia': True,
                'gpu_name': gpu_names[0],
                'gpu_names': gpu_names,
                'driver_version': driver_version,
                'cuda_version': cuda_version,
                'cuda_available': True
            }

        except FileNotFoundError:
            logger.info("nvidia-smi not found")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi command timed out")
            return None
        except Exception as e:
            logger.warning(f"nvidia-smi detection failed: {e}")
            return None

    def _detect_via_registry(self) -> Optional[dict]:
        """
        Detect GPU using Windows Registry.

        Checks for NVIDIA driver installation in registry.
        Windows only.
        """
        if sys.platform != 'win32':
            logger.info("Registry detection only available on Windows")
            return None

        try:
            import winreg

            # Check multiple registry locations
            registry_paths = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\Global"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Services\nvlddmkm"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\NVIDIA Corporation\Global"),
            ]

            driver_version = None
            gpu_found = False

            for hkey, path in registry_paths:
                try:
                    with winreg.OpenKey(hkey, path) as key:
                        logger.info(f"Found NVIDIA registry key: {path}")
                        gpu_found = True

                        # Try to get driver version
                        try:
                            driver_version = winreg.QueryValueEx(key, "DriverVersion")[0]
                            logger.info(f"Found driver version in registry: {driver_version}")
                        except:
                            pass

                        break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    logger.debug(f"Error reading registry path {path}: {e}")

            if gpu_found:
                # Try to get GPU name from Display adapter key
                gpu_name = "NVIDIA GPU (model unknown)"
                try:
                    display_key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, display_key_path) as display_key:
                        # Enumerate subkeys
                        for i in range(100):  # Check first 100 subkeys
                            try:
                                subkey_name = winreg.EnumKey(display_key, i)
                                with winreg.OpenKey(display_key, subkey_name) as subkey:
                                    try:
                                        desc = winreg.QueryValueEx(subkey, "DriverDesc")[0]
                                        if "NVIDIA" in desc.upper():
                                            gpu_name = desc
                                            logger.info(f"Found GPU name in registry: {gpu_name}")
                                            break
                                    except:
                                        pass
                            except OSError:
                                break
                except Exception as e:
                    logger.debug(f"Could not get GPU name from registry: {e}")

                return {
                    'has_nvidia': True,
                    'gpu_name': gpu_name,
                    'gpu_names': [gpu_name],
                    'driver_version': driver_version,
                    'cuda_available': False  # Can't determine CUDA from registry alone
                }

            logger.info("No NVIDIA entries found in registry")
            return None

        except ImportError:
            logger.info("winreg module not available")
            return None
        except Exception as e:
            logger.warning(f"Registry detection failed: {e}")
            return None

    def _detect_via_wmi(self) -> Optional[dict]:
        """
        Detect GPU using Windows Management Instrumentation (WMI).

        Windows only.
        """
        if sys.platform != 'win32':
            logger.info("WMI detection only available on Windows")
            return None

        try:
            import wmi

            # WMI connection can fail in VMs or restricted environments
            try:
                c = wmi.WMI()
            except Exception as wmi_error:
                logger.info(f"WMI connection failed (common in VMs): {wmi_error}")
                return None

            gpu_names = []

            # Query video controllers
            try:
                for gpu in c.Win32_VideoController():
                    if gpu and hasattr(gpu, 'Name') and gpu.Name and "NVIDIA" in gpu.Name.upper():
                        logger.info(f"Found NVIDIA GPU via WMI: {gpu.Name}")
                        gpu_names.append(gpu.Name)
            except Exception as query_error:
                logger.info(f"WMI query failed: {query_error}")
                return None

            if gpu_names:
                # Try to get driver version
                driver_version = None
                try:
                    nvidia_gpu = next(g for g in c.Win32_VideoController() if g and hasattr(g, 'Name') and g.Name and "NVIDIA" in g.Name.upper())
                    if hasattr(nvidia_gpu, 'DriverVersion'):
                        driver_version = nvidia_gpu.DriverVersion
                        logger.info(f"Driver version from WMI: {driver_version}")
                except Exception:
                    pass

                return {
                    'has_nvidia': True,
                    'gpu_name': gpu_names[0],
                    'gpu_names': gpu_names,
                    'driver_version': driver_version,
                    'cuda_available': False
                }

            logger.info("No NVIDIA GPUs found via WMI")
            return None

        except ImportError:
            logger.info("wmi module not available (install with: pip install wmi)")
            return None
        except Exception as e:
            logger.warning(f"WMI detection failed: {e}")
            return None

    def _detect_via_pytorch(self) -> Optional[dict]:
        """
        Detect GPU using PyTorch CUDA detection.

        Only works if PyTorch is already installed.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("PyTorch reports CUDA not available")
                return None

            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.info("PyTorch reports 0 CUDA devices")
                return None

            gpu_names = []
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_names.append(gpu_name)
                logger.info(f"PyTorch detected GPU {i}: {gpu_name}")

            # Get CUDA version
            cuda_version = torch.version.cuda

            return {
                'has_nvidia': True,
                'gpu_name': gpu_names[0],
                'gpu_names': gpu_names,
                'cuda_version': cuda_version,
                'cuda_available': True
            }

        except ImportError:
            logger.info("PyTorch not installed yet")
            return None
        except Exception as e:
            logger.warning(f"PyTorch detection failed: {e}")
            return None

    def _detect_via_cuda_dll(self) -> Optional[dict]:
        """
        Detect CUDA by checking for CUDA DLL files.

        Checks common CUDA installation locations.
        """
        try:
            # Common CUDA DLL locations
            cuda_dll_names = ['cudart64_*.dll', 'nvcuda.dll', 'nvrtc64_*.dll']
            cuda_paths = [
                r'C:\Windows\System32',
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
                r'C:\Program Files\NVIDIA Corporation\NVSMI',
            ]

            # Add CUDA_PATH from environment
            if 'CUDA_PATH' in os.environ:
                cuda_paths.insert(0, os.environ['CUDA_PATH'])

            cuda_found = False
            cuda_version = None

            for base_path in cuda_paths:
                if not Path(base_path).exists():
                    continue

                # Search for CUDA DLLs
                for dll_pattern in cuda_dll_names:
                    import glob
                    matches = glob.glob(str(Path(base_path) / '**' / dll_pattern), recursive=True)
                    if matches:
                        logger.info(f"Found CUDA DLL: {matches[0]}")
                        cuda_found = True

                        # Try to extract version from path
                        path_str = str(matches[0])
                        if 'v' in path_str:
                            try:
                                version_part = [p for p in path_str.split('\\') if p.startswith('v')]
                                if version_part:
                                    cuda_version = version_part[0].replace('v', '')
                            except:
                                pass
                        break

                if cuda_found:
                    break

            if cuda_found:
                return {
                    'has_nvidia': True,
                    'gpu_name': 'NVIDIA GPU (detected via CUDA DLLs)',
                    'gpu_names': ['NVIDIA GPU (detected via CUDA DLLs)'],
                    'cuda_version': cuda_version,
                    'cuda_available': True
                }

            logger.info("No CUDA DLLs found")
            return None

        except Exception as e:
            logger.warning(f"CUDA DLL detection failed: {e}")
            return None

    def _detect_via_environment(self) -> Optional[dict]:
        """
        Detect GPU using environment variables.

        Checks for CUDA-related environment variables.
        """
        try:
            cuda_env_vars = [
                'CUDA_PATH',
                'CUDA_HOME',
                'CUDA_PATH_V11_8',
                'CUDA_PATH_V12_0',
                'NVCUDASAMPLES_ROOT',
            ]

            found_vars = []
            for var in cuda_env_vars:
                if var in os.environ:
                    found_vars.append(f"{var}={os.environ[var]}")
                    logger.info(f"Found CUDA environment variable: {var}={os.environ[var]}")

            if found_vars:
                # Extract version if possible
                cuda_version = None
                if 'CUDA_PATH' in os.environ:
                    cuda_path = os.environ['CUDA_PATH']
                    # Try to extract version from path (e.g., "v11.8")
                    import re
                    version_match = re.search(r'v?(\d+\.\d+)', cuda_path)
                    if version_match:
                        cuda_version = version_match.group(1)

                return {
                    'has_nvidia': True,
                    'gpu_name': 'NVIDIA GPU (detected via environment variables)',
                    'gpu_names': ['NVIDIA GPU (detected via environment variables)'],
                    'cuda_version': cuda_version,
                    'cuda_available': True
                }

            logger.info("No CUDA environment variables found")
            return None

        except Exception as e:
            logger.warning(f"Environment variable detection failed: {e}")
            return None


# Convenience functions for quick detection
def detect_gpu() -> GPUInfo:
    """
    Quick GPU detection using all available methods.

    Returns:
        GPUInfo object with detection results
    """
    detector = RobustGPUDetector()
    return detector.detect()


def has_nvidia_gpu() -> bool:
    """
    Quick check if NVIDIA GPU is present.

    Returns:
        True if NVIDIA GPU detected, False otherwise
    """
    gpu_info = detect_gpu()
    return gpu_info.has_nvidia


def get_gpu_name() -> str:
    """
    Get the name of the first detected GPU.

    Returns:
        GPU name or "None" if no GPU detected
    """
    gpu_info = detect_gpu()
    return gpu_info.gpu_name


# Main test function
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ROBUST GPU DETECTOR - TEST MODE")
    print("=" * 70 + "\n")

    # Run detection
    gpu_info = detect_gpu()

    # Print results in a nice format
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)
    print(json.dumps(gpu_info.to_dict(), indent=2))
    print("=" * 70)

    # Simple usage examples
    print("\nSimple API Examples:")
    print(f"  has_nvidia_gpu() = {has_nvidia_gpu()}")
    print(f"  get_gpu_name() = {get_gpu_name()}")
