#!/usr/bin/env python3
"""
CUDA Diagnostics and Error Handling System

Provides user-friendly error messages and actionable solutions for CUDA/GPU issues.
Includes comprehensive diagnostics and troubleshooting steps.
"""

import os
import sys
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from datetime import datetime

# Try to import torch, but don't fail if it's not installed yet
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


@dataclass
class CUDADiagnosticInfo:
    """Container for CUDA diagnostic information."""
    cuda_available: bool
    cuda_version: Optional[str]
    driver_version: Optional[str]
    device_count: int
    devices: List[Dict[str, Any]]
    memory_info: List[Dict[str, Any]]
    system_memory: Dict[str, Any]
    pytorch_version: str
    recommendations: List[str]
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class CUDADiagnostics:
    """
    Comprehensive CUDA diagnostics and error handling system.
    
    Provides user-friendly error messages, system diagnostics,
    and actionable troubleshooting steps.
    """
    
    def __init__(self):
        """Initialize CUDA diagnostics."""
        self.logger = logging.getLogger(__name__)
        
        # Common CUDA error patterns and their solutions
        self.error_patterns = {
            "CUDA error: unknown error": {
                "type": "CUDA_UNKNOWN_ERROR",
                "description": "Generic CUDA error - usually memory or driver related",
                "solutions": [
                    "Reduce batch size or number of environments",
                    "Update NVIDIA drivers",
                    "Restart the application",
                    "Check GPU memory usage"
                ]
            },
            "CUDA out of memory": {
                "type": "CUDA_OOM",
                "description": "GPU memory exhausted",
                "solutions": [
                    "Reduce batch size (current: 256 → try 128 or 64)",
                    "Reduce number of environments (current: 8 → try 4 or 2)",
                    "Close other GPU applications",
                    "Use CPU training instead"
                ]
            },
            "Memory allocation failure": {
                "type": "MEMORY_ALLOCATION_FAILURE",
                "description": "Failed to allocate GPU memory",
                "solutions": [
                    "Reduce training parameters",
                    "Clear GPU cache",
                    "Restart the application",
                    "Check available GPU memory"
                ]
            },
            "CUDA kernel errors might be asynchronously reported": {
                "type": "CUDA_KERNEL_ERROR",
                "description": "CUDA kernel execution failed",
                "solutions": [
                    "Set CUDA_LAUNCH_BLOCKING=1 for better error reporting",
                    "Update PyTorch and CUDA drivers",
                    "Reduce computational load",
                    "Check GPU compatibility"
                ]
            },
            "device-side assertion": {
                "type": "CUDA_DEVICE_ASSERTION",
                "description": "GPU device assertion failed",
                "solutions": [
                    "Enable TORCH_USE_CUDA_DSA for detailed debugging",
                    "Check input data validity",
                    "Update PyTorch version",
                    "Use CPU training for debugging"
                ]
            }
        }
    
    def diagnose_system(self) -> CUDADiagnosticInfo:
        """
        Perform comprehensive CUDA system diagnostics.

        Returns:
            CUDADiagnosticInfo with complete system information
        """
        # Check if torch is available
        if not TORCH_AVAILABLE or torch is None:
            # PyTorch not installed - return minimal diagnostics
            system_memory = {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "percent_used": psutil.virtual_memory().percent
            }
            return CUDADiagnosticInfo(
                cuda_available=False,
                cuda_version=None,
                driver_version=None,
                device_count=0,
                devices=[],
                memory_info=[],
                system_memory=system_memory,
                pytorch_version="Not installed",
                recommendations=["PyTorch is not installed. Please run the setup wizard to install dependencies."]
            )

        # Basic CUDA availability
        cuda_available = torch.cuda.is_available()
        
        # CUDA version info
        cuda_version = torch.version.cuda if cuda_available else None
        
        # Driver version
        driver_version = self._get_driver_version()
        
        # Device information
        device_count = torch.cuda.device_count() if cuda_available else 0
        devices = []
        memory_info = []
        
        if cuda_available and device_count > 0:
            for i in range(device_count):
                device_props = torch.cuda.get_device_properties(i)
                devices.append({
                    "id": i,
                    "name": device_props.name,
                    "compute_capability": f"{device_props.major}.{device_props.minor}",
                    "total_memory_gb": device_props.total_memory / (1024**3),
                    "multiprocessor_count": device_props.multi_processor_count
                })
                
                # Memory info
                if torch.cuda.is_available():
                    try:
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_free = (device_props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
                        
                        memory_info.append({
                            "device_id": i,
                            "total_gb": device_props.total_memory / (1024**3),
                            "reserved_gb": memory_reserved,
                            "allocated_gb": memory_allocated,
                            "free_gb": memory_free,
                            "utilization_pct": (memory_reserved / (device_props.total_memory / (1024**3))) * 100
                        })
                    except Exception as e:
                        memory_info.append({
                            "device_id": i,
                            "error": str(e)
                        })
        
        # System memory
        system_memory = {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "used_gb": psutil.virtual_memory().used / (1024**3),
            "percent_used": psutil.virtual_memory().percent
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cuda_available, device_count, devices, memory_info, system_memory
        )
        
        return CUDADiagnosticInfo(
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            driver_version=driver_version,
            device_count=device_count,
            devices=devices,
            memory_info=memory_info,
            system_memory=system_memory,
            pytorch_version=torch.__version__,
            recommendations=recommendations
        )
    
    def analyze_error(self, error_message: str, traceback_str: Optional[str] = None) -> CUDADiagnosticInfo:
        """
        Analyze a CUDA error and provide specific diagnostics.
        
        Args:
            error_message: The error message to analyze
            traceback_str: Optional full traceback
            
        Returns:
            CUDADiagnosticInfo with error-specific information
        """
        # Get base system diagnostics
        diagnostics = self.diagnose_system()
        
        # Analyze error pattern
        error_type = None
        specific_solutions = []
        
        for pattern, info in self.error_patterns.items():
            if pattern.lower() in error_message.lower():
                error_type = info["type"]
                specific_solutions = info["solutions"]
                break
        
        # Add error-specific information
        diagnostics.error_type = error_type
        diagnostics.error_message = error_message
        
        # Combine general recommendations with error-specific solutions
        if specific_solutions:
            diagnostics.recommendations = specific_solutions + diagnostics.recommendations
        
        return diagnostics
    
    def _get_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass
        return None
    
    def _generate_recommendations(
        self, 
        cuda_available: bool, 
        device_count: int, 
        devices: List[Dict], 
        memory_info: List[Dict], 
        system_memory: Dict
    ) -> List[str]:
        """Generate system-specific recommendations."""
        recommendations = []
        
        if not cuda_available:
            recommendations.extend([
                "CUDA is not available - install NVIDIA drivers and CUDA toolkit",
                "Consider using CPU training as fallback",
                "Check if GPU is properly connected and recognized"
            ])
            return recommendations
        
        if device_count == 0:
            recommendations.extend([
                "No CUDA devices detected - check GPU installation",
                "Verify NVIDIA drivers are properly installed",
                "Try restarting the system"
            ])
            return recommendations
        
        # Memory-based recommendations
        for mem_info in memory_info:
            if "utilization_pct" in mem_info:
                if mem_info["utilization_pct"] > 80:
                    recommendations.append(f"GPU {mem_info['device_id']} memory usage is high ({mem_info['utilization_pct']:.1f}%) - consider reducing batch size")
                
                if mem_info["free_gb"] < 1.0:
                    recommendations.append(f"GPU {mem_info['device_id']} has low free memory ({mem_info['free_gb']:.1f}GB) - close other GPU applications")
        
        # System memory recommendations
        if system_memory["percent_used"] > 85:
            recommendations.append(f"System RAM usage is high ({system_memory['percent_used']:.1f}%) - close other applications")
        
        # Device-specific recommendations
        for device in devices:
            if device["total_memory_gb"] < 4:
                recommendations.append(f"GPU {device['id']} ({device['name']}) has limited memory ({device['total_memory_gb']:.1f}GB) - use smaller batch sizes")
        
        return recommendations
    
    def format_diagnostic_report(self, diagnostics: CUDADiagnosticInfo) -> str:
        """
        Format diagnostics into a user-friendly report.
        
        Args:
            diagnostics: CUDADiagnosticInfo to format
            
        Returns:
            Formatted diagnostic report string
        """
        report = []
        
        # Header
        report.append("🔍 CUDA DIAGNOSTICS REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Error information (if present)
        if diagnostics.error_type:
            report.append("❌ ERROR ANALYSIS")
            report.append(f"Error Type: {diagnostics.error_type}")
            report.append(f"Error Message: {diagnostics.error_message}")
            report.append("")
        
        # System information
        report.append("🖥️ SYSTEM INFORMATION")
        report.append(f"PyTorch Version: {diagnostics.pytorch_version}")
        report.append(f"CUDA Available: {'✅ Yes' if diagnostics.cuda_available else '❌ No'}")
        
        if diagnostics.cuda_version:
            report.append(f"CUDA Version: {diagnostics.cuda_version}")
        
        if diagnostics.driver_version:
            report.append(f"Driver Version: {diagnostics.driver_version}")
        
        report.append(f"GPU Devices: {diagnostics.device_count}")
        report.append("")
        
        # GPU information
        if diagnostics.devices:
            report.append("🎮 GPU DEVICES")
            for device in diagnostics.devices:
                report.append(f"GPU {device['id']}: {device['name']}")
                report.append(f"  Memory: {device['total_memory_gb']:.1f}GB")
                report.append(f"  Compute: {device['compute_capability']}")
                report.append(f"  Cores: {device['multiprocessor_count']}")
                report.append("")
        
        # Memory information
        if diagnostics.memory_info:
            report.append("💾 MEMORY USAGE")
            for mem in diagnostics.memory_info:
                if "error" not in mem:
                    report.append(f"GPU {mem['device_id']}:")
                    report.append(f"  Total: {mem['total_gb']:.1f}GB")
                    report.append(f"  Used: {mem['allocated_gb']:.1f}GB ({mem['utilization_pct']:.1f}%)")
                    report.append(f"  Free: {mem['free_gb']:.1f}GB")
                    report.append("")
        
        # System memory
        report.append("🖥️ SYSTEM MEMORY")
        report.append(f"Total: {diagnostics.system_memory['total_gb']:.1f}GB")
        report.append(f"Used: {diagnostics.system_memory['used_gb']:.1f}GB ({diagnostics.system_memory['percent_used']:.1f}%)")
        report.append(f"Available: {diagnostics.system_memory['available_gb']:.1f}GB")
        report.append("")
        
        # Recommendations
        if diagnostics.recommendations:
            report.append("💡 RECOMMENDATIONS")
            for i, rec in enumerate(diagnostics.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)
    
    def get_training_config_suggestions(self, diagnostics: CUDADiagnosticInfo) -> Dict[str, Any]:
        """
        Get suggested training configuration based on system capabilities.
        
        Args:
            diagnostics: System diagnostic information
            
        Returns:
            Dictionary with suggested configuration parameters
        """
        suggestions = {
            "use_gpu": diagnostics.cuda_available and diagnostics.device_count > 0,
            "device": "cuda" if diagnostics.cuda_available else "cpu"
        }
        
        if not diagnostics.cuda_available or diagnostics.device_count == 0:
            # CPU fallback configuration
            suggestions.update({
                "vec_envs": 2,  # Reduced for CPU
                "batch_size": 64,  # Smaller batch size
                "n_steps": 64,  # Reduced steps
                "learning_rate": 3e-4
            })
            return suggestions
        
        # GPU-based suggestions
        total_gpu_memory = sum(device["total_memory_gb"] for device in diagnostics.devices)
        
        if total_gpu_memory >= 8:
            # High-end GPU configuration
            suggestions.update({
                "vec_envs": 8,
                "batch_size": 256,
                "n_steps": 128
            })
        elif total_gpu_memory >= 4:
            # Mid-range GPU configuration
            suggestions.update({
                "vec_envs": 4,
                "batch_size": 128,
                "n_steps": 128
            })
        else:
            # Low-end GPU configuration
            suggestions.update({
                "vec_envs": 2,
                "batch_size": 64,
                "n_steps": 64
            })
        
        return suggestions


def create_user_friendly_error_message(error: Exception, traceback_str: Optional[str] = None) -> str:
    """
    Create a user-friendly error message with diagnostics and solutions.
    
    Args:
        error: The exception that occurred
        traceback_str: Optional full traceback string
        
    Returns:
        Formatted user-friendly error message
    """
    diagnostics_system = CUDADiagnostics()
    error_message = str(error)
    
    # Analyze the error
    diagnostics = diagnostics_system.analyze_error(error_message, traceback_str)
    
    # Create user-friendly message
    message_parts = []
    
    # Error summary
    message_parts.append("🚨 TRAINING ERROR DETECTED")
    message_parts.append("=" * 40)
    
    if diagnostics.error_type:
        error_descriptions = {
            "CUDA_OOM": "GPU ran out of memory",
            "CUDA_UNKNOWN_ERROR": "GPU encountered an unknown error",
            "MEMORY_ALLOCATION_FAILURE": "Failed to allocate GPU memory",
            "CUDA_KERNEL_ERROR": "GPU computation failed",
            "CUDA_DEVICE_ASSERTION": "GPU device assertion failed"
        }
        
        description = error_descriptions.get(diagnostics.error_type, "Unknown GPU error")
        message_parts.append(f"Problem: {description}")
    else:
        message_parts.append(f"Problem: {error_message}")
    
    message_parts.append("")
    
    # Quick solutions
    if diagnostics.recommendations:
        message_parts.append("🔧 IMMEDIATE SOLUTIONS TO TRY:")
        for i, solution in enumerate(diagnostics.recommendations[:5], 1):  # Top 5 solutions
            message_parts.append(f"  {i}. {solution}")
        message_parts.append("")
    
    # System info summary
    if diagnostics.cuda_available:
        gpu_info = f"{diagnostics.device_count} GPU(s) available"
        if diagnostics.memory_info:
            total_memory = sum(mem.get("total_gb", 0) for mem in diagnostics.memory_info)
            gpu_info += f" ({total_memory:.1f}GB total)"
        message_parts.append(f"System: {gpu_info}")
    else:
        message_parts.append("System: No CUDA GPUs available")
    
    message_parts.append("")
    message_parts.append("💡 For detailed diagnostics, check the ML Dashboard or run system diagnostics.")
    
    return "\n".join(message_parts)


# Export main functions
__all__ = [
    "CUDADiagnostics",
    "CUDADiagnosticInfo", 
    "create_user_friendly_error_message"
]
