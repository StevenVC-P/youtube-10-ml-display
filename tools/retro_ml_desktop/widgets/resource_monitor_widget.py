"""
Resource Usage Monitor Widget - Real-time system resource monitoring.

Displays:
- GPU utilization and VRAM usage
- CPU usage
- System RAM usage

Updates every 2 seconds via polling (not event-driven).
"""

import customtkinter as ctk
import psutil
from typing import Optional, Dict, Any

try:
    from tools.retro_ml_desktop.gpu_detector import RobustGPUDetector
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class ResourceMonitorWidget(ctk.CTkFrame):
    """Widget displaying real-time system resource usage."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.gpu_detector = RobustGPUDetector() if GPU_AVAILABLE else None
        self._update_interval = 2000  # 2 seconds
        self._update_job = None

        self._init_ui()
        self._start_monitoring()

    def _init_ui(self):
        """Initialize the user interface."""
        # Header
        header = ctk.CTkLabel(
            self,
            text="Resource Usage",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        header.pack(pady=(10, 5), padx=10, anchor="w")

        # GPU section
        if GPU_AVAILABLE and self.gpu_detector:
            gpu_frame = ctk.CTkFrame(self, fg_color="transparent")
            gpu_frame.pack(fill="x", padx=10, pady=5)

            gpu_label = ctk.CTkLabel(
                gpu_frame,
                text="GPU",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            gpu_label.pack(anchor="w", pady=(0, 2))

            # GPU utilization bar
            self.gpu_util_label = ctk.CTkLabel(
                gpu_frame,
                text="Utilization: --",
                font=ctk.CTkFont(size=11)
            )
            self.gpu_util_label.pack(anchor="w")

            self.gpu_util_bar = ctk.CTkProgressBar(gpu_frame, width=300)
            self.gpu_util_bar.pack(fill="x", pady=2)
            self.gpu_util_bar.set(0)

            # GPU memory bar
            self.gpu_mem_label = ctk.CTkLabel(
                gpu_frame,
                text="VRAM: --",
                font=ctk.CTkFont(size=11)
            )
            self.gpu_mem_label.pack(anchor="w", pady=(5, 0))

            self.gpu_mem_bar = ctk.CTkProgressBar(gpu_frame, width=300)
            self.gpu_mem_bar.pack(fill="x", pady=2)
            self.gpu_mem_bar.set(0)
        else:
            # No GPU available message
            no_gpu_label = ctk.CTkLabel(
                self,
                text="No GPU detected",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            no_gpu_label.pack(padx=10, pady=5, anchor="w")

        # CPU section
        cpu_frame = ctk.CTkFrame(self, fg_color="transparent")
        cpu_frame.pack(fill="x", padx=10, pady=10)

        cpu_label = ctk.CTkLabel(
            cpu_frame,
            text="CPU",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        cpu_label.pack(anchor="w", pady=(0, 2))

        self.cpu_label = ctk.CTkLabel(
            cpu_frame,
            text="Usage: --",
            font=ctk.CTkFont(size=11)
        )
        self.cpu_label.pack(anchor="w")

        self.cpu_bar = ctk.CTkProgressBar(cpu_frame, width=300)
        self.cpu_bar.pack(fill="x", pady=2)
        self.cpu_bar.set(0)

        # RAM section
        ram_frame = ctk.CTkFrame(self, fg_color="transparent")
        ram_frame.pack(fill="x", padx=10, pady=5)

        ram_label = ctk.CTkLabel(
            ram_frame,
            text="RAM",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        ram_label.pack(anchor="w", pady=(0, 2))

        self.ram_label = ctk.CTkLabel(
            ram_frame,
            text="Usage: --",
            font=ctk.CTkFont(size=11)
        )
        self.ram_label.pack(anchor="w")

        self.ram_bar = ctk.CTkProgressBar(ram_frame, width=300)
        self.ram_bar.pack(fill="x", pady=2)
        self.ram_bar.set(0)

    def _start_monitoring(self):
        """Start periodic resource monitoring."""
        self._update_resources()

    def _update_resources(self):
        """Update resource usage displays."""
        try:
            # Update GPU if available
            if GPU_AVAILABLE and self.gpu_detector:
                self._update_gpu()

            # Update CPU
            self._update_cpu()

            # Update RAM
            self._update_ram()

        except Exception as e:
            print(f"[ResourceMonitor] Error updating resources: {e}")

        # Schedule next update
        self._update_job = self.after(self._update_interval, self._update_resources)

    def _update_gpu(self):
        """Update GPU utilization and memory using PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                device_id = 0  # First GPU

                # Get GPU memory info from PyTorch
                mem_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
                mem_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
                mem_total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB

                # Calculate memory percentage (use reserved as it's more accurate)
                mem_pct = (mem_reserved / mem_total) * 100 if mem_total > 0 else 0

                # Update memory display
                self.gpu_mem_label.configure(
                    text=f"VRAM: {mem_reserved:.1f} / {mem_total:.1f} GB ({mem_pct:.0f}%)"
                )
                self.gpu_mem_bar.set(mem_pct / 100)

                # Set color based on memory usage
                if mem_pct > 90:
                    self.gpu_mem_bar.configure(progress_color="red")
                elif mem_pct > 80:
                    self.gpu_mem_bar.configure(progress_color="orange")
                else:
                    self.gpu_mem_bar.configure(progress_color="green")

                # For utilization, show as "Active" if memory is being used
                util_indicator = "Active" if mem_reserved > 0.1 else "Idle"
                self.gpu_util_label.configure(
                    text=f"Status: {util_indicator}"
                )
                # Show green bar at 50% for active, 0% for idle
                util_display = 0.5 if mem_reserved > 0.1 else 0
                self.gpu_util_bar.set(util_display)
                self.gpu_util_bar.configure(progress_color="green" if mem_reserved > 0.1 else "gray")
            else:
                self.gpu_util_label.configure(text="GPU: Not available")
                self.gpu_util_bar.set(0)
                self.gpu_mem_label.configure(text="VRAM: --")
                self.gpu_mem_bar.set(0)

        except Exception as e:
            print(f"[ResourceMonitor] GPU update error: {e}")
            self.gpu_util_label.configure(text="GPU: Error")
            self.gpu_util_bar.set(0)

    def _update_cpu(self):
        """Update CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_label.configure(text=f"Usage: {cpu_percent:.0f}%")
            self.cpu_bar.set(cpu_percent / 100)

            # Set color based on CPU usage
            if cpu_percent > 90:
                self.cpu_bar.configure(progress_color="red")
            elif cpu_percent > 80:
                self.cpu_bar.configure(progress_color="orange")
            else:
                self.cpu_bar.configure(progress_color="green")

        except Exception as e:
            print(f"[ResourceMonitor] CPU update error: {e}")
            self.cpu_label.configure(text="CPU: Error")
            self.cpu_bar.set(0)

    def _update_ram(self):
        """Update RAM usage."""
        try:
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            percent = mem.percent

            self.ram_label.configure(
                text=f"Usage: {used_gb:.1f} / {total_gb:.1f} GB ({percent:.0f}%)"
            )
            self.ram_bar.set(percent / 100)

            # Set color based on RAM usage
            if percent > 90:
                self.ram_bar.configure(progress_color="red")
            elif percent > 80:
                self.ram_bar.configure(progress_color="orange")
            else:
                self.ram_bar.configure(progress_color="green")

        except Exception as e:
            print(f"[ResourceMonitor] RAM update error: {e}")
            self.ram_label.configure(text="RAM: Error")
            self.ram_bar.set(0)

    def cleanup(self):
        """Cleanup resources before widget is destroyed."""
        # Cancel pending update
        if self._update_job:
            self.after_cancel(self._update_job)
