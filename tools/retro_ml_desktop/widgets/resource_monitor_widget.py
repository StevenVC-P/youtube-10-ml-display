"""
Resource Usage Monitor Widget - Real-time system resource monitoring.

Displays:
- GPU utilization and VRAM usage (with temperature and power draw)
- CPU usage
- System RAM usage

Updates every 1 second via the new GPUMonitor module.
"""

import customtkinter as ctk
import psutil
import logging
from typing import Optional

from tools.retro_ml_desktop.gpu_monitor import get_gpu_monitor, GPUMetrics


logger = logging.getLogger(__name__)

class ResourceMonitorWidget(ctk.CTkFrame):
    """Widget displaying real-time system resource usage."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        # Get the global GPU monitor instance
        self.gpu_monitor = get_gpu_monitor()
        self.gpu_info = self.gpu_monitor.gpu_info
        self.latest_gpu_metrics: Optional[GPUMetrics] = None

        self._update_interval = 1000  # 1 second (faster updates)
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
        if self.gpu_info and self.gpu_info.is_available:
            gpu_frame = ctk.CTkFrame(self, fg_color="transparent")
            gpu_frame.pack(fill="x", padx=10, pady=5)

            # GPU Header with device name
            gpu_name = self.gpu_info.name or "NVIDIA GPU"
            gpu_header = ctk.CTkLabel(
                gpu_frame,
                text=f"GPU: {gpu_name}",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            gpu_header.pack(anchor="w", pady=(0, 2))

            # Device info (CUDA capability, VRAM)
            cuda_cap = f"{self.gpu_info.cuda_capability[0]}.{self.gpu_info.cuda_capability[1]}"
            info_text = f"CUDA Capability: {cuda_cap} | VRAM: {self.gpu_info.total_memory_gb:.1f} GB"

            self.gpu_info_label = ctk.CTkLabel(
                gpu_frame,
                text=info_text,
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            self.gpu_info_label.pack(anchor="w", pady=(0, 5))

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

            # GPU memory bar (Total VRAM)
            self.gpu_mem_label = ctk.CTkLabel(
                gpu_frame,
                text="VRAM: --",
                font=ctk.CTkFont(size=11)
            )
            self.gpu_mem_label.pack(anchor="w", pady=(5, 0))

            self.gpu_mem_bar = ctk.CTkProgressBar(gpu_frame, width=300)
            self.gpu_mem_bar.pack(fill="x", pady=2)
            self.gpu_mem_bar.set(0)

            # Temperature and Power (optional metrics)
            self.gpu_temp_power_label = ctk.CTkLabel(
                gpu_frame,
                text="Temp: -- | Power: --",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            self.gpu_temp_power_label.pack(anchor="w", pady=(2, 0))
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
        # Register callback with GPU monitor
        if self.gpu_info and self.gpu_info.is_available:
            self.gpu_monitor.add_callback(self._on_gpu_metrics)
            # Start the GPU monitor if not already running
            if not self.gpu_monitor.is_running:
                self.gpu_monitor.start()

        self._update_resources()

    def _on_gpu_metrics(self, metrics: GPUMetrics):
        """Callback for GPU metrics updates."""
        self.latest_gpu_metrics = metrics

    def _update_resources(self):
        """Update resource usage displays."""
        try:
            # Update GPU if available
            if self.gpu_info and self.gpu_info.is_available:
                self._update_gpu()

            # Update CPU
            self._update_cpu()

            # Update RAM
            self._update_ram()

        except Exception as e:
            logger.error(f"Error updating resources: {e}")

        # Schedule next update
        self._update_job = self.after(self._update_interval, self._update_resources)

    def _update_gpu(self):
        """Update GPU utilization and memory using the latest metrics."""
        try:
            # Use the latest metrics from the GPU monitor callback
            if not self.latest_gpu_metrics:
                # If no metrics yet, try to get current snapshot
                self.latest_gpu_metrics = self.gpu_monitor.get_current_metrics()

            if not self.latest_gpu_metrics:
                self.gpu_util_label.configure(text="GPU: No data")
                self.gpu_util_bar.set(0)
                self.gpu_mem_label.configure(text="VRAM: --")
                self.gpu_mem_bar.set(0)
                self.gpu_temp_power_label.configure(text="Temp: -- | Power: --")
                return

            metrics = self.latest_gpu_metrics

            # Update GPU utilization
            util_pct = metrics.utilization_percent
            self.gpu_util_label.configure(text=f"Utilization: {util_pct}%")
            self.gpu_util_bar.set(util_pct / 100)

            # Set color based on utilization
            if util_pct > 90:
                self.gpu_util_bar.configure(progress_color="red")
            elif util_pct > 70:
                self.gpu_util_bar.configure(progress_color="orange")
            else:
                self.gpu_util_bar.configure(progress_color="green")

            # Update VRAM usage
            mem_pct = metrics.memory_percent
            self.gpu_mem_label.configure(
                text=f"VRAM: {metrics.memory_used_gb:.1f} / {metrics.memory_total_gb:.1f} GB ({mem_pct:.0f}%)"
            )
            self.gpu_mem_bar.set(mem_pct / 100)

            # Set color based on memory usage
            if mem_pct > 90:
                self.gpu_mem_bar.configure(progress_color="red")
            elif mem_pct > 80:
                self.gpu_mem_bar.configure(progress_color="orange")
            else:
                self.gpu_mem_bar.configure(progress_color="green")

            # Update temperature and power draw
            temp_text = f"{metrics.temperature_c}°C" if metrics.temperature_c is not None else "--"
            power_text = f"{metrics.power_draw_w:.1f}W" if metrics.power_draw_w is not None else "--"
            self.gpu_temp_power_label.configure(text=f"Temp: {temp_text} | Power: {power_text}")

        except Exception as e:
            logger.error(f"GPU update error: {e}")
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

        # Remove GPU monitor callback
        if self.gpu_info and self.gpu_info.is_available:
            self.gpu_monitor.remove_callback(self._on_gpu_metrics)
