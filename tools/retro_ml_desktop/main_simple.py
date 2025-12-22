"""
Retro ML Desktop - Simple Process Manager (No Docker Required)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import yaml
import re
import threading
import logging
import subprocess
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path so we can import our modules
# Handle both frozen and normal execution
if getattr(sys, 'frozen', False):
    # Running as frozen executable
    project_root = Path(sys.executable).parent
else:
    # Running as normal Python script
    project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Fix Windows console encoding to support Unicode/emoji characters
if sys.platform == 'win32':
    import io
    # Reconfigure stdout and stderr to use UTF-8 with error handling
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from tools.retro_ml_desktop.monitor import SystemMonitor, SystemMetrics, get_gpu_status_message
from tools.retro_ml_desktop.process_manager import ProcessManager, ProcessInfo, ResourceLimits, generate_run_id, get_available_cpus, get_available_gpus, get_detailed_cpu_info, get_detailed_gpu_info, get_recommended_resources
from tools.retro_ml_desktop.resource_selector import ResourceSelectorDialog
from tools.retro_ml_desktop.ram_cleaner import RAMCleanupDialog, get_memory_recommendations
from tools.retro_ml_desktop.storage_cleaner import show_storage_cleanup_dialog
from tools.retro_ml_desktop.video_player import VideoPlayerDialog, play_video_with_player, get_video_info
from tools.retro_ml_desktop.ml_database import MetricsDatabase
from tools.retro_ml_desktop.ml_collector import MetricsCollector
from tools.retro_ml_desktop.ml_dashboard import MLDashboard
from tools.retro_ml_desktop.run_supervisor import LOG_PATH as SUPERVISOR_LOG_PATH
from tools.retro_ml_desktop.cuda_diagnostics import CUDADiagnostics, create_user_friendly_error_message
from tools.retro_ml_desktop.widgets import (
    RecentActivityWidget,
    LiveProgressWidget,
    ResourceMonitorWidget,
    CollapsibleFrame,
    StatusBadge
)
from tools.retro_ml_desktop.theme import Theme
from tools.retro_ml_desktop.naming import build_display_name, next_branch_token


class RetroMLSimple:
    """Simple ML training process manager - no Docker required."""

    def __init__(self, config=None):
        """
        Initialize Retro ML Simple application.

        Args:
            config: Optional ConfigManager instance. If None, uses default paths.
        """
        # Store config manager
        self.config_manager = config

        # Configure logging to show in terminal
        # Set root logger to WARNING to reduce noise
        # Use UTF-8 encoding to handle emoji characters on Windows
        logging.basicConfig(
            level=logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),  # stdout now uses UTF-8
                logging.FileHandler('ml_dashboard.log', encoding='utf-8', errors='replace')
            ]
        )

        # Set specific loggers to appropriate levels
        # Only show warnings and errors from these modules
        logging.getLogger('tools.retro_ml_desktop.ml_plotting').setLevel(logging.WARNING)
        logging.getLogger('tools.retro_ml_desktop.ml_dashboard').setLevel(logging.WARNING)
        logging.getLogger('tools.retro_ml_desktop.ml_collector').setLevel(logging.WARNING)

        # Keep root logger at INFO for important messages
        logging.getLogger('root').setLevel(logging.INFO)

        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Initialize components
        self.project_root = project_root
        self.system_monitor = SystemMonitor(update_interval=2.0)
        self.process_manager = ProcessManager(str(project_root))

        # Initialize ML tracking system
        # Use config manager's database path if available
        if self.config_manager:
            db_path = str(self.config_manager.get_database_path())
        else:
            db_path = str(project_root / "ml_experiments.db")

        self.ml_database = MetricsDatabase(db_path)
        self.ml_collector = MetricsCollector(self.ml_database)

        # Connect database to process manager (Phase 1: Use set_database to initialize Experiment Manager)
        self.process_manager.set_database(self.ml_database)

        # Background GPU telemetry sampler (writes global GPU samples per active run)
        self._gpu_sampler_stop_event = threading.Event()
        self._gpu_sampler_thread = None
        self._gpu_sampler_init_thread = None
        self._start_gpu_sampler_async()

        # Initialize CUDA diagnostics
        self.cuda_diagnostics = CUDADiagnostics()

        self.presets = {}
        self.games = []
        self.algorithms = []
        self._last_resource_config = None  # Store last resource configuration
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Retro ML Desktop - Simple Process Manager")
        self.root.geometry("1000x700")

        # Setup close handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Load presets
        self._load_presets()
        
        # Create UI
        self._create_ui()
        
        # Start monitoring
        self.system_monitor.add_callback(self._update_dashboard)
        self.system_monitor.start_monitoring()

        # Recover any paused processes after UI is setup
        self.root.after(1000, self.recover_paused_processes)
        
        # Refresh processes initially
        self._refresh_processes()

        # Start supervisor to keep runs alive if UI exits
        self._ensure_supervisor_running()

    def _start_gpu_sampler_async(self):
        """Kick off GPU telemetry without blocking UI startup (torch import can be slow)."""
        if self._gpu_sampler_init_thread and self._gpu_sampler_init_thread.is_alive():
            return

        def _init_sampler():
            try:
                self._start_gpu_telemetry_sampler()
            except Exception:
                # Telemetry is optional; failures should not block the app.
                return

        self._gpu_sampler_init_thread = threading.Thread(
            target=_init_sampler, daemon=True, name="GpuSamplerInit"
        )
        self._gpu_sampler_init_thread.start()

    def _start_gpu_telemetry_sampler(self):
        """Start background sampling of GPU metrics into the database for active runs."""
        try:
            from tools.retro_ml_desktop.gpu_monitor import get_gpu_monitor, get_gpu_pid_vram_usage_mb

            gpu_monitor = get_gpu_monitor()
            if not gpu_monitor.is_running:
                gpu_monitor.start()

            def sampler_loop():
                while not self._gpu_sampler_stop_event.wait(2.0):
                    try:
                        metrics = gpu_monitor.get_current_metrics()
                        if not metrics:
                            continue

                        active_run_pids = self.process_manager.get_active_run_pids()
                        if not active_run_pids:
                            continue

                        ts = datetime.now().isoformat()
                        vram_used_mb = float(metrics.memory_used_gb) * 1024.0
                        vram_total_mb = float(metrics.memory_total_gb) * 1024.0

                        pid_vram_map = get_gpu_pid_vram_usage_mb(gpu_index=0)
                        active_pid_mem = 0.0
                        for pid in active_run_pids.values():
                            used = pid_vram_map.get(pid)
                            if used and used > 0:
                                active_pid_mem += float(used)

                        for run_id, pid in active_run_pids.items():
                            pid_vram_used_mb = pid_vram_map.get(pid)
                            run_gpu_util_est_pct = None
                            if active_pid_mem > 0 and pid_vram_used_mb and pid_vram_used_mb > 0:
                                run_gpu_util_est_pct = float(metrics.utilization_percent) * (float(pid_vram_used_mb) / active_pid_mem)

                            self.ml_database.add_gpu_metric(
                                run_id=run_id,
                                ts=ts,
                                gpu_util_pct=float(metrics.utilization_percent),
                                vram_used_mb=vram_used_mb,
                                vram_total_mb=vram_total_mb,
                                temp_c=float(metrics.temperature_c) if metrics.temperature_c is not None else None,
                                power_w=float(metrics.power_draw_w) if metrics.power_draw_w is not None else None,
                                pid=int(pid) if pid is not None else None,
                                pid_vram_used_mb=float(pid_vram_used_mb) if pid_vram_used_mb is not None else None,
                                run_gpu_util_est_pct=float(run_gpu_util_est_pct) if run_gpu_util_est_pct is not None else None,
                            )
                    except Exception:
                        # Keep sampler resilient; avoid spamming the UI thread/logs.
                        continue

            self._gpu_sampler_thread = threading.Thread(target=sampler_loop, daemon=True, name="GpuTelemetrySampler")
            self._gpu_sampler_thread.start()
        except Exception:
            # GPU telemetry is optional; do not break app init if unavailable.
            return

    def _ensure_supervisor_running(self):
        """Start run supervisor if not already running."""
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                cmdline = proc.info.get("cmdline") or []
                joined = " ".join(cmdline).lower()
                if "tools.retro_ml_desktop.run_supervisor" in joined or "run_supervisor.py" in joined:
                    return
        except Exception:
            # If detection fails, fall back to attempting start
            pass

        try:
            cmd = [
                sys.executable,
                "-m",
                "tools.retro_ml_desktop.run_supervisor",
                "--db",
                str(self.ml_database.db_path),
            ]
            creationflags = 0
            close_fds = True
            if sys.platform == "win32":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                close_fds = False
            subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                creationflags=creationflags,
                close_fds=close_fds,
            )
            self._append_log(f"[supervisor] started background supervisor; log: {SUPERVISOR_LOG_PATH}")
        except Exception as e:
            self._append_log(f"[supervisor] failed to start: {e}")
    
    def _load_presets(self):
        """Load training presets from YAML file."""
        presets_file = Path(__file__).parent / "training_presets.yaml"
        try:
            with open(presets_file, 'r') as f:
                data = yaml.safe_load(f)
                self.presets = data.get('presets', {})
                self.games = data.get('games', [
                    'BreakoutNoFrameskip-v4',
                    'PongNoFrameskip-v4', 
                    'SpaceInvadersNoFrameskip-v4',
                    'AsteroidsNoFrameskip-v4',
                    'MsPacmanNoFrameskip-v4',
                    'FroggerNoFrameskip-v4'
                ])
                self.algorithms = data.get('algorithms', ['ppo', 'dqn'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load presets: {e}")
            self.presets = {}
            self.games = []
            self.algorithms = ['ppo', 'dqn']
    
    def _create_ui(self):
        """Create the main user interface."""
        # Create main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create sidebar
        self._create_sidebar(main_frame)
        
        # Create main content area with tabs
        self._create_main_content(main_frame)
    
    def _create_sidebar(self, parent):
        """Create the sidebar with controls."""
        sidebar = ctk.CTkFrame(parent, width=250)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)
        
        # Title
        title = ctk.CTkLabel(sidebar, text="ML Training", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=(20, 10))
        
        # Start Training button
        start_btn = ctk.CTkButton(
            sidebar, text="Start Training", font=ctk.CTkFont(size=14, weight="bold"),
            height=40, command=self._show_start_training_dialog
        )
        start_btn.pack(pady=20, padx=10, fill="x")

        # CUDA Diagnostics button
        diagnostics_btn = ctk.CTkButton(
            sidebar, text="CUDA Diagnostics", font=ctk.CTkFont(size=12),
            height=35, command=self._show_cuda_diagnostics,
            **Theme.get_button_colors("info")
        )
        diagnostics_btn.pack(pady=(5, 20), padx=10, fill="x")

        # Visual separator
        separator1 = ctk.CTkFrame(sidebar, height=2, fg_color=Theme.SEPARATOR)
        separator1.pack(fill="x", padx=20, pady=10)

        # System info
        ctk.CTkLabel(sidebar, text="System Status:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        
        # CPU info
        self.cpu_info_label = ctk.CTkLabel(sidebar, text="CPU: Loading...")
        self.cpu_info_label.pack(pady=5, padx=10, anchor="w")
        
        # Memory info
        self.memory_info_label = ctk.CTkLabel(sidebar, text="Memory: Loading...")
        self.memory_info_label.pack(pady=5, padx=10, anchor="w")
        
        # GPU info
        gpu_status = get_gpu_status_message()
        self.gpu_info_label = ctk.CTkLabel(sidebar, text=f"GPU: {gpu_status}")
        self.gpu_info_label.pack(pady=5, padx=10, anchor="w")

        # Visual separator
        separator2 = ctk.CTkFrame(sidebar, height=2, fg_color=Theme.SEPARATOR)
        separator2.pack(fill="x", padx=20, pady=15)

        # Available resources with detailed info
        ctk.CTkLabel(sidebar, text="System Resources:", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 5))

        # Get detailed resource information
        cpu_info = get_detailed_cpu_info()
        gpu_info = get_detailed_gpu_info()
        recommendations = get_recommended_resources()

        available_cpus = len([cpu for cpu in cpu_info if cpu.available])
        available_gpus = len([gpu for gpu in gpu_info if gpu.available])

        resources_text = (
            f"CPU: {available_cpus}/{len(cpu_info)} cores available\n"
            f"GPU: {available_gpus}/{len(gpu_info)} devices available\n"
            f"Recommended: {recommendations['cpu_cores']} cores"
        )
        ctk.CTkLabel(sidebar, text=resources_text, justify="left").pack(pady=5, padx=10, anchor="w")

        # Advanced resource selector button
        advanced_resources_btn = ctk.CTkButton(
            sidebar, text="Advanced Resource Selection",
            command=self._show_resource_selector,
            font=ctk.CTkFont(size=12),
            height=30
        )
        advanced_resources_btn.pack(pady=5, padx=10, fill="x")

        # RAM cleanup button
        ram_cleanup_btn = ctk.CTkButton(
            sidebar, text="RAM Cleanup & Optimization",
            command=self._show_ram_cleanup,
            font=ctk.CTkFont(size=12),
            height=30
        )
        ram_cleanup_btn.pack(pady=5, padx=10, fill="x")

        # Storage cleanup button
        storage_cleanup_btn = ctk.CTkButton(
            sidebar, text="Storage Cleanup",
            command=self._show_storage_cleanup,
            font=ctk.CTkFont(size=12),
            height=30,
            **Theme.get_button_colors("danger")
        )
        storage_cleanup_btn.pack(pady=5, padx=10, fill="x")

        # Visual separator
        separator3 = ctk.CTkFrame(sidebar, height=2, fg_color=Theme.SEPARATOR)
        separator3.pack(fill="x", padx=20, pady=15)

        # Training controls info
        controls_frame = ctk.CTkFrame(sidebar)
        controls_frame.pack(fill="x", padx=10, pady=(5, 5))

        ctk.CTkLabel(controls_frame, text="Training Controls",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))

        controls_text = (
            "- Stop: Gracefully terminate training\n"
            "- Pause: Suspend training (resume later)\n"
            "- Resume: Continue paused training\n"
            "- Clear Data: Delete training outputs\n"
            "- Remove: Remove from process list"
        )

        ctk.CTkLabel(controls_frame, text=controls_text,
                    font=ctk.CTkFont(size=11), justify="left").pack(pady=(0, 10), padx=10)
    
    def _create_main_content(self, parent):
        """Create the main content area with tabs."""
        # Create tabview
        self.tabview = ctk.CTkTabview(parent)
        self.tabview.pack(side="right", fill="both", expand=True)

        # Create tabs
        self.processes_tab = self.tabview.add("Training Processes")
        self.ml_dashboard_tab = self.tabview.add("ML Dashboard")
        self.videos_tab = self.tabview.add("Video Gallery")
        self.settings_tab = self.tabview.add("Settings")

        # Setup each tab
        self._setup_processes_tab()
        self._setup_ml_dashboard_tab()
        self._setup_videos_tab()
        self._setup_settings_tab()
    
    def _setup_processes_tab(self):
        """Setup the processes tab with process list."""
        # Control buttons
        controls_frame = ctk.CTkFrame(self.processes_tab)
        controls_frame.pack(fill="x", padx=10, pady=5)

        refresh_btn = ctk.CTkButton(controls_frame, text="Refresh", command=self._refresh_processes)
        refresh_btn.pack(side="left", padx=5, pady=5)

        stop_btn = ctk.CTkButton(controls_frame, text="Stop Selected", command=self._stop_selected_process,
                                **Theme.get_button_colors("danger"))
        stop_btn.pack(side="left", padx=5)

        pause_btn = ctk.CTkButton(controls_frame, text="Pause Selected", command=self._pause_selected_process,
                                 **Theme.get_button_colors("warning"))
        pause_btn.pack(side="left", padx=5)

        resume_btn = ctk.CTkButton(controls_frame, text="Resume Selected", command=self._resume_selected_process,
                                   **Theme.get_button_colors("success"))
        resume_btn.pack(side="left", padx=5)

        remove_btn = ctk.CTkButton(controls_frame, text="Remove Selected", command=self._remove_selected_process)
        remove_btn.pack(side="left", padx=5, pady=5)

        # Clear data buttons
        clear_frame = ctk.CTkFrame(self.processes_tab)
        clear_frame.pack(fill="x", padx=10, pady=5)

        clear_selected_btn = ctk.CTkButton(clear_frame, text="Clear Selected Data",
                                         command=self._clear_selected_data, **Theme.get_button_colors("warning"))
        clear_selected_btn.pack(side="left", padx=5)

        clear_all_btn = ctk.CTkButton(clear_frame, text="Clear ALL Training Data",
                                    command=self._clear_all_data, **Theme.get_button_colors("danger"))
        clear_all_btn.pack(side="left", padx=5)
        
        # Process list (using tkinter Treeview for table)
        list_frame = ctk.CTkFrame(self.processes_tab)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create Treeview
        columns = ("Name", "Status", "PID", "Created")
        self.process_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            self.process_tree.heading(col, text=col)
            self.process_tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.process_tree.yview)
        self.process_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.process_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _setup_ml_dashboard_tab(self):
        """Setup the comprehensive ML dashboard tab with tabbed interface."""
        # Create tabbed view with Overview, Charts, Metrics, Activity tabs
        self.dashboard_tabview = ctk.CTkTabview(self.ml_dashboard_tab)
        self.dashboard_tabview.pack(fill="both", expand=True, padx=5, pady=5)

        # Create tabs
        self.dashboard_tabview.add("Overview")
        self.dashboard_tabview.add("Charts")
        self.dashboard_tabview.add("Metrics")
        self.dashboard_tabview.add("Activity")

        # Get experiment manager
        experiment_manager = self.process_manager.experiment_manager

        # Setup each tab
        self._setup_overview_tab(experiment_manager)
        self._setup_charts_tab()
        self._setup_metrics_tab()
        self._setup_activity_tab(experiment_manager)

    def _setup_overview_tab(self, experiment_manager):
        """Setup Overview tab with collapsible sections using new CollapsibleFrame widget."""
        overview_tab = self.dashboard_tabview.tab("Overview")

        # Container for collapsible sections
        container = ctk.CTkScrollableFrame(overview_tab)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # Section 1: Charts (using new CollapsibleFrame)
        charts_section = CollapsibleFrame(
            container,
            title="Charts",
            icon="📊",
            collapsed_icon="📊",
            initially_collapsed=False
        )
        charts_section.pack(fill="both", expand=True, pady=(0, 10))

        # Charts content
        charts_content = charts_section.get_content_frame()
        self.ml_dashboard_overview = MLDashboard(
            parent_frame=charts_content,
            database=self.ml_database,
            collector=self.ml_collector,
            process_manager=self.process_manager
        )

        # Section 2: Live Metrics (Live Progress + Resource Monitor)
        metrics_section = CollapsibleFrame(
            container,
            title="Live Metrics",
            icon="📈",
            collapsed_icon="📈",
            initially_collapsed=False
        )
        metrics_section.pack(fill="both", expand=False, pady=(0, 10))

        # Metrics content (side by side)
        metrics_content = metrics_section.get_content_frame()
        metrics_content.configure(height=220)
        metrics_content.pack_propagate(False)

        # Live Progress Widget (left)
        live_progress_frame = ctk.CTkFrame(metrics_content)
        live_progress_frame.pack(side="left", fill="both", expand=True, padx=(0, 2.5))
        self.live_progress_widget = LiveProgressWidget(parent=live_progress_frame)
        self.live_progress_widget.pack(fill="both", expand=True)

        # Resource Monitor Widget (right)
        resource_monitor_frame = ctk.CTkFrame(metrics_content)
        resource_monitor_frame.pack(side="right", fill="both", expand=True, padx=(2.5, 0))
        self.resource_monitor_widget = ResourceMonitorWidget(parent=resource_monitor_frame)
        self.resource_monitor_widget.pack(fill="both", expand=True)

        # Section 3: Recent Activity
        activity_section = CollapsibleFrame(
            container,
            title="Recent Activity",
            icon="🕒",
            collapsed_icon="🕒",
            initially_collapsed=False
        )
        activity_section.pack(fill="both", expand=False, pady=(0, 10))

        # Activity content
        activity_content = activity_section.get_content_frame()
        activity_content.configure(height=250)
        activity_content.pack_propagate(False)

        self.recent_activity_widget_overview = RecentActivityWidget(
            experiment_manager=experiment_manager,
            parent=activity_content
        )
        self.recent_activity_widget_overview.pack(fill="both", expand=True)

    def _setup_charts_tab(self):
        """Setup Charts tab with full-screen charts."""
        charts_tab = self.dashboard_tabview.tab("Charts")

        # Full-screen ML Dashboard
        self.ml_dashboard = MLDashboard(
            parent_frame=charts_tab,
            database=self.ml_database,
            collector=self.ml_collector,
            process_manager=self.process_manager
        )

    def _setup_metrics_tab(self):
        """Setup Metrics tab with Live Progress and Resource Monitor."""
        metrics_tab = self.dashboard_tabview.tab("Metrics")

        # Container for side-by-side layout
        container = ctk.CTkFrame(metrics_tab, fg_color="transparent")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Live Progress Widget (left side, larger)
        live_progress_frame = ctk.CTkFrame(container)
        live_progress_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        self.live_progress_widget_metrics = LiveProgressWidget(parent=live_progress_frame)
        self.live_progress_widget_metrics.pack(fill="both", expand=True, padx=10, pady=10)

        # Resource Monitor Widget (right side, larger)
        resource_monitor_frame = ctk.CTkFrame(container)
        resource_monitor_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        self.resource_monitor_widget_metrics = ResourceMonitorWidget(parent=resource_monitor_frame)
        self.resource_monitor_widget_metrics.pack(fill="both", expand=True, padx=10, pady=10)

    def _setup_activity_tab(self, experiment_manager):
        """Setup Activity tab with full-screen Recent Activity widget."""
        activity_tab = self.dashboard_tabview.tab("Activity")

        # Full-screen Recent Activity Widget
        self.recent_activity_widget = RecentActivityWidget(
            experiment_manager=experiment_manager,
            parent=activity_tab
        )
        self.recent_activity_widget.pack(fill="both", expand=True, padx=10, pady=10)

    def _setup_videos_tab(self):
        """Setup the video gallery tab for viewing training videos."""
        # Main container
        main_frame = ctk.CTkFrame(self.videos_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="Training Video Gallery",
                                   font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(10, 20))

        # Controls frame
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Refresh videos button
        refresh_videos_btn = ctk.CTkButton(controls_frame, text="Refresh Videos",
                                         command=self._refresh_videos)
        refresh_videos_btn.pack(side="left", padx=5, pady=5)

        # Generate videos button (NEW!)
        generate_videos_btn = ctk.CTkButton(controls_frame, text="Generate Videos from Training",
                                           command=self._generate_videos_dialog,
                                           **Theme.get_button_colors("primary"))
        generate_videos_btn.pack(side="left", padx=5, pady=5)

        # Video post-processing buttons (NEW!)
        timelapse_btn = ctk.CTkButton(controls_frame, text="Create Time-Lapse",
                                     command=self._create_timelapse_dialog,
                                     **Theme.get_button_colors("info"))
        timelapse_btn.pack(side="left", padx=5, pady=5)

        progression_btn = ctk.CTkButton(controls_frame, text="Milestone Progression",
                                       command=self._create_progression_dialog,
                                       **Theme.get_button_colors("info"))
        progression_btn.pack(side="left", padx=5, pady=5)

        # Open video folder button
        open_folder_btn = ctk.CTkButton(controls_frame, text="Open Video Folder",
                                       command=self._open_video_folder)
        open_folder_btn.pack(side="left", padx=5, pady=5)

        # Video filter dropdown
        filter_label = ctk.CTkLabel(controls_frame, text="Filter:")
        filter_label.pack(side="left", padx=(20, 5), pady=5)

        self.video_filter_var = tk.StringVar(value="All Videos")
        filter_dropdown = ctk.CTkOptionMenu(controls_frame, variable=self.video_filter_var,
                                           values=["All Videos", "Milestone Videos", "Hour Videos", "Evaluation Videos"],
                                           command=self._filter_videos)
        filter_dropdown.pack(side="left", padx=5, pady=5)

        # Video list frame with scrollbar
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create video list (using Treeview for better display)
        columns = ("Name", "Type", "Duration", "Size", "Created", "Training Run")
        self.video_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)

        # Configure columns
        self.video_tree.heading("Name", text="Video Name")
        self.video_tree.heading("Type", text="Type")
        self.video_tree.heading("Duration", text="Duration")
        self.video_tree.heading("Size", text="File Size")
        self.video_tree.heading("Created", text="Created")
        self.video_tree.heading("Training Run", text="Training Run")

        # Column widths
        self.video_tree.column("Name", width=200)
        self.video_tree.column("Type", width=100)
        self.video_tree.column("Duration", width=80)
        self.video_tree.column("Size", width=80)
        self.video_tree.column("Created", width=120)
        self.video_tree.column("Training Run", width=150)

        # Scrollbar for video list
        video_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.video_tree.yview)
        self.video_tree.configure(yscrollcommand=video_scrollbar.set)

        # Pack video list and scrollbar
        self.video_tree.pack(side="left", fill="both", expand=True)
        video_scrollbar.pack(side="right", fill="y")

        # Video action buttons
        action_frame = ctk.CTkFrame(main_frame)
        action_frame.pack(fill="x", padx=10, pady=5)

        play_btn = ctk.CTkButton(action_frame, text="Play Video", command=self._play_selected_video,
                                **Theme.get_button_colors("success"))
        play_btn.pack(side="left", padx=5, pady=5)

        player_btn = ctk.CTkButton(action_frame, text="Video Player", command=self._open_video_player)
        player_btn.pack(side="left", padx=5, pady=5)

        preview_btn = ctk.CTkButton(action_frame, text="Quick Preview", command=self._preview_selected_video)
        preview_btn.pack(side="left", padx=5, pady=5)

        info_btn = ctk.CTkButton(action_frame, text="Video Info", command=self._show_video_info)
        info_btn.pack(side="left", padx=5, pady=5)

        delete_btn = ctk.CTkButton(action_frame, text="Delete Video", command=self._delete_selected_video,
                                  **Theme.get_button_colors("danger"))
        delete_btn.pack(side="right", padx=5, pady=5)

        # Auto-refresh videos when tab is opened
        self._refresh_videos()

        # Start auto-refresh timer for in-progress videos
        self._video_refresh_timer = None
        self._schedule_video_refresh()

    def _setup_settings_tab(self):
        """Setup the settings tab with system configuration and ROM installation."""
        # Main container
        main_frame = ctk.CTkFrame(self.settings_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="Settings & Configuration",
                                   font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(10, 20))

        # Create scrollable frame for settings
        scrollable_frame = ctk.CTkScrollableFrame(main_frame)
        scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ========== ROM Installation Section ==========
        rom_section = ctk.CTkFrame(scrollable_frame)
        rom_section.pack(fill="x", padx=10, pady=(0, 20))

        rom_title = ctk.CTkLabel(rom_section, text="Atari ROM Installation",
                                font=ctk.CTkFont(size=16, weight="bold"))
        rom_title.pack(pady=(15, 10), padx=15, anchor="w")

        rom_info_frame = ctk.CTkFrame(rom_section)
        rom_info_frame.pack(fill="x", padx=15, pady=5)

        # Check ROM status
        rom_status = self._check_rom_status()

        if rom_status['installed']:
            status_prefix = "[OK]"
            status_text = f"{status_prefix} ROMs Installed ({rom_status['count']} games available)"
            status_color = "green"
        else:
            status_prefix = "[WARN]"
            status_text = f"{status_prefix} ROMs Not Installed"
            status_color = "orange"

        self.rom_status_label = ctk.CTkLabel(rom_info_frame,
                                            text=status_text,
                                            font=ctk.CTkFont(size=14),
                                            text_color=status_color)
        self.rom_status_label.pack(pady=10, padx=10, anchor="w")

        rom_desc = ctk.CTkLabel(rom_info_frame,
                               text="Atari 2600 ROMs are required for training. They will be downloaded\n"
                                    "using AutoROM under the Atari 2600 license.",
                               justify="left",
                               text_color="gray")
        rom_desc.pack(pady=5, padx=10, anchor="w")

        # Install/Reinstall button
        button_text = "Reinstall ROMs" if rom_status['installed'] else "Install ROMs"
        self.install_roms_btn = ctk.CTkButton(rom_section,
                                             text=button_text,
                                             command=self._install_roms_from_settings,
                                             height=40,
                                             font=ctk.CTkFont(size=14),
                                             **Theme.get_button_colors("primary"))
        self.install_roms_btn.pack(pady=10, padx=15, fill="x")

        # ROM progress bar (hidden by default)
        self.rom_progress_bar = ctk.CTkProgressBar(rom_section)
        self.rom_progress_bar.set(0)

        # ROM status message (hidden by default)
        self.rom_install_status = ctk.CTkLabel(rom_section, text="", font=ctk.CTkFont(size=12))

        # ========== System Information Section ==========
        system_section = ctk.CTkFrame(scrollable_frame)
        system_section.pack(fill="x", padx=10, pady=(0, 20))

        system_title = ctk.CTkLabel(system_section, text="System Information",
                                   font=ctk.CTkFont(size=16, weight="bold"))
        system_title.pack(pady=(15, 10), padx=15, anchor="w")

        # System info display
        if self.config_manager:
            caps = self.config_manager.config.get('system', {})
            system_info_text = (
                f"GPU Detected: {'Yes' if caps.get('gpu_detected') else 'No'}\n"
                f"CUDA Available: {'Yes' if caps.get('cuda_available') else 'No'}\n"
                f"FFmpeg Available: {'Yes' if caps.get('ffmpeg_available') else 'No'}\n"
                f"Atari ROMs: {'Installed' if caps.get('atari_roms_installed') else 'Not Installed'}"
            )
        else:
            system_info_text = "Configuration not available"

        system_info_label = ctk.CTkLabel(system_section,
                                        text=system_info_text,
                                        justify="left",
                                        font=ctk.CTkFont(size=13))
        system_info_label.pack(pady=10, padx=15, anchor="w")

        # ========== Paths Section ==========
        paths_section = ctk.CTkFrame(scrollable_frame)
        paths_section.pack(fill="x", padx=10, pady=(0, 20))

        paths_title = ctk.CTkLabel(paths_section, text="Configuration Paths",
                                   font=ctk.CTkFont(size=16, weight="bold"))
        paths_title.pack(pady=(15, 10), padx=15, anchor="w")

        if self.config_manager:
            paths_info_text = (
                f"Installation: {self.config_manager.get_path('install_dir')}\n"
                f"Models: {self.config_manager.get_path('models_dir')}\n"
                f"Videos: {self.config_manager.get_path('videos_dir')}\n"
                f"Database: {self.config_manager.get_path('database_dir')}\n"
                f"Logs: {self.config_manager.get_path('logs_dir')}\n"
                f"Config: {self.config_manager.get_path('config_dir')}"
            )
        else:
            paths_info_text = f"Project Root: {self.project_root}"

        paths_info_label = ctk.CTkLabel(paths_section,
                                       text=paths_info_text,
                                       justify="left",
                                       font=ctk.CTkFont(size=12),
                                       text_color="gray")
        paths_info_label.pack(pady=10, padx=15, anchor="w")

    def _check_rom_status(self):
        """Check if Atari ROMs are installed."""
        try:
            import ale_py.roms as roms
            rom_dir = Path(roms.__file__).parent
            rom_files = list(rom_dir.glob('*.bin'))
            return {
                'installed': len(rom_files) > 0,
                'count': len(rom_files),
                'path': str(rom_dir)
            }
        except (ImportError, AttributeError):
            return {
                'installed': False,
                'count': 0,
                'path': None
            }

    def _install_roms_from_settings(self):
        """Install Atari ROMs from the Settings tab."""
        import subprocess
        import sys

        # Disable button and show progress
        self.install_roms_btn.configure(state="disabled")
        self.rom_progress_bar.pack(pady=5, padx=15, fill="x")
        self.rom_progress_bar.set(0)
        self.rom_install_status.configure(text="Installing ROMs...", text_color="blue")
        self.rom_install_status.pack(pady=5, padx=15)

        def install_thread():
            try:
                # Check if autorom is installed
                try:
                    import autorom
                except ImportError:
                    # AutoROM not installed, install it first
                    self.root.after(0, lambda: self.rom_install_status.configure(
                        text="Installing AutoROM package...", text_color="blue"))

                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "autorom[accept-rom-license]"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )

                    if result.returncode != 0:
                        error_msg = "Failed to install AutoROM package"
                        self.root.after(0, lambda: self._on_rom_install_error(error_msg))
                        return

                # Update progress
                self.root.after(0, lambda: self.rom_progress_bar.set(0.3))
                self.root.after(0, lambda: self.rom_install_status.configure(
                    text="Downloading Atari ROMs...", text_color="blue"))

                # Find AutoROM executable
                # AutoROM installs as an executable (AutoROM.exe), not a Python module
                if getattr(sys, 'frozen', False):
                    # Running as frozen executable - AutoROM should be in same dir
                    autorom_exe = Path(sys.executable).parent / "AutoROM.exe"
                else:
                    # Running as script - AutoROM in Scripts dir of Python installation or venv
                    autorom_exe = Path(sys.executable).parent / "AutoROM.exe"
                    if not autorom_exe.exists():
                        # Try Scripts directory
                        autorom_exe = Path(sys.executable).parent / "Scripts" / "AutoROM.exe"

                if not autorom_exe.exists():
                    error_msg = f"AutoROM.exe not found at {autorom_exe}"
                    self.root.after(0, lambda: self._on_rom_install_error(error_msg))
                    return

                # Run autorom installation using the executable
                result = subprocess.run(
                    [str(autorom_exe), "--accept-license"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    # Success
                    self.root.after(0, lambda: self._on_rom_install_success())
                else:
                    # Error
                    error_msg = result.stderr or result.stdout or "Unknown error"
                    self.root.after(0, lambda: self._on_rom_install_error(error_msg))

            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self._on_rom_install_error("Installation timed out"))
            except Exception as e:
                self.root.after(0, lambda: self._on_rom_install_error(str(e)))

        # Run installation in background thread
        threading.Thread(target=install_thread, daemon=True).start()

    def _on_rom_install_success(self):
        """Handle successful ROM installation."""
        self.rom_progress_bar.set(1.0)
        self.rom_install_status.configure(text="ROMs installed successfully!", text_color="green")

        # Update ROM status
        rom_status = self._check_rom_status()
        self.rom_status_label.configure(
            text=f"ROMs Installed ({rom_status['count']} games available)",
            text_color="green"
        )
        self.install_roms_btn.configure(text="Reinstall ROMs", state="normal")

        # Update config if available
        if self.config_manager:
            self.config_manager.set('system.atari_roms_installed', True)
            self.config_manager.save()

        # Hide progress after 3 seconds
        self.root.after(3000, lambda: self.rom_progress_bar.pack_forget())

    def _on_rom_install_error(self, error_msg: str):
        """Handle ROM installation error."""
        self.rom_progress_bar.set(0)
        self.rom_progress_bar.pack_forget()
        self.rom_install_status.configure(
            text=f"Installation failed: {error_msg}\n"
                  "You can try installing manually: pip install autorom[accept-rom-license] && autorom --accept-license",
            text_color="red"
        )
        self.install_roms_btn.configure(state="normal")

    def _update_dashboard(self, metrics: SystemMetrics):
        """Update sidebar with new metrics."""
        def update_ui():
            # Update CPU
            cpu_text = f"CPU: {metrics.cpu.percent:.1f}% ({metrics.cpu.logical_cores} cores)"
            self.cpu_info_label.configure(text=cpu_text)
            
            # Update Memory
            memory_text = f"Memory: {metrics.memory.used_gb:.1f}/{metrics.memory.total_gb:.1f} GB ({metrics.memory.percent:.1f}%)"
            self.memory_info_label.configure(text=memory_text)
            
            # Update GPU
            if metrics.gpus:
                gpu_text = f"GPU: {len(metrics.gpus)} available"
                for gpu in metrics.gpus:
                    if gpu.load_percent > 0:
                        gpu_text += f" (GPU{gpu.id}: {gpu.load_percent:.1f}%)"
                        break
            else:
                gpu_text = f"GPU: {get_gpu_status_message()}"
            
            self.gpu_info_label.configure(text=gpu_text)
        
        # Schedule UI update on main thread
        self.root.after(0, update_ui)
    
    def _refresh_processes(self):
        """Refresh the process list."""
        # Clear existing items
        for item in self.process_tree.get_children():
            self.process_tree.delete(item)
        
        # Get processes
        processes = self.process_manager.get_processes()
        
        # Add processes to tree with status indicators
        for process in processes:
            created_str = process.created.strftime("%Y-%m-%d %H:%M")
            pid_str = str(process.pid) if process.pid else "N/A"

            # Add visual status indicators
            status_display = process.status
            if process.status == "running":
                status_display = "Running"
            elif process.status == "paused":
                status_display = "Paused"
            elif process.status == "stopped":
                status_display = "Stopped"
            elif process.status == "finished":
                status_display = "Finished"
            elif process.status == "failed":
                status_display = "Failed"

            self.process_tree.insert("", "end", values=(
                process.name,
                status_display,
                pid_str,
                created_str
            ), tags=(process.id,))
    
    def _stop_selected_process(self):
        """Stop the selected process."""
        selection = self.process_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a process to stop.")
            return

        item = selection[0]
        process_id = self.process_tree.item(item)["tags"][0]

        # Get process info for better confirmation message
        processes = self.process_manager.get_processes()
        process_info = next((p for p in processes if p.id == process_id), None)

        if process_info:
            confirm_msg = f"Stop training process '{process_info.name}'?\n\nThis will terminate the training and save current progress."
        else:
            confirm_msg = "Stop selected training process?"

        if messagebox.askyesno("Confirm Stop", confirm_msg):
            self._append_log(f"Stopping process: {process_id}...")
            success = self.process_manager.stop_process(process_id)
            if success:
                self._refresh_processes()
                self._append_log(f"✅ Successfully stopped process: {process_id}")
                messagebox.showinfo("Success", f"Process {process_id} stopped successfully.")
            else:
                self._append_log(f"❌ Failed to stop process: {process_id}")
                messagebox.showerror("Error", f"Failed to stop process {process_id}. It may have already finished or crashed.")
    
    def _pause_selected_process(self):
        """Pause the selected process."""
        selection = self.process_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a process to pause.")
            return

        item = selection[0]
        process_id = self.process_tree.item(item)["tags"][0]

        # Get process info for better confirmation message
        processes = self.process_manager.get_processes()
        process_info = next((p for p in processes if p.id == process_id), None)

        if process_info and process_info.status == "paused":
            messagebox.showinfo("Already Paused", f"Process {process_id} is already paused.")
            return

        if process_info:
            confirm_msg = f"Pause training process '{process_info.name}'?\n\nThis will suspend the process but keep it in memory. You can resume it later."
        else:
            confirm_msg = "Pause selected training process?"

        if messagebox.askyesno("Confirm Pause", confirm_msg):
            self._append_log(f"Pausing process: {process_id}...")
            success = self.process_manager.pause_process(process_id)
            if success:
                self._refresh_processes()
                self._append_log(f"⏸️ Successfully paused process: {process_id}")
                messagebox.showinfo("Success", f"Process {process_id} paused successfully. Use Resume to continue.")
            else:
                self._append_log(f"❌ Failed to pause process: {process_id}")
                messagebox.showerror("Error", f"Failed to pause process {process_id}. It may have already finished or crashed.")

    def _resume_selected_process(self):
        """Resume the selected process."""
        selection = self.process_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a process to resume.")
            return

        item = selection[0]
        process_id = self.process_tree.item(item)["tags"][0]

        # Get process info for better confirmation message
        processes = self.process_manager.get_processes()
        process_info = next((p for p in processes if p.id == process_id), None)

        if process_info and process_info.status != "paused":
            messagebox.showinfo("Not Paused", f"Process {process_id} is not paused (status: {process_info.status}).")
            return

        if process_info:
            confirm_msg = f"Resume training process '{process_info.name}'?\n\nThis will continue the paused training from where it left off."
        else:
            confirm_msg = "Resume selected training process?"

        if messagebox.askyesno("Confirm Resume", confirm_msg):
            self._append_log(f"Resuming process: {process_id}...")
            success = self.process_manager.resume_process(process_id)
            if success:
                self._refresh_processes()
                self._append_log(f"▶️ Successfully resumed process: {process_id}")
                messagebox.showinfo("Success", f"Process {process_id} resumed successfully.")
            else:
                self._append_log(f"❌ Failed to resume process: {process_id}")
                messagebox.showerror("Error", f"Failed to resume process {process_id}. It may have crashed or been terminated.")

    def _remove_selected_process(self):
        """Remove the selected process."""
        selection = self.process_tree.selection()
        if not selection:
            return

        item = selection[0]
        process_id = self.process_tree.item(item)["tags"][0]

        if messagebox.askyesno("Confirm", "Remove selected process from list?"):
            success = self.process_manager.remove_process(process_id)
            if success:
                self._refresh_processes()
                self._append_log(f"Removed process: {process_id}")
            else:
                messagebox.showerror("Error", "Failed to remove process")

    def _clear_selected_data(self):
        """Clear training data for the selected process."""
        selection = self.process_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a process to clear data for.")
            return

        item = selection[0]
        process_id = self.process_tree.item(item)["tags"][0]

        # Confirm action
        result = messagebox.askyesno("Confirm Clear Data",
                                   f"Are you sure you want to clear all training data for {process_id}?\n\n"
                                   f"This will delete:\n"
                                   f"• Model checkpoints\n"
                                   f"• Generated videos\n"
                                   f"• Training logs\n\n"
                                   f"This action cannot be undone!")

        if result:
            success = self.process_manager.clear_training_data(process_id=process_id)
            if success:
                messagebox.showinfo("Success", f"Training data cleared for {process_id}.")
                self._refresh_progress()
                self._append_log(f"Cleared training data for: {process_id}")
            else:
                messagebox.showerror("Error", f"Failed to clear training data for {process_id}.")

    def _clear_all_data(self):
        """Clear all training data."""
        # Confirm action with strong warning
        result = messagebox.askyesno("⚠️ DANGER: Clear ALL Data",
                                   f"⚠️ WARNING: This will delete ALL training data!\n\n"
                                   f"This includes:\n"
                                   f"• ALL model checkpoints\n"
                                   f"• ALL generated videos\n"
                                   f"• ALL training logs\n"
                                   f"• ALL output files\n\n"
                                   f"This action CANNOT be undone!\n\n"
                                   f"Are you absolutely sure?")

        if result:
            # Double confirmation
            result2 = messagebox.askyesno("Final Confirmation",
                                        f"This is your final warning!\n\n"
                                        f"Clicking 'Yes' will permanently delete ALL training data.\n\n"
                                        f"Continue?")

            if result2:
                success = self.process_manager.clear_training_data(clear_all=True)
                if success:
                    messagebox.showinfo("Success", "All training data has been cleared.")
                    self._refresh_processes()
                    self._refresh_progress()
                    self._append_log("Cleared ALL training data")
                else:
                    messagebox.showerror("Error", "Failed to clear all training data.")

    def _show_resource_selector(self):
        """Show the advanced resource selection dialog."""
        dialog = ResourceSelectorDialog(self.root)
        result = dialog.show()

        if result:
            # Show the selected configuration
            config_text = (
                f"Selected Configuration:\n"
                f"• CPU Cores: {result['cpu_cores']}\n"
                f"• GPU: {result['gpu_id']}\n"
                f"• Memory Limit: {result.get('memory_limit_gb', 'No limit')} GB\n\n"
                f"This configuration will be used for the next training session."
            )
            messagebox.showinfo("Resource Configuration", config_text)

            # Store for next training session
            self._last_resource_config = result

    def _show_ram_cleanup(self):
        """Show the RAM cleanup dialog."""
        dialog = RAMCleanupDialog(self.root)
        dialog.show()

        # Refresh system monitor after cleanup
        self._update_system_status()

    def _show_storage_cleanup(self):
        """Show the storage cleanup dialog."""
        show_storage_cleanup_dialog(self.root, self.project_root)

        # Refresh video gallery after cleanup
        self._refresh_videos()

    def _append_log(self, message: str):
        """Append a message to the log (uses Python logging system)."""
        # Use Python's logging system instead of UI widget
        # This allows logs to be captured in terminal and log file

        # Replace emoji characters with ASCII equivalents for Windows console compatibility
        emoji_replacements = {
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🎬': '[VIDEO]',
            '🎥': '[CAMERA]',
            '📹': '[REC]',
            '⏸️': '[PAUSE]',
            '▶️': '[PLAY]',
            '⏹️': '[STOP]',
            '🔄': '[REFRESH]',
            '🗑️': '[DELETE]',
            '🧠': '[RAM]',
            '💾': '[SAVE]',
            '📊': '[STATS]',
            '🎮': '[GAME]',
            '🧪': '[TEST]',
            '⚡': '[FAST]',
            '🔧': '[CONFIG]',
            '📦': '[PACKAGE]',
        }

        clean_message = message
        for emoji, replacement in emoji_replacements.items():
            clean_message = clean_message.replace(emoji, replacement)

        logging.info(clean_message)

    def _refresh_progress(self):
        """Refresh the progress tracking display."""
        # Clear existing progress widgets
        for widget in self.progress_content.winfo_children():
            if hasattr(widget, 'pack_info') and widget.pack_info():
                widget.destroy()

        # Get all processes
        processes = self.process_manager.get_processes()

        if not processes:
            no_processes_label = ctk.CTkLabel(self.progress_content,
                                            text="No training processes running",
                                            font=ctk.CTkFont(size=14))
            no_processes_label.pack(pady=20)
            return

        # Create progress display for each process
        for process in processes:
            self._create_process_progress_widget(process)

    def _create_process_progress_widget(self, process):
        """Create a progress widget for a single training process."""
        # Main frame for this process
        process_frame = ctk.CTkFrame(self.progress_content)
        process_frame.pack(fill="x", padx=10, pady=10)

        # Process header
        header_text = f"🎮 {process.name} (ID: {process.id})"
        header_label = ctk.CTkLabel(process_frame, text=header_text,
                                   font=ctk.CTkFont(size=16, weight="bold"))
        header_label.pack(pady=(10, 5))

        # Status
        status_text = f"Status: {process.status}"
        if process.pid:
            status_text += f" | PID: {process.pid}"
        status_label = ctk.CTkLabel(process_frame, text=status_text)
        status_label.pack(pady=2)

        # Get progress information
        progress_info = self._get_process_progress(process)

        # Training Progress Section
        training_frame = ctk.CTkFrame(process_frame)
        training_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(training_frame, text="Training Progress",
                    font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        if progress_info['training']:
            training = progress_info['training']

            # Check if we have leg information
            if 'leg_number' in training and training['leg_number'] > 1:
                # Multi-leg run - show both leg progress and total progress
                training_text = (
                    f"Leg {training['leg_number']} Progress: {training['leg_steps_completed']:,} / {training['leg_total_steps']:,} "
                    f"({training['progress_pct']:.1f}%)\n"
                    f"Total Progress: {training['total_timesteps_all_legs']:,} timesteps completed\n"
                    f"ETA: {training['eta']}\n"
                    f"Current Reward: {training['current_reward']}\n"
                    f"FPS: {training['fps']}\n"
                    f"Checkpoints: {training['checkpoints_saved']}"
                )
            elif 'leg_number' in training:
                # First leg - show simpler display
                training_text = (
                    f"Progress: {training['current_steps']:,} / {training['total_steps']:,} "
                    f"({training['progress_pct']:.1f}%)\n"
                    f"ETA: {training['eta']}\n"
                    f"Current Reward: {training['current_reward']}\n"
                    f"FPS: {training['fps']}\n"
                    f"Checkpoints: {training['checkpoints_saved']}"
                )
            else:
                # Fallback for runs without leg tracking
                training_text = (
                    f"Timesteps: {training['current_steps']:,} / {training['total_steps']:,} "
                    f"({training['progress_pct']:.1f}%)\n"
                    f"ETA: {training['eta']}\n"
                    f"Current Reward: {training['current_reward']}\n"
                    f"FPS: {training['fps']}\n"
                    f"Checkpoints: {training['checkpoints_saved']}"
                )
        else:
            training_text = "Training progress not available (starting up...)"

        ctk.CTkLabel(training_frame, text=training_text, justify="left").pack(pady=2, padx=10)

        # Video Generation Progress Section
        video_frame = ctk.CTkFrame(process_frame)
        video_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(video_frame, text="Video Generation Progress",
                    font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        if progress_info['videos']:
            video_text = (
                f"Milestone Videos: {progress_info['videos']['milestones_completed']} / "
                f"{progress_info['videos']['total_milestones']} completed\n"
                f"Hour Videos: {progress_info['videos']['hour_videos_completed']} / "
                f"{progress_info['videos']['target_hours']} completed\n"
                f"Total Video Duration: {progress_info['videos']['total_duration']}"
            )
        else:
            video_text = "Video generation not started"

        ctk.CTkLabel(video_frame, text=video_text, justify="left").pack(pady=2, padx=10)

        # Files and Output Section
        files_frame = ctk.CTkFrame(process_frame)
        files_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkLabel(files_frame, text="Output Files",
                    font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        files_text = (
            f"Model Checkpoints: {progress_info['files']['checkpoints']}\n"
            f"Milestone Videos: {progress_info['files']['milestone_videos']}\n"
            f"Hour Videos: {progress_info['files']['hour_videos']}\n"
            f"Output Directory: {progress_info['files']['output_dir']}"
        )

        ctk.CTkLabel(files_frame, text=files_text, justify="left").pack(pady=2, padx=10)

    def _get_process_progress(self, process):
        """Get detailed progress information for a training process."""
        progress_info = {
            'training': None,
            'videos': None,
            'files': {
                'checkpoints': 0,
                'milestone_videos': 0,
                'hour_videos': 0,
                'output_dir': 'N/A'
            }
        }

        try:
            # Get the actual output paths from the process manager
            output_paths = self.process_manager.get_process_output_paths(process.id)

            if output_paths:
                # Use the actual video output path
                video_output_dir = Path(output_paths.get('videos_base', f"outputs/{process.id}"))
                models_dir = Path(output_paths.get('models', f"models/checkpoints/{process.id}"))

                progress_info['files']['output_dir'] = str(video_output_dir)

                # Count milestone videos
                milestones_dir = video_output_dir / "milestones"
                if milestones_dir.exists():
                    milestone_videos = list(milestones_dir.glob("*.mp4"))
                    progress_info['files']['milestone_videos'] = len(milestone_videos)

                # Count hour videos (look for hour_*.mp4 pattern)
                parts_dir = video_output_dir / "parts"
                if parts_dir.exists():
                    hour_videos = list(parts_dir.glob("*.mp4"))
                    progress_info['files']['hour_videos'] = len(hour_videos)
                else:
                    # Fallback: look in main directory
                    hour_videos = list(video_output_dir.glob("hour_*.mp4"))
                    progress_info['files']['hour_videos'] = len(hour_videos)
            else:
                # Fallback to default paths
                output_dir = self.project_root / "outputs" / process.id
                models_dir = self.project_root / "models" / "checkpoints" / process.id

                if output_dir.exists():
                    progress_info['files']['output_dir'] = str(output_dir)

                    # Count milestone videos
                    milestones_dir = output_dir / "milestones"
                    if milestones_dir.exists():
                        milestone_videos = list(milestones_dir.glob("*.mp4"))
                        progress_info['files']['milestone_videos'] = len(milestone_videos)

                    # Count hour videos (look for hour_*.mp4 pattern)
                    hour_videos = list(output_dir.glob("hour_*.mp4"))
                    progress_info['files']['hour_videos'] = len(hour_videos)

            # Count model checkpoints
            if models_dir.exists():
                checkpoints = list(models_dir.glob("*.zip"))
                progress_info['files']['checkpoints'] = len(checkpoints)

            # Parse actual training progress from logs
            training_progress = self._parse_training_logs(process)
            progress_info['training'] = training_progress

            # Debug: Log progress data
            if training_progress['current_steps'] > 0:
                print(f"📊 Progress for {process.id}: {training_progress['current_steps']:,}/{training_progress['total_steps']:,} ({training_progress['progress_pct']:.1f}%)")

            # Video progress based on milestones
            total_milestones = 10  # From your config: 10%, 20%, ..., 100%
            target_hours = 10  # From your config

            progress_info['videos'] = {
                'milestones_completed': progress_info['files']['milestone_videos'],
                'total_milestones': total_milestones,
                'hour_videos_completed': progress_info['files']['hour_videos'],
                'target_hours': target_hours,
                'total_duration': f"{progress_info['files']['hour_videos']} hours"
            }

        except Exception as e:
            print(f"Error getting progress for {process.id}: {e}")

        return progress_info

    def _parse_training_logs(self, process):
        """Parse training progress from process logs."""
        training_info = {
            'current_steps': 0,
            'total_steps': 4000000,  # Default
            'progress_pct': 0.0,
            'eta': 'Unknown',
            'checkpoints_saved': 0,
            'current_reward': 'N/A',
            'fps': 'N/A'
        }

        try:
            # Get config data for total steps
            if hasattr(process, 'config_data') and process.config_data:
                config = process.config_data
                training_info['total_steps'] = config.get('train', {}).get('total_timesteps', 4000000)

            # Parse recent logs for current progress
            recent_logs = self.process_manager.get_recent_logs(process.id)
            if recent_logs:
                # Debug: Print recent logs to see what we're parsing (only if we have new data)
                if len(recent_logs) > 1000:  # Only debug if we have substantial logs
                    print(f"🔍 Recent logs for {process.id} ({len(recent_logs)} chars)")
                    # Look for key training indicators in recent logs
                    if '[Stats] Progress:' in recent_logs[-500:]:
                        print(f"   ✅ Found progress data in recent logs")
                lines = recent_logs.split('\n')

                # Look for training progress indicators
                import re
                for line in reversed(lines[-50:]):  # Check last 50 lines
                    line = line.strip()

                    # Look for TrainingProgressCallback output: [Stats] Progress: 10,000/4,000,000 (0.3%) | Speed: 1000 steps/s | ETA: 1.1h
                    if '[Stats] Progress:' in line:
                        # Extract current/total steps and percentage
                        progress_match = re.search(r'Progress:\s*([\d,]+)/([\d,]+)\s*\(([\d.]+)%\)', line)
                        if progress_match:
                            current_steps = int(progress_match.group(1).replace(',', ''))
                            total_steps = int(progress_match.group(2).replace(',', ''))
                            progress_pct = float(progress_match.group(3))

                            training_info['current_steps'] = current_steps
                            training_info['total_steps'] = total_steps
                            training_info['progress_pct'] = progress_pct

                        # Extract speed (steps/s)
                        speed_match = re.search(r'Speed:\s*([\d.]+)\s*steps/s', line)
                        if speed_match:
                            training_info['fps'] = f"{float(speed_match.group(1)):.0f}"

                        # Extract ETA
                        eta_match = re.search(r'ETA:\s*([\d.]+)h', line)
                        if eta_match:
                            eta_hours = float(eta_match.group(1))
                            if eta_hours < 1:
                                training_info['eta'] = f"{eta_hours * 60:.0f}m"
                            else:
                                training_info['eta'] = f"{eta_hours:.1f}h"

                        # This is the most recent progress line, so break after processing
                        break

                    # Look for milestone information: [Milestone] Milestone reached: 10% at step 400,000
                    elif '[Milestone]' in line:
                        milestone_match = re.search(r'Milestone reached:\s*([\d.]+)%\s*at step\s*([\d,]+)', line)
                        if milestone_match:
                            milestone_pct = float(milestone_match.group(1))
                            milestone_step = int(milestone_match.group(2).replace(',', ''))
                            # Update current steps if this is more recent
                            if milestone_step > training_info['current_steps']:
                                training_info['current_steps'] = milestone_step
                                training_info['progress_pct'] = milestone_pct

                    # Look for training start: [Start] Training started - Target: 4,000,000 timesteps
                    elif '[Start] Training started' in line:
                        target_match = re.search(r'Target:\s*([\d,]+)\s*timesteps', line)
                        if target_match:
                            training_info['total_steps'] = int(target_match.group(1).replace(',', ''))

                    # Look for reward information from stable-baselines3 logs
                    elif 'ep_rew_mean' in line.lower():
                        reward_match = re.search(r'ep_rew_mean[:\s|]*([+-]?\d*\.?\d+)', line, re.IGNORECASE)
                        if reward_match:
                            training_info['current_reward'] = f"{float(reward_match.group(1)):.2f}"

                # Get leg information from database to calculate proper progress
                try:
                    run_id = process.id
                    leg_start_timestep = 0
                    leg_number = 1

                    # Query database for leg information
                    if hasattr(self, 'ml_database'):
                        conn = self.ml_database.connection
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT leg_start_timestep, leg_number FROM experiment_runs WHERE run_id = ?",
                            (run_id,)
                        )
                        row = cursor.fetchone()
                        if row:
                            leg_start_timestep = row[0] if row[0] else 0
                            leg_number = row[1] if row[1] else 1

                        # Always store leg number for display
                        training_info['leg_number'] = leg_number
                        training_info['leg_start_timestep'] = leg_start_timestep

                    # Calculate leg-specific progress
                    # Leg progress = (current - leg_start) / total * 100
                    if training_info['current_steps'] >= leg_start_timestep and training_info['total_steps'] > 0:
                        leg_steps_completed = training_info['current_steps'] - leg_start_timestep
                        leg_total_steps = training_info['total_steps']

                        if leg_total_steps > 0:
                            leg_progress_pct = (leg_steps_completed / leg_total_steps) * 100
                            # Cap at 100%
                            training_info['progress_pct'] = min(100.0, leg_progress_pct)

                        # Store additional info for display
                        training_info['leg_steps_completed'] = leg_steps_completed
                        training_info['leg_total_steps'] = leg_total_steps
                        training_info['total_timesteps_all_legs'] = training_info['current_steps']
                    elif training_info['progress_pct'] == 0.0:
                        # Fallback if we can't calculate leg progress
                        if training_info['current_steps'] > 0 and training_info['total_steps'] > 0:
                            training_info['progress_pct'] = min(100.0, (training_info['current_steps'] / training_info['total_steps']) * 100)
                except Exception as e:
                    print(f"Error calculating leg progress: {e}")
                    # Fallback to simple calculation
                    if training_info['progress_pct'] == 0.0:
                        if training_info['current_steps'] > 0 and training_info['total_steps'] > 0:
                            training_info['progress_pct'] = min(100.0, (training_info['current_steps'] / training_info['total_steps']) * 100)

        except Exception as e:
            print(f"Error parsing training logs for {process.id}: {e}")

        return training_info

    def _schedule_progress_refresh(self):
        """Schedule automatic progress refresh."""
        # Refresh progress tracking every 10 seconds
        self.root.after(10000, self._auto_refresh_progress)

    def _auto_refresh_progress(self):
        """Auto-refresh progress and schedule next refresh."""
        try:
            # Only refresh if we're on the progress tab
            current_tab = self.tabview.get()
            if current_tab == "Progress Tracking":
                self._refresh_progress()
        except Exception:
            pass  # Don't let refresh errors break the app

        # Schedule next refresh
        self._schedule_progress_refresh()

    def _schedule_process_refresh(self):
        """Schedule automatic process refresh."""
        # Refresh processes every 15 seconds
        self.root.after(15000, self._auto_refresh_processes)

    def _auto_refresh_processes(self):
        """Auto-refresh processes and schedule next refresh."""
        try:
            # Only refresh if we're on the processes tab
            current_tab = self.tabview.get()
            if current_tab == "Training Processes":
                self._refresh_processes()
        except Exception:
            pass  # Don't let refresh errors break the app

        # Schedule next refresh
        self._schedule_process_refresh()

    def _refresh_videos(self):
        """Refresh the video gallery with all available videos."""
        # Clear existing items
        for item in self.video_tree.get_children():
            self.video_tree.delete(item)

        # Get all video files from all training runs
        videos = self._discover_videos()

        # Apply current filter
        filtered_videos = self._apply_video_filter(videos)

        # Add videos to tree
        for video in filtered_videos:
            # Check if this is an in-progress video
            is_in_progress = video.get('in_progress', False)

            # Use different tags for in-progress videos
            if is_in_progress:
                item_tags = (video['path'], 'in_progress')
            else:
                item_tags = (video['path'],)

            self.video_tree.insert("", "end", values=(
                video['name'],
                video['type'],
                video['duration'],
                video['size'],
                video['created'],
                video['training_run']
            ), tags=item_tags)

        # Configure tag styling for in-progress videos (grayed out)
        self.video_tree.tag_configure('in_progress', foreground='gray')

    def _schedule_video_refresh(self):
        """Schedule periodic refresh of video gallery when videos are being generated."""
        # Cancel existing timer if any
        if hasattr(self, '_video_refresh_timer') and self._video_refresh_timer:
            try:
                self.root.after_cancel(self._video_refresh_timer)
            except:
                pass

        # Check if there are any in-progress videos
        try:
            import sqlite3
            conn = sqlite3.connect('ml_experiments.db')
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM video_generation_progress WHERE status = 'in-progress'")
            in_progress_count = cursor.fetchone()[0]
            conn.close()

            # If there are in-progress videos, refresh every 5 seconds
            if in_progress_count > 0:
                self._video_refresh_timer = self.root.after(5000, self._auto_refresh_videos)
            else:
                # Otherwise, check again in 30 seconds
                self._video_refresh_timer = self.root.after(30000, self._schedule_video_refresh)
        except Exception as e:
            print(f"Error checking for in-progress videos: {e}")
            # Try again in 30 seconds
            self._video_refresh_timer = self.root.after(30000, self._schedule_video_refresh)

    def _auto_refresh_videos(self):
        """Auto-refresh videos and reschedule."""
        try:
            self._refresh_videos()
        except Exception as e:
            print(f"Error auto-refreshing videos: {e}")

        # Schedule next refresh
        self._schedule_video_refresh()

    def _discover_videos(self):
        """Discover all video files from training runs."""
        videos = []

        try:
            # Get all processes to find their output paths
            processes = self.process_manager.get_processes()
            self._append_log(f"🔍 Scanning videos for {len(processes)} training processes...")

            # Check each process's output directory
            for process in processes:
                output_paths = self.process_manager.get_process_output_paths(process.id)
                if output_paths:
                    self._append_log(f"  📂 Checking process {process.name} (ID: {process.id[:8]}...)")
                    self._append_log(f"     Milestones: {output_paths.get('videos_milestones', 'N/A')}")
                    self._append_log(f"     Eval: {output_paths.get('videos_eval', 'N/A')}")
                    self._append_log(f"     Parts: {output_paths.get('videos_parts', 'N/A')}")
                    process_videos = self._scan_process_videos(process, output_paths)
                    if process_videos:
                        self._append_log(f"     ✅ Found {len(process_videos)} video(s)")
                    videos.extend(process_videos)

            # Also check default outputs directory
            default_outputs = self.project_root / "outputs"
            if default_outputs.exists():
                self._append_log(f"🔍 Scanning default outputs directory: {default_outputs}")
                for run_dir in default_outputs.iterdir():
                    if run_dir.is_dir():
                        dir_videos = self._scan_directory_videos(run_dir, run_dir.name)
                        if dir_videos:
                            self._append_log(f"  ✅ Found {len(dir_videos)} video(s) in {run_dir.name}")
                        videos.extend(dir_videos)

            # Check post-processed videos directory (time-lapses, progressions, comparisons)
            video_output_dir = self.project_root / "video" / "output"
            if video_output_dir.exists():
                self._append_log(f"🔍 Scanning post-processed videos directory: {video_output_dir}")
                # Videos directly under video/output
                output_videos = self._scan_directory_videos(video_output_dir, "Post-Processed")
                if output_videos:
                    self._append_log(f"  ✅ Found {len(output_videos)} post-processed video(s)")
                videos.extend(output_videos)

                # Videos inside per-run subdirectories (e.g., video/output/run-xxxx)
                for subdir in video_output_dir.iterdir():
                    if subdir.is_dir():
                        sub_videos = self._scan_directory_videos(subdir, subdir.name)
                        if sub_videos:
                            self._append_log(f"  ✅ Found {len(sub_videos)} post-processed video(s) in {subdir.name}")
                        videos.extend(sub_videos)

            # NEW: Check database for video_path entries (for custom output locations)
            if self.ml_database:
                self._append_log(f"🔍 Checking database for video paths...")
                db_videos = self._scan_database_videos()
                if db_videos:
                    self._append_log(f"  ✅ Found {len(db_videos)} video(s) from database")
                videos.extend(db_videos)

            # NEW: Check for in-progress video generation
            self._append_log(f"🔍 Checking for in-progress video generation...")
            in_progress_videos = self._scan_in_progress_videos()
            if in_progress_videos:
                self._append_log(f"  ⏳ Found {len(in_progress_videos)} video(s) being generated")
            videos.extend(in_progress_videos)

            self._append_log(f"✅ Video discovery complete: Found {len(videos)} total video(s)")

        except Exception as e:
            error_msg = f"Error discovering videos: {e}"
            print(error_msg)
            self._append_log(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()

        return videos

    def _scan_database_videos(self):
        """Scan for videos listed in the database (for custom output paths)."""
        videos = []

        try:
            import sqlite3
            conn = sqlite3.connect('ml_experiments.db')
            cursor = conn.cursor()

            # Get all runs with video_path set
            cursor.execute('SELECT run_id, video_path FROM experiment_runs WHERE video_path IS NOT NULL AND video_path != ""')
            rows = cursor.fetchall()

            for run_id, video_path in rows:
                if video_path and Path(video_path).exists():
                    # Check if this is a single video file or a directory
                    video_path_obj = Path(video_path)

                    if video_path_obj.is_file():
                        # Single video file
                        try:
                            stat = video_path_obj.stat()
                            size_mb = stat.st_size / (1024 * 1024)
                            created = datetime.fromtimestamp(stat.st_mtime)
                            duration = self._get_video_duration(video_path_obj)

                            videos.append({
                                'name': video_path_obj.name,
                                'path': str(video_path_obj),
                                'type': "Training",
                                'duration': duration,
                                'size': f"{size_mb:.1f} MB",
                                'created': created.strftime("%Y-%m-%d %H:%M"),
                                'training_run': run_id
                            })
                        except Exception as e:
                            print(f"Error processing database video {video_path}: {e}")

                    elif video_path_obj.is_dir():
                        # Directory containing videos - scan it
                        dir_videos = self._scan_directory_videos(video_path_obj, run_id)
                        videos.extend(dir_videos)

                    else:
                        # Path might be a parent directory - check parent
                        parent_dir = video_path_obj.parent
                        if parent_dir.exists() and parent_dir.is_dir():
                            dir_videos = self._scan_directory_videos(parent_dir, run_id)
                            videos.extend(dir_videos)

            conn.close()

        except Exception as e:
            print(f"Error scanning database videos: {e}")
            import traceback
            traceback.print_exc()

        return videos

    def _scan_in_progress_videos(self):
        """Scan for videos currently being generated from the database."""
        videos = []

        try:
            import sqlite3
            conn = sqlite3.connect('ml_experiments.db')
            cursor = conn.cursor()

            # Get all in-progress video generation entries
            cursor.execute("""
                SELECT video_id, run_id, video_name, video_path,
                       progress_percentage, estimated_seconds_remaining,
                       processed_frames, total_frames, started_at
                FROM video_generation_progress
                WHERE status = 'in-progress'
                ORDER BY started_at DESC
            """)
            rows = cursor.fetchall()

            for row in rows:
                video_id, run_id, video_name, video_path, progress_pct, eta_seconds, processed_frames, total_frames, started_at = row

                # Format progress information
                progress_str = f"{progress_pct:.1f}%" if progress_pct else "0%"

                # Format ETA
                if eta_seconds and eta_seconds > 0:
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds}s remaining"
                    elif eta_seconds < 3600:
                        eta_str = f"{eta_seconds // 60}m {eta_seconds % 60}s remaining"
                    else:
                        hours = eta_seconds // 3600
                        minutes = (eta_seconds % 3600) // 60
                        eta_str = f"{hours}h {minutes}m remaining"
                else:
                    eta_str = "Calculating..."

                # Format duration as progress indicator
                if total_frames and processed_frames:
                    duration_str = f"{progress_str} ({processed_frames}/{total_frames} frames)"
                else:
                    duration_str = progress_str

                # Format size as ETA
                size_str = eta_str

                # Format created time
                try:
                    from datetime import datetime
                    created_dt = datetime.fromisoformat(started_at)
                    created_str = created_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    created_str = "In Progress"

                videos.append({
                    'name': f"🎬 {video_name}",  # Add emoji to indicate in-progress
                    'path': video_path if video_path else f"generating_{video_id}",
                    'type': "⏳ Generating",
                    'duration': duration_str,
                    'size': size_str,
                    'created': created_str,
                    'training_run': run_id,
                    'in_progress': True,  # Flag to disable interaction
                    'progress_pct': progress_pct,
                    'video_id': video_id
                })

            conn.close()

        except Exception as e:
            print(f"Error scanning in-progress videos: {e}")
            import traceback
            traceback.print_exc()

        return videos

    def _scan_process_videos(self, process, output_paths):
        """Scan videos for a specific process."""
        videos = []

        try:
            # Scan milestone videos
            milestones_path = output_paths.get('videos_milestones')
            if milestones_path:
                milestone_path_obj = Path(milestones_path)
                if milestone_path_obj.exists():
                    milestone_videos = self._scan_video_directory(
                        milestone_path_obj, "Milestone", process.name
                    )
                    videos.extend(milestone_videos)
                else:
                    self._append_log(f"     ⚠️ Milestone directory does not exist: {milestones_path}")

            # Scan evaluation videos
            eval_path = output_paths.get('videos_eval')
            if eval_path:
                eval_path_obj = Path(eval_path)
                if eval_path_obj.exists():
                    eval_videos = self._scan_video_directory(
                        eval_path_obj, "Evaluation", process.name
                    )
                    videos.extend(eval_videos)
                else:
                    self._append_log(f"     ⚠️ Eval directory does not exist: {eval_path}")

            # Scan hour/part videos
            parts_path = output_paths.get('videos_parts')
            if parts_path:
                parts_path_obj = Path(parts_path)
                if parts_path_obj.exists():
                    part_videos = self._scan_video_directory(
                        parts_path_obj, "Hour", process.name
                    )
                    videos.extend(part_videos)
                else:
                    self._append_log(f"     ⚠️ Parts directory does not exist: {parts_path}")

        except Exception as e:
            error_msg = f"Error scanning process videos for {process.id}: {e}"
            print(error_msg)
            self._append_log(f"     ❌ {error_msg}")
            import traceback
            traceback.print_exc()

        return videos

    def _scan_directory_videos(self, directory, training_run):
        """Scan videos in a directory structure."""
        videos = []

        try:
            # Check for milestone videos
            milestones_dir = directory / "milestones"
            if milestones_dir.exists():
                videos.extend(self._scan_video_directory(milestones_dir, "Milestone", training_run))

            # Check for evaluation videos
            eval_dir = directory / "eval"
            if eval_dir.exists():
                videos.extend(self._scan_video_directory(eval_dir, "Evaluation", training_run))

            # Check for hour/part videos
            parts_dir = directory / "parts"
            if parts_dir.exists():
                videos.extend(self._scan_video_directory(parts_dir, "Hour", training_run))

            # Check for videos directly in the directory
            videos.extend(self._scan_video_directory(directory, "Other", training_run))

        except Exception as e:
            print(f"Error scanning directory {directory}: {e}")

        return videos

    def _scan_video_directory(self, directory, video_type, training_run):
        """Scan a specific directory for video files."""
        videos = []

        try:
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

            for file_path in directory.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                    try:
                        # Get file info
                        stat = file_path.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        created = datetime.fromtimestamp(stat.st_mtime)

                        # Try to get video duration (basic approach)
                        duration = self._get_video_duration(file_path)

                        # Try to derive run id from filename (e.g., run-xxxx_*)
                        run_label = training_run
                        m = re.match(r'(run-[A-Za-z0-9]+)', file_path.stem)
                        if m:
                            run_label = m.group(1)
                        elif file_path.parent.name.startswith("run-"):
                            run_label = file_path.parent.name

                        # Detect post-processed video type from filename
                        detected_type = video_type
                        filename_lower = file_path.name.lower()
                        if 'timelapse' in filename_lower or '_timelapse_' in filename_lower:
                            detected_type = "Time-Lapse"
                        elif 'progression' in filename_lower or '_progression_' in filename_lower:
                            detected_type = "Progression"
                        elif 'comparison' in filename_lower or '_comparison_' in filename_lower:
                            detected_type = "Comparison"
                        elif 'training' in filename_lower and video_type == "Other":
                            detected_type = "Training"

                        videos.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'type': detected_type,
                            'duration': duration,
                            'size': f"{size_mb:.1f} MB",
                            'created': created.strftime("%Y-%m-%d %H:%M"),
                            'training_run': run_label
                        })

                    except Exception as e:
                        print(f"Error processing video {file_path}: {e}")

        except Exception as e:
            print(f"Error scanning video directory {directory}: {e}")

        return videos

    def _get_video_duration(self, video_path):
        """Get video duration (basic implementation)."""
        try:
            # This is a simple approach - in production you might use ffprobe or similar
            # For now, we'll estimate based on file size (very rough)
            stat = video_path.stat()
            size_mb = stat.st_size / (1024 * 1024)

            # Very rough estimate: ~1MB per minute for compressed video
            estimated_minutes = int(size_mb)
            if estimated_minutes < 1:
                return "< 1 min"
            elif estimated_minutes < 60:
                return f"{estimated_minutes} min"
            else:
                hours = estimated_minutes // 60
                minutes = estimated_minutes % 60
                return f"{hours}h {minutes}m"

        except Exception:
            return "Unknown"

    def _apply_video_filter(self, videos):
        """Apply the current video filter."""
        filter_value = self.video_filter_var.get()

        if filter_value == "All Videos":
            return videos
        elif filter_value == "Milestone Videos":
            return [v for v in videos if v['type'] == "Milestone"]
        elif filter_value == "Hour Videos":
            return [v for v in videos if v['type'] == "Hour"]
        elif filter_value == "Evaluation Videos":
            return [v for v in videos if v['type'] == "Evaluation"]
        else:
            return videos

    def _filter_videos(self, filter_value):
        """Handle video filter change."""
        self._refresh_videos()

    def _is_video_in_progress(self, item):
        """Check if a video tree item is in-progress."""
        tags = self.video_tree.item(item)["tags"]
        return 'in_progress' in tags

    def _play_selected_video(self):
        """Play the selected video in the default video player."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to play.")
            return

        item = selection[0]

        # Check if video is in-progress
        if self._is_video_in_progress(item):
            messagebox.showinfo("Video In Progress",
                              "This video is currently being generated. Please wait until it completes.")
            return

        video_path = self.video_tree.item(item)["tags"][0]

        try:
            # Open video with default system player
            import os
            import platform

            if platform.system() == "Windows":
                os.startfile(video_path)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{video_path}'")
            else:  # Linux
                os.system(f"xdg-open '{video_path}'")

            self._append_log(f"🎬 Opened video: {Path(video_path).name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video: {e}")
            self._append_log(f"❌ Failed to open video: {e}")

    def _open_video_player(self):
        """Open the enhanced video player dialog."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to open in the video player.")
            return

        item = selection[0]

        # Check if video is in-progress
        if self._is_video_in_progress(item):
            messagebox.showinfo("Video In Progress",
                              "This video is currently being generated. Please wait until it completes.")
            return

        video_path = self.video_tree.item(item)["tags"][0]
        values = self.video_tree.item(item)["values"]

        # Create video info dictionary
        video_info = {
            'name': values[0],
            'type': values[1],
            'duration': values[2],
            'size': values[3],
            'created': values[4],
            'training_run': values[5],
            'path': video_path
        }

        try:
            # Open the enhanced video player dialog
            player_dialog = VideoPlayerDialog(self.root, video_path, video_info)
            player_dialog.show()

            self._append_log(f"🎬 Opened video player for: {values[0]}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video player: {e}")
            self._append_log(f"❌ Failed to open video player: {e}")

    def _preview_selected_video(self):
        """Show a quick preview/thumbnail of the selected video."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to preview.")
            return

        item = selection[0]
        video_path = self.video_tree.item(item)["tags"][0]
        video_name = Path(video_path).name

        # For now, just show video info - could be enhanced with actual thumbnail
        messagebox.showinfo("Video Preview",
                           f"Video: {video_name}\n\n"
                           f"This would show a thumbnail preview.\n"
                           f"Click 'Play Video' to watch the full video.")

    def _show_video_info(self):
        """Show detailed information about the selected video."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to view info.")
            return

        item = selection[0]
        video_path = self.video_tree.item(item)["tags"][0]

        try:
            file_path = Path(video_path)
            stat = file_path.stat()

            # Get detailed file information
            size_bytes = stat.st_size
            size_mb = size_bytes / (1024 * 1024)
            size_gb = size_mb / 1024

            created = datetime.fromtimestamp(stat.st_ctime)
            modified = datetime.fromtimestamp(stat.st_mtime)

            # Get video details from tree
            values = self.video_tree.item(item)["values"]

            info_text = (
                f"📹 Video Information\n\n"
                f"Name: {values[0]}\n"
                f"Type: {values[1]}\n"
                f"Duration: {values[2]}\n"
                f"Training Run: {values[5]}\n\n"
                f"📁 File Details\n"
                f"Path: {video_path}\n"
                f"Size: {size_mb:.1f} MB ({size_bytes:,} bytes)\n"
                f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"🎬 Actions Available:\n"
                f"• Play Video - Open in default player\n"
                f"• Quick Preview - Show thumbnail (coming soon)\n"
                f"• Delete Video - Remove from disk"
            )

            messagebox.showinfo("Video Information", info_text)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to get video info: {e}")

    def _delete_selected_video(self):
        """Delete the selected video file."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to delete.")
            return

        item = selection[0]

        # Check if video is in-progress
        if self._is_video_in_progress(item):
            messagebox.showinfo("Video In Progress",
                              "Cannot delete a video that is currently being generated.")
            return

        video_path = self.video_tree.item(item)["tags"][0]
        video_name = Path(video_path).name

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this video?\n\n"
            f"Video: {video_name}\n"
            f"Path: {video_path}\n\n"
            f"⚠️ This action cannot be undone!"
        )

        if confirm:
            try:
                Path(video_path).unlink()
                self._refresh_videos()
                self._append_log(f"🗑️ Deleted video: {video_name}")
                messagebox.showinfo("Success", f"Video deleted successfully: {video_name}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete video: {e}")
                self._append_log(f"❌ Failed to delete video: {e}")

    def _open_video_folder(self):
        """Open the video output folder in file explorer."""
        try:
            # Try to find a video folder to open
            processes = self.process_manager.get_processes()

            folder_to_open = None

            # Try to get folder from most recent process
            if processes:
                latest_process = max(processes, key=lambda p: p.created)
                output_paths = self.process_manager.get_process_output_paths(latest_process.id)

                if output_paths:
                    # Try milestone folder first
                    milestones_path = output_paths.get('videos_milestones')
                    if milestones_path and Path(milestones_path).exists():
                        folder_to_open = str(Path(milestones_path).parent)
                    else:
                        # Try base video folder
                        videos_base = output_paths.get('videos_base')
                        if videos_base and Path(videos_base).exists():
                            folder_to_open = videos_base

            # Fallback to default outputs directory
            if not folder_to_open:
                default_outputs = self.project_root / "outputs"
                if default_outputs.exists():
                    folder_to_open = str(default_outputs)

            # Final fallback to project root
            if not folder_to_open:
                folder_to_open = str(self.project_root)

            # Open the folder
            import os
            import platform

            if platform.system() == "Windows":
                os.startfile(folder_to_open)
            elif platform.system() == "Darwin":  # macOS
                os.system(f"open '{folder_to_open}'")
            else:  # Linux
                os.system(f"xdg-open '{folder_to_open}'")

            self._append_log(f"📁 Opened folder: {folder_to_open}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open video folder: {e}")
            self._append_log(f"❌ Failed to open video folder: {e}")

    def _generate_videos_dialog(self):
        """Show dialog to generate videos from completed training runs."""
        try:
            # DB-first: populate from the same accessor the dashboard uses so completed runs
            # are always selectable (even if artifacts are missing).
            db_runs = self.ml_database.get_experiment_runs()
            all_runs = []
            for run in db_runs:
                all_runs.append({
                    'id': run.run_id,  # canonical identifier
                    'name': run.display_name or run.custom_name or run.run_id,
                    'status': run.status,
                    'tracked': False
                })

            if not all_runs:
                messagebox.showinfo("No Training Runs",
                                  "No training runs found in the database.\n\n"
                                  "Start a training session first.")
                return

            # Create dialog
            dialog = tk.Toplevel(self.root)
            dialog.title("Generate Videos from Training")
            dialog.geometry("700x500")
            dialog.transient(self.root)
            dialog.grab_set()

            # Title
            title_label = ctk.CTkLabel(dialog, text="🎥 Generate Videos from Training Checkpoints",
                                      font=ctk.CTkFont(size=16, weight="bold"))
            title_label.pack(pady=15)

            # Info label
            info_label = ctk.CTkLabel(dialog,
                                     text="Select a training run to generate milestone videos from saved checkpoints.\n"
                                          "This creates videos showing agent performance at different training stages.",
                                     font=ctk.CTkFont(size=11))
            info_label.pack(pady=5)

            # Training run selection frame
            selection_frame = ctk.CTkFrame(dialog)
            selection_frame.pack(fill="both", expand=True, padx=20, pady=10)

            # List of training runs
            runs_label = ctk.CTkLabel(selection_frame, text="Select Training Run:",
                                     font=ctk.CTkFont(size=12, weight="bold"))
            runs_label.pack(pady=5)

            # Create listbox for runs
            runs_listbox = tk.Listbox(selection_frame, height=10)
            runs_listbox.pack(fill="both", expand=True, padx=10, pady=5)

            # Populate with training runs
            run_data = []
            for run_info in all_runs:
                # Check for checkpoints in the correct location: models/checkpoints/{run_id}/milestones/
                checkpoint_dir = self.project_root / "models" / "checkpoints" / run_info['id'] / "milestones"
                latest_path = self.project_root / "models" / "checkpoints" / run_info['id'] / "latest.zip"

                # Count checkpoints
                checkpoint_count = 0
                if checkpoint_dir.exists():
                    checkpoint_count = len(list(checkpoint_dir.glob("*.zip")))

                # Read metadata to get target_hours
                metadata_path = self.project_root / "models" / "checkpoints" / run_info['id'] / "run_metadata.json"
                target_hours = None
                if metadata_path.exists():
                    try:
                        import json
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            target_hours = metadata.get('target_hours')
                    except Exception:
                        pass

                # Store target_hours in run_info for later use
                run_info['target_hours'] = target_hours
                run_info['milestone_zip_count'] = checkpoint_count
                run_info['latest_exists'] = latest_path.exists()

                # Status icon
                status = run_info['status']
                if status == "completed":
                    status_icon = "✅"
                elif status == "running":
                    status_icon = "🔄"
                elif status == "unknown":
                    status_icon = "📦"  # Untracked run
                else:
                    status_icon = "⏸️"

                # Add target hours to display if available
                target_info = f" | Target: {target_hours}h" if target_hours else ""
                latest_info = "latest: yes" if run_info['latest_exists'] else "latest: no"
                if checkpoint_count > 0:
                    artifact_info = f"milestones: {checkpoint_count}, {latest_info}"
                else:
                    artifact_info = f"no milestone checkpoints, {latest_info}"
                display_text = f"{status_icon} {run_info['name']} | {run_info['id']} ({artifact_info}{target_info})"
                runs_listbox.insert(tk.END, display_text)
                run_data.append(run_info)

            # Options frame
            options_frame = ctk.CTkFrame(dialog)
            options_frame.pack(fill="x", padx=20, pady=10)

            # Video length option
            clip_label = ctk.CTkLabel(options_frame, text="Total Video Length (seconds):")
            clip_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

            clip_var = tk.StringVar(value="600")  # Default 10 minutes
            clip_entry = ctk.CTkEntry(options_frame, textvariable=clip_var, width=100)
            clip_entry.grid(row=0, column=1, padx=5, pady=5)

            # Info label for target video length
            target_info_label = ctk.CTkLabel(options_frame,
                                            text="Select a run to auto-fill its target video length",
                                            font=ctk.CTkFont(size=10),
                                            text_color="gray")
            target_info_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

            def on_run_selected(event):
                """Update video length when a run is selected."""
                selection = runs_listbox.curselection()
                if selection:
                    selected_run = run_data[selection[0]]
                    target_hours = selected_run.get('target_hours')
                    if target_hours:
                        # Calculate total video seconds from target hours
                        total_seconds = int(target_hours * 3600)
                        clip_var.set(str(total_seconds))

                        # Format display
                        if total_seconds >= 3600:
                            hours = total_seconds / 3600
                            display = f"{hours:.1f}h"
                        elif total_seconds >= 60:
                            minutes = total_seconds / 60
                            display = f"{minutes:.1f}m"
                        else:
                            display = f"{total_seconds}s"

                        target_info_label.configure(
                            text=f"Target: {display} ({total_seconds}s) - will generate one continuous video",
                            text_color="#28a745"
                        )
                    else:
                        target_info_label.configure(
                            text="No target video length saved for this run (using manual input)",
                            text_color="gray"
                        )

            runs_listbox.bind('<<ListboxSelect>>', on_run_selected)

            # Buttons frame
            buttons_frame = ctk.CTkFrame(dialog)
            buttons_frame.pack(fill="x", padx=20, pady=10)

            def generate_videos():
                """Generate videos for selected run."""
                selection = runs_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a training run first.")
                    return

                selected_run = run_data[selection[0]]
                run_id = selected_run['id']
                run_name = selected_run['name']
                milestone_zip_count = int(selected_run.get('milestone_zip_count') or 0)
                latest_exists = bool(selected_run.get('latest_exists'))

                try:
                    total_seconds = int(clip_var.get())
                    if total_seconds <= 0:
                        raise ValueError("Video length must be positive")
                except ValueError as e:
                    messagebox.showerror("Invalid Input", f"Invalid video length: {e}")
                    return

                # Do not silently omit runs: guard here with a clear reason.
                # Note: post_training_video_generator.py does NOT currently match "latest.zip",
                # so milestone zips are required for this UI path.
                if milestone_zip_count <= 0:
                    msg = (
                        f"Cannot generate videos for {run_id}.\n\n"
                        f"Missing required artifact: milestone checkpoints (*.zip)\n"
                        f"Expected at: {self.project_root / 'models' / 'checkpoints' / run_id / 'milestones'}\n\n"
                    )
                    if latest_exists:
                        msg += "Note: latest.zip exists, but the current post-training generator does not support it.\n"
                    messagebox.showwarning("Missing Checkpoints", msg)
                    return

                # The checkpoints are in models/checkpoints/{run_id}/milestones/
                checkpoint_base_dir = self.project_root / "models" / "checkpoints" / run_id / "milestones"

                # Video output directory
                video_output_dir = self.project_root / "outputs" / run_id / "milestones"

                if not checkpoint_base_dir.exists():
                    messagebox.showerror("Error",
                                       f"Checkpoint directory not found: {checkpoint_base_dir}\n\n"
                                       f"Make sure this training run has saved checkpoints.")
                    return

                # Create output directory if it doesn't exist
                video_output_dir.mkdir(parents=True, exist_ok=True)

                # Close dialog
                dialog.destroy()

                # Format display
                if total_seconds >= 3600:
                    hours = total_seconds / 3600
                    display = f"{hours:.1f}h"
                elif total_seconds >= 60:
                    minutes = total_seconds / 60
                    display = f"{minutes:.1f}m"
                else:
                    display = f"{total_seconds}s"

                # Show progress
                self._append_log(f"🎥 Generating continuous video for {run_name}...")
                self._append_log(f"   Checkpoint directory: {checkpoint_base_dir}")
                self._append_log(f"   Output directory: {video_output_dir}")
                self._append_log(f"   Total video length: {display} ({total_seconds}s)")

                # Import and call the video generator
                from tools.retro_ml_desktop.process_manager import generate_post_training_videos

                # Get config path from process (if tracked)
                config_path = None
                if selected_run['tracked']:
                    config_path = self.process_manager._temp_configs.get(run_id)

                if not config_path or not Path(config_path).exists():
                    # Fallback to default config
                    config_path = str(self.project_root / "conf" / "config.yaml")
                    self._append_log(f"   Using default config: {config_path}")

                # Generate videos in a separate thread to avoid blocking UI
                def generate_thread():
                    try:
                        success = generate_post_training_videos(
                            config_path=config_path,
                            model_dir=str(checkpoint_base_dir),
                            output_dir=str(video_output_dir),
                            clip_seconds=90,  # Not used when total_seconds is provided
                            total_seconds=total_seconds,
                            verbose=2,  # Increased verbosity for debugging
                            db=self.ml_database,  # Pass database for progress tracking
                            run_id=run_id  # Pass run_id for progress tracking
                        )

                        # Update UI on main thread
                        self.root.after(0, lambda: self._on_video_generation_complete(success, run_name))
                    except Exception as e:
                        error_msg = f"Exception during video generation: {e}"
                        self._append_log(f"❌ {error_msg}")
                        import traceback
                        traceback.print_exc()
                        self.root.after(0, lambda: self._on_video_generation_complete(False, run_name))

                import threading
                thread = threading.Thread(target=generate_thread, daemon=True)
                thread.start()

            generate_btn = ctk.CTkButton(buttons_frame, text="🎬 Generate Videos",
                                        command=generate_videos,
                                        **Theme.get_button_colors("success"))
            generate_btn.pack(side="left", padx=5, pady=5)

            cancel_btn = ctk.CTkButton(buttons_frame, text="❌ Cancel",
                                      command=dialog.destroy)
            cancel_btn.pack(side="right", padx=5, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show video generation dialog: {e}")
            self._append_log(f"❌ Error showing video generation dialog: {e}")
            import traceback
            traceback.print_exc()

    def _on_video_generation_complete(self, success: bool, run_name: str):
        """Called when video generation completes."""
        if success:
            self._append_log(f"✅ Video generation complete for {run_name}!")
            self._append_log(f"   Refreshing video gallery...")
            self._refresh_videos()
            messagebox.showinfo("Success", f"Videos generated successfully for {run_name}!\n\nCheck the Video Gallery.")
        else:
            self._append_log(f"❌ Video generation failed for {run_name}")
            messagebox.showerror("Error", f"Failed to generate videos for {run_name}.\nCheck the logs for details.")

    def _create_timelapse_dialog(self):
        """Show dialog to create time-lapse video from training recordings."""
        try:
            from conf.config import load_config
            from training.video_post_processor import VideoPostProcessor

            # Load config
            config = load_config(str(self.project_root / "conf" / "config.yaml"))
            processor = VideoPostProcessor(config)

            # DB-first: show all runs (including completed) even if recordings are missing.
            db_runs = self.ml_database.get_experiment_runs()
            if not db_runs:
                messagebox.showwarning("No Training Runs",
                                     "No training runs found in the database.\n\n"
                                     "Start a training session first.")
                return

            # Training-time recordings root (may not exist in checkpoint-only mode)
            training_dir = Path(config.get('paths', {}).get('videos_training', 'video/training'))

            # Create dialog
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("Create Time-Lapse Video")
            dialog.geometry("650x600")  # Increased width and height
            dialog.transient(self.root)
            dialog.grab_set()

            # Create scrollable frame
            main_container = ctk.CTkScrollableFrame(dialog)
            main_container.pack(fill="both", expand=True, padx=10, pady=10)

            # Title
            title_label = ctk.CTkLabel(main_container, text="Create Time-Lapse from Training Videos",
                                       font=ctk.CTkFont(size=16, weight="bold"))
            title_label.pack(pady=10)

            # Info
            info_label = ctk.CTkLabel(main_container,
                                     text="Create a sped-up time-lapse video from training recordings.",
                                     font=ctk.CTkFont(size=11))
            info_label.pack(pady=5)

            # Run selection
            selection_frame = ctk.CTkFrame(main_container)
            selection_frame.pack(fill="x", padx=10, pady=10)

            runs_label = ctk.CTkLabel(selection_frame, text="Select Training Run:",
                                     font=ctk.CTkFont(size=12, weight="bold"))
            runs_label.pack(pady=5)

            runs_listbox = tk.Listbox(selection_frame, height=6)  # Reduced from 8 to 6
            runs_listbox.pack(fill="x", padx=10, pady=5)

            # Populate runs with custom names from database
            run_data = []
            for run in db_runs:
                run_id = run.run_id
                display_name = run.display_name or run.custom_name or run_id
                run_dir = training_dir / run_id
                video_count = 0
                if run_dir.exists():
                    video_count = len(list(run_dir.glob("*.mp4")))

                suffix = f"(recordings: {video_count})" if video_count > 0 else "(no training recordings)"
                run_data.append({'id': run_id, 'dir': run_dir, 'video_count': video_count, 'display_name': display_name})
                runs_listbox.insert(tk.END, f"{display_name} | {run_id} {suffix}")

            # Speed settings
            speed_frame = ctk.CTkFrame(main_container)
            speed_frame.pack(fill="x", padx=10, pady=10)

            speed_label = ctk.CTkLabel(speed_frame, text="Speed Multiplier:",
                                      font=ctk.CTkFont(size=12, weight="bold"))
            speed_label.pack(pady=5)

            speed_var = tk.StringVar(value="10")
            speed_entry = ctk.CTkEntry(speed_frame, textvariable=speed_var, width=100)
            speed_entry.pack(pady=5)

            speed_info = ctk.CTkLabel(speed_frame, text="(e.g., 10 = 10x faster)",
                                     font=ctk.CTkFont(size=10), text_color="gray")
            speed_info.pack()

            # Add overlays checkbox
            options_frame = ctk.CTkFrame(main_container)
            options_frame.pack(fill="x", padx=10, pady=10)

            ctk.CTkLabel(options_frame, text="Options:", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=5)

            overlays_var = tk.BooleanVar(value=False)
            overlays_checkbox = ctk.CTkCheckBox(
                options_frame,
                text="Include neural network overlays (shows AI learning)",
                variable=overlays_var
            )
            overlays_checkbox.pack(pady=5)

            ctk.CTkLabel(
                options_frame,
                text="Overlays show neural network visualization and stats.",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            ).pack(pady=5)

            # Buttons
            buttons_frame = ctk.CTkFrame(dialog)
            buttons_frame.pack(fill="x", padx=20, pady=10)

            def create_timelapse():
                selection = runs_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a training run first.")
                    return

                selected_run = run_data[selection[0]]
                run_id = selected_run['id']
                video_count = int(selected_run.get('video_count') or 0)

                if video_count <= 0:
                    messagebox.showwarning(
                        "No Training Recordings",
                        f"Run {run_id} has no training recordings.\n\n"
                        f"Expected: {training_dir / run_id}/*.mp4\n\n"
                        f"Tip: enable training video recording, or use checkpoint-based generation instead."
                    )
                    return

                try:
                    speed = float(speed_var.get())
                    if speed <= 0:
                        raise ValueError("Speed must be positive")
                except ValueError as e:
                    messagebox.showerror("Invalid Input", f"Invalid speed multiplier: {e}")
                    return

                add_overlays = overlays_var.get()
                dialog.destroy()

                overlay_text = "with overlays" if add_overlays else "without overlays"
                self._append_log(f"⏩ Creating time-lapse for {run_id} at {speed}x speed {overlay_text}...")

                def process_thread():
                    try:
                        output_path = processor.create_timelapse(
                            run_id=run_id,
                            speed_multiplier=speed,
                            add_overlays=add_overlays
                        )

                        if output_path:
                            self.root.after(0, lambda: self._on_timelapse_complete(True, output_path))
                        else:
                            self.root.after(0, lambda: self._on_timelapse_complete(False, None))
                    except Exception as e:
                        self._append_log(f"❌ Error creating time-lapse: {e}")
                        import traceback
                        traceback.print_exc()
                        self.root.after(0, lambda: self._on_timelapse_complete(False, None))

                thread = threading.Thread(target=process_thread, daemon=True)
                thread.start()

            create_btn = ctk.CTkButton(buttons_frame, text="Create Time-Lapse",
                                       command=create_timelapse,
                                       **Theme.get_button_colors("success"))
            create_btn.pack(side="left", padx=5, pady=5)

            cancel_btn = ctk.CTkButton(buttons_frame, text="❌ Cancel",
                                      command=dialog.destroy)
            cancel_btn.pack(side="right", padx=5, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show time-lapse dialog: {e}")
            import traceback
            traceback.print_exc()

    def _on_timelapse_complete(self, success: bool, output_path: Optional[str]):
        """Called when time-lapse creation completes."""
        if success:
            self._append_log(f"✅ Time-lapse created: {output_path}")
            self._refresh_videos()
            messagebox.showinfo("Success", f"Time-lapse video created successfully!\n\n{output_path}")
        else:
            self._append_log(f"❌ Time-lapse creation failed")
            messagebox.showerror("Error", "Failed to create time-lapse video.\nCheck the logs for details.")

    def _create_progression_dialog(self):
        """Show dialog to create milestone progression video."""
        try:
            from conf.config import load_config
            from training.video_post_processor import VideoPostProcessor

            # Load config
            config = load_config(str(self.project_root / "conf" / "config.yaml"))
            processor = VideoPostProcessor(config)

            # DB-first: show all runs (including completed) even if recordings are missing.
            db_runs = self.ml_database.get_experiment_runs()
            if not db_runs:
                messagebox.showwarning("No Training Runs",
                                     "No training runs found in the database.\n\n"
                                     "Start a training session first.")
                return

            # Training-time recordings root (may not exist in checkpoint-only mode)
            training_dir = Path(config.get('paths', {}).get('videos_training', 'video/training'))

            # Create dialog
            dialog = ctk.CTkToplevel(self.root)
            dialog.title("📊 Create Milestone Progression Video")
            dialog.geometry("600x550")
            dialog.transient(self.root)
            dialog.grab_set()

            # Title
            title_label = ctk.CTkLabel(dialog, text="📊 Create Milestone Progression Video",
                                      font=ctk.CTkFont(size=16, weight="bold"))
            title_label.pack(pady=15)

            # Info
            info_label = ctk.CTkLabel(dialog,
                                     text="Create a side-by-side comparison showing AI progress at different milestones.\n"
                                          "Shows early, mid, and late training gameplay simultaneously.",
                                     font=ctk.CTkFont(size=11))
            info_label.pack(pady=5)

            # Run selection
            selection_frame = ctk.CTkFrame(dialog)
            selection_frame.pack(fill="both", expand=True, padx=20, pady=10)

            runs_label = ctk.CTkLabel(selection_frame, text="Select Training Run:",
                                     font=ctk.CTkFont(size=12, weight="bold"))
            runs_label.pack(pady=5)

            runs_listbox = tk.Listbox(selection_frame, height=8)
            runs_listbox.pack(fill="both", expand=True, padx=10, pady=5)

            # Populate runs
            run_data = []
            for run in db_runs:
                run_id = run.run_id
                display_name = run.display_name or run.custom_name or run_id
                run_dir = training_dir / run_id
                video_count = 0
                if run_dir.exists():
                    video_count = len(list(run_dir.glob("*.mp4")))
                suffix = f"(recordings: {video_count})" if video_count > 0 else "(no training recordings)"
                run_data.append({'id': run_id, 'dir': run_dir, 'video_count': video_count, 'display_name': display_name})
                runs_listbox.insert(tk.END, f"{display_name} | {run_id} {suffix}")

            # Layout settings
            layout_frame = ctk.CTkFrame(dialog)
            layout_frame.pack(fill="x", padx=20, pady=10)

            layout_label = ctk.CTkLabel(layout_frame, text="Layout:",
                                        font=ctk.CTkFont(size=12, weight="bold"))
            layout_label.pack(pady=5)

            layout_var = tk.StringVar(value="horizontal")

            layout_options = ctk.CTkFrame(layout_frame)
            layout_options.pack(pady=5)

            ctk.CTkRadioButton(layout_options, text="Horizontal (side-by-side)",
                             variable=layout_var, value="horizontal").pack(side="left", padx=10)
            ctk.CTkRadioButton(layout_options, text="Vertical (stacked)",
                             variable=layout_var, value="vertical").pack(side="left", padx=10)
            ctk.CTkRadioButton(layout_options, text="Grid (2x2)",
                             variable=layout_var, value="grid").pack(side="left", padx=10)

            # Clip duration
            duration_frame = ctk.CTkFrame(dialog)
            duration_frame.pack(fill="x", padx=20, pady=10)

            duration_label = ctk.CTkLabel(duration_frame, text="Clip Duration (seconds):",
                                         font=ctk.CTkFont(size=12, weight="bold"))
            duration_label.pack(pady=5)

            duration_var = tk.StringVar(value="30")
            duration_entry = ctk.CTkEntry(duration_frame, textvariable=duration_var, width=100)
            duration_entry.pack(pady=5)

            # Buttons
            buttons_frame = ctk.CTkFrame(dialog)
            buttons_frame.pack(fill="x", padx=20, pady=10)

            def create_progression():
                selection = runs_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a training run first.")
                    return

                selected_run = run_data[selection[0]]
                run_id = selected_run['id']
                layout = layout_var.get()
                video_count = int(selected_run.get('video_count') or 0)

                if video_count <= 0:
                    messagebox.showwarning(
                        "No Training Recordings",
                        f"Run {run_id} has no training recordings.\n\n"
                        f"Expected: {training_dir / run_id}/*.mp4\n\n"
                        f"Tip: enable training video recording."
                    )
                    return

                try:
                    clip_duration = float(duration_var.get())
                    if clip_duration <= 0:
                        raise ValueError("Duration must be positive")
                except ValueError as e:
                    messagebox.showerror("Invalid Input", f"Invalid clip duration: {e}")
                    return

                dialog.destroy()

                self._append_log(f"📊 Creating milestone progression video for {run_id}...")

                def process_thread():
                    try:
                        output_path = processor.create_milestone_progression_video(
                            run_id=run_id,
                            milestone_percentages=[10, 50, 100],
                            layout=layout,
                            clip_duration=clip_duration
                        )

                        if output_path:
                            self.root.after(0, lambda: self._on_progression_complete(True, output_path))
                        else:
                            self.root.after(0, lambda: self._on_progression_complete(False, None))
                    except Exception as e:
                        self._append_log(f"❌ Error creating progression video: {e}")
                        import traceback
                        traceback.print_exc()
                        self.root.after(0, lambda: self._on_progression_complete(False, None))

                thread = threading.Thread(target=process_thread, daemon=True)
                thread.start()

            create_btn = ctk.CTkButton(buttons_frame, text="📊 Create Progression Video",
                                      command=create_progression,
                                      **Theme.get_button_colors("success"))
            create_btn.pack(side="left", padx=5, pady=5)

            cancel_btn = ctk.CTkButton(buttons_frame, text="❌ Cancel",
                                      command=dialog.destroy)
            cancel_btn.pack(side="right", padx=5, pady=5)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to show progression dialog: {e}")
            import traceback
            traceback.print_exc()

    def _on_progression_complete(self, success: bool, output_path: Optional[str]):
        """Called when progression video creation completes."""
        if success:
            self._append_log(f"✅ Progression video created: {output_path}")
            self._refresh_videos()
            messagebox.showinfo("Success", f"Milestone progression video created successfully!\n\n{output_path}")
        else:
            self._append_log(f"❌ Progression video creation failed")
            messagebox.showerror("Error", "Failed to create progression video.\nCheck the logs for details.")

    def _show_start_training_dialog(self):
        """Show the start training dialog."""
        dialog = StartTrainingDialog(self.root, self.presets, self.games, self.algorithms, app=self)
        result = dialog.show()

        if result:
            self._start_training_process(result)
    
    def _start_training_process(self, config: Dict):
        """Start a new training process with the given configuration."""
        try:
            # Handle both old preset-based config and new simple config
            if config.get('preset') == 'custom' or config.get('preset') not in self.presets:
                # Use simple interface configuration with optimized defaults
                preset = {
                    'total_timesteps': config.get('total_timesteps', 4000000),
                    'vec_envs': 16,  # Increased from 4 to 16 for better parallelization
                    'save_freq': 100000,
                    'extra_args': []
                }
            else:
                # Use traditional preset
                preset = self.presets[config['preset']]

            # Prepare resource limits with optimized defaults
            # CRITICAL FIX: Detect actual CPU count to avoid "invalid CPU" error on VMs
            import psutil
            actual_cpu_count = psutil.cpu_count(logical=True)
            requested_cores = config.get('cpu_cores', 12)
            # Only use cores that actually exist
            available_cores = min(requested_cores, actual_cpu_count) if actual_cpu_count else requested_cores

            resources = ResourceLimits(
                cpu_affinity=list(range(available_cores)),  # Only use cores that exist
                memory_limit_gb=config.get('memory_limit_gb'),
                priority=config.get('priority', 'normal'),
                gpu_id=config.get('gpu_id') if config.get('gpu_id') != 'auto' else None
            )

            # Debug: Print configuration being used
            run_mode = config.get('run_mode', 'new')
            print(f"[CONFIG] Starting training with config:")
            print(f"   Mode: {run_mode}")
            print(f"   Game: {config['game']}")
            print(f"   Algorithm: {config['algorithm']}")
            print(f"   Timesteps: {config.get('total_timesteps', preset['total_timesteps'])}")
            print(f"   Vec Envs: {preset['vec_envs']}")
            print(f"   Save Freq: {preset['save_freq']}")
            print(f"   Output Path: {config.get('output_path')}")
            if run_mode == 'continue':
                print(f"   Resume From: {config.get('resume_checkpoint')}")

            requested_run_id = config.get('run_id')
            if not requested_run_id:
                requested_run_id = generate_run_id()
                config['run_id'] = requested_run_id
                print(f"[RUN FLOW] generated_run_id={requested_run_id} (config missing run_id)")

            # Prepare extra args with Fast Mode support
            extra_args = list(preset.get('extra_args', []))
            if config.get('fast_mode'):
                extra_args.append('--fast')

            # Create process
            process_id = self.process_manager.create_process(
                game=config['game'],
                algorithm=config['algorithm'],
                run_id=requested_run_id,
                total_timesteps=config.get('total_timesteps', preset['total_timesteps']),
                vec_envs=preset['vec_envs'],
                save_freq=preset['save_freq'],
                resources=resources,
                extra_args=extra_args,
                custom_output_path=config.get('output_path'),
                resume_from_checkpoint=config.get('resume_checkpoint'),
                target_hours=config.get('target_hours'),
                hyperparameters=config.get('hyperparameters'),
                training_video_enabled=config.get('training_video_enabled', False),
                custom_name=config.get('custom_name'),
                base_run_id=config.get('base_run_id'),
                leg_index=config.get('leg_index'),
                branch_id=config.get('branch_id'),
                root_name=config.get('root_name'),
                display_name=config.get('display_name'),
                branch_token=config.get('branch_token'),
                variant_index=config.get('variant_index'),
                parent_run_id=config.get('parent_run_id'),
                parent_checkpoint_path=config.get('parent_checkpoint_path'),
                start_timestep=config.get('start_timestep', 0),
                target_timestep=config.get('target_timestep')
            )

            print(f"[RUN FLOW] requested_run_id={requested_run_id} returned_process_id={process_id}")
            if process_id != requested_run_id:
                mismatch_message = f"Run ID mismatch: requested {requested_run_id} but process manager returned {process_id}. Aborting launch."
                logging.error(mismatch_message)
                self._append_log(mismatch_message)
                try:
                    self.process_manager.stop_process(process_id)
                except Exception:
                    pass
                return

            # Create experiment run in ML database
            db_run_id, collector_run_id = self._create_experiment_run(process_id, config, preset)
            print(f"[RUN FLOW] db_run_id={db_run_id} collector_run_id={collector_run_id}")

            # Start log streaming
            self.process_manager.start_log_stream(process_id, self._append_log)

            # Refresh process list
            self._refresh_processes()

            # Switch to Training Processes tab to show the new process
            self.tabview.set("Training Processes")

            # Create friendly log message
            game_display = config.get('game_display', config['game'])
            system = config.get('system', 'Unknown')
            video_length = config.get('video_length_option', 'Standard Training')
            run_mode = config.get('run_mode', 'new')

            if run_mode == 'continue':
                self._append_log(f"▶️ Resuming AI Training:")
            else:
                self._append_log(f"🚀 Started AI Training:")
            self._append_log(f"   🕹️ System: {system}")
            self._append_log(f"   🎯 Game: {game_display}")
            self._append_log(f"   ⏱️ Length: {video_length}")
            self._append_log(f"   🤖 Algorithm: {config['algorithm'].upper()}")
            self._append_log(f"   🏷️ Run ID: {process_id}")
            if run_mode == 'continue':
                self._append_log(f"   🔄 Resuming from checkpoint")
            self._append_log(f"   📁 Videos: {config.get('output_path', 'Default location')}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
            self._append_log(f"❌ Failed to start training: {e}")

    def _create_experiment_run(self, process_id: str, config: Dict, preset: Dict):
        """Create an experiment run in the ML database and start metrics collection."""
        db_run_id = None
        collector_run_id = None
        try:
            from .ml_metrics import ExperimentRun, ExperimentConfig

            # Get hyperparameters from config or use defaults
            hyperparams = config.get('hyperparameters', {})

            # Create experiment configuration
            experiment_config = ExperimentConfig(
                algorithm=config['algorithm'],
                policy_type="CnnPolicy",  # Default for Atari
                learning_rate=hyperparams.get('learning_rate', 0.00025),
                batch_size=hyperparams.get('batch_size', 256),
                n_steps=hyperparams.get('n_steps', 128),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.1),
                ent_coef=hyperparams.get('ent_coef', 0.01),
                vf_coef=hyperparams.get('vf_coef', 0.5),
                max_grad_norm=hyperparams.get('max_grad_norm', 0.5),
                env_id=config['game'],
                n_envs=preset['vec_envs'],
                frame_stack=4,  # Default for Atari
                action_repeat=1,  # Default
                total_timesteps=config.get('total_timesteps', preset['total_timesteps']),
                eval_freq=10000,  # Default
                save_freq=preset['save_freq'],
                device="auto",
                seed=42  # Default seed
            )

            # Build experiment display name
            game_short_name = config['game'].split('/')[-1].replace('NoFrameskip-v4', '').replace('-v5', '')
            custom_name = config.get('custom_name')
            leg_number = config.get('leg_number', 1)

            if custom_name:
                experiment_name = f"{custom_name} - {game_short_name} - leg {leg_number}"
            else:
                experiment_name = f"{config['algorithm']}-{game_short_name} - leg {leg_number}"

            # Create experiment run
            experiment_run = ExperimentRun(
                run_id=process_id,
                experiment_name=experiment_name,
                custom_name=config.get('custom_name'),
                leg_number=leg_number,
                base_run_id=config.get('base_run_id'),
                root_name=config.get('root_name'),
                display_name=config.get('display_name'),
                branch_token=config.get('branch_token'),
                variant_index=config.get('variant_index', 1),
                start_time=datetime.now(),
                status="running",
                leg_start_timestep=config.get('leg_start_timestep', 0),
                config=experiment_config,
                description=f"Training {config['algorithm']} on {config['game']} for {config.get('total_timesteps', preset['total_timesteps']):,} timesteps (Leg {leg_number})",
                tags=[config['algorithm'], config['game'].split('/')[-1], "auto-generated", f"leg-{leg_number}"]
            )

            # Store in database
            success = self.ml_database.create_experiment_run(experiment_run)
            if success:
                db_run_id = process_id
                print(f"✅ Created experiment run in ML database: {process_id}")

                # Persist lineage/provenance fields that are not part of ExperimentRun dataclass
                self.ml_database.update_experiment_run(
                    process_id,
                    base_run_id=config.get('base_run_id'),
                    leg_number=config.get('leg_number'),
                    leg_start_timestep=config.get('leg_start_timestep', 0),
                    branch_id=config.get('branch_id'),
                    branch_token=config.get('branch_token'),
                    root_name=config.get('root_name'),
                    display_name=config.get('display_name'),
                    variant_index=config.get('variant_index'),
                    parent_run_id=config.get('parent_run_id'),
                    parent_checkpoint_path=config.get('parent_checkpoint_path'),
                    start_timestep=config.get('start_timestep'),
                    target_timestep=config.get('target_timestep')
                )

                # Start metrics collection
                def get_logs():
                    return self.process_manager.get_recent_logs(process_id) or ""

                # Get process PID for system monitoring
                processes = self.process_manager.get_processes()
                process = next((p for p in processes if p.id == process_id), None)
                pid = process.pid if process else None

                # Store PID in database for process recovery
                if pid:
                    self.ml_database.update_process_info(
                        run_id=process_id,
                        pid=pid,
                        paused=False
                    )

                self.ml_collector.start_collection(
                    run_id=process_id,
                    log_source=get_logs,
                    pid=pid,
                    interval=5.0  # Collect metrics every 5 seconds
                )
                collector_run_id = process_id
                print(f"✅ Started metrics collection for: {process_id}")
            else:
                print(f"❌ Failed to create experiment run in ML database: {process_id}")

        except Exception as e:
            print(f"❌ Error creating experiment run: {e}")
            self._append_log(f"Warning: Failed to create ML experiment tracking: {e}")

        return db_run_id, collector_run_id
    
    def _show_exit_prompt(self) -> str:
        """Prompt user when active runs exist. Returns 'keep', 'pause', or 'cancel'."""
        choice = {"value": "cancel"}

        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Close Retro ML Desktop")
        dialog.geometry("420x200")
        dialog.grab_set()
        dialog.lift()
        dialog.attributes("-topmost", True)

        label = ctk.CTkLabel(
            dialog,
            text="Active training is in progress. What do you want to do?",
            wraplength=380,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        label.pack(pady=(20, 15), padx=20)

        buttons_frame = ctk.CTkFrame(dialog)
        buttons_frame.pack(pady=5, padx=20, fill="x")

        def set_choice(val: str):
            choice["value"] = val
            dialog.destroy()

        keep_btn = ctk.CTkButton(
            buttons_frame,
            text="Keep training running (recommended)",
            command=lambda: set_choice("keep"),
            height=36,
        )
        keep_btn.pack(fill="x", pady=4)

        pause_btn = ctk.CTkButton(
            buttons_frame,
            text="Pause runs, then exit",
            command=lambda: set_choice("pause"),
            height=36,
        )
        pause_btn.pack(fill="x", pady=4)

        cancel_btn = ctk.CTkButton(
            buttons_frame,
            text="Cancel",
            command=lambda: set_choice("cancel"),
            height=32,
        )
        cancel_btn.pack(fill="x", pady=4)

        self.root.wait_window(dialog)
        return choice["value"]

    def on_closing(self):
        """Handle application close event."""
        # Check for running processes
        running_processes = []
        processes = self.process_manager.get_processes()

        for process in processes:
            if process.status == "running":
                # Get progress from database
                try:
                    conn = self.ml_database.connection
                    cursor = conn.cursor()
                    cursor.execute("SELECT current_timestep FROM experiment_runs WHERE run_id = ?", (process.id,))
                    result = cursor.fetchone()
                    progress = (result[0] / 1000000 * 100) if result and result[0] else 0.0
                except:
                    progress = 0.0

                from tools.retro_ml_desktop.close_confirmation_dialog import ProcessInfo
                running_processes.append(ProcessInfo(
                    id=process.id,
                    name=process.name,
                    status=process.status,
                    progress=progress
                ))

        if running_processes:
            result = self._show_exit_prompt()

            if result == "cancel":
                return  # Don't close
            elif result == "pause":
                self.pause_all_training()
                timestamp = datetime.now().isoformat()
                for proc in running_processes:
                    note = f"ui_exit_pause: UI closed at {timestamp}, run paused"
                    try:
                        self.ml_database.append_status_note(proc.id, note)
                    except Exception as e:
                        self._append_log(f"[exit] failed to append note for {proc.id}: {e}")
            elif result == "keep":
                self._ensure_supervisor_running()
                timestamp = datetime.now().isoformat()
                for proc in running_processes:
                    note = f"ui_exit_keep_running: UI closed at {timestamp}, run left active"
                    try:
                        self.ml_database.append_status_note(proc.id, note)
                    except Exception as e:
                        self._append_log(f"[exit] failed to append note for {proc.id}: {e}")

        # Close the application
        self.cleanup_and_exit()

    def _show_cuda_diagnostics(self):
        """Show CUDA diagnostics dialog."""
        try:
            # Run diagnostics
            diagnostics = self.cuda_diagnostics.diagnose_system()

            # Create diagnostics dialog
            dialog = CUDADiagnosticsDialog(self.root, diagnostics, self.cuda_diagnostics)
            dialog.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to run CUDA diagnostics: {e}")
            self._append_log(f"❌ Failed to run CUDA diagnostics: {e}")

    def pause_all_training(self):
        """Pause all running training processes."""
        processes = self.process_manager.get_processes()

        for process in processes:
            if process.status == "running":
                try:
                    # Pause the process
                    self.process_manager.pause_process(process.id)

                    # Update database to mark as paused
                    self.ml_database.update_process_info(
                        run_id=process.id,
                        pid=process.pid,
                        paused=True
                    )

                    print(f"✅ Paused training: {process.id}")
                except Exception as e:
                    print(f"❌ Error pausing {process.id}: {e}")

    def stop_all_training(self):
        """Stop all running training processes."""
        processes = self.process_manager.get_processes()

        for process in processes:
            if process.status == "running":
                try:
                    # Stop the process
                    self.process_manager.stop_process(process.id)

                    # Update database status
                    self.ml_database.update_experiment_status(
                        run_id=process.id,
                        status="stopped",
                        end_time=datetime.now().isoformat()
                    )

                    print(f"✅ Stopped training: {process.id}")
                except Exception as e:
                    print(f"❌ Error stopping {process.id}: {e}")

    def recover_paused_processes(self):
        """Recover paused processes when application starts."""
        try:
            paused_experiments = self.ml_database.get_paused_experiments()

            if paused_experiments:
                print(f"🔄 Found {len(paused_experiments)} paused experiments")

                # Show recovery dialog
                self.show_recovery_dialog(paused_experiments)
        except Exception as e:
            print(f"❌ Error recovering paused processes: {e}")

    def show_recovery_dialog(self, paused_experiments):
        """Show dialog to recover paused experiments."""
        if not paused_experiments:
            return

        # Create simple recovery dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Resume Paused Training?")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (400 // 2)
        y = (dialog.winfo_screenheight() // 2) - (300 // 2)
        dialog.geometry(f"400x300+{x}+{y}")

        # Content
        label = ctk.CTkLabel(
            dialog,
            text=f"Found {len(paused_experiments)} paused training sessions.\nWould you like to resume them?",
            font=ctk.CTkFont(size=14)
        )
        label.pack(pady=20)

        # List paused experiments
        for exp in paused_experiments[:3]:  # Show first 3
            exp_label = ctk.CTkLabel(
                dialog,
                text=f"• {exp['experiment_name']} ({exp['current_timestep']} steps)",
                font=ctk.CTkFont(size=12)
            )
            exp_label.pack(pady=5)

        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)

        def resume_all():
            dialog.destroy()
            self.resume_paused_experiments(paused_experiments)

        def skip_all():
            # Mark as no longer paused
            for exp in paused_experiments:
                self.ml_database.update_process_info(exp['run_id'], paused=False)
            dialog.destroy()

        resume_btn = ctk.CTkButton(button_frame, text="Resume All", command=resume_all)
        resume_btn.pack(side="left", padx=10)

        skip_btn = ctk.CTkButton(button_frame, text="Skip", command=skip_all)
        skip_btn.pack(side="left", padx=10)

    def resume_paused_experiments(self, experiments):
        """Resume paused experiments."""
        for exp in experiments:
            try:
                # TODO: Implement resume logic
                # This would restart the training process from the last checkpoint
                print(f"🔄 Resuming: {exp['experiment_name']}")

                # For now, just mark as no longer paused
                self.ml_database.update_process_info(exp['run_id'], paused=False)

            except Exception as e:
                print(f"❌ Error resuming {exp['run_id']}: {e}")

    def cleanup_and_exit(self):
        """Clean up resources and exit."""
        try:
            # Stop system monitoring
            self.system_monitor.stop_monitoring()

            # Stop GPU telemetry sampler
            if hasattr(self, "_gpu_sampler_stop_event") and self._gpu_sampler_stop_event:
                self._gpu_sampler_stop_event.set()

            # Stop ML collector
            self.ml_collector.stop_all_collection()

            # Close database
            self.ml_database.close()

            print("✅ Application cleanup complete")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
        finally:
            self.root.destroy()

    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.system_monitor.stop_monitoring()
            if hasattr(self, "_gpu_sampler_stop_event") and self._gpu_sampler_stop_event:
                self._gpu_sampler_stop_event.set()
            self.ml_collector.stop_all_collection()
            self.ml_database.close()


class CUDADiagnosticsDialog:
    """Dialog for displaying CUDA diagnostics and troubleshooting."""

    def __init__(self, parent, diagnostics, cuda_diagnostics):
        self.parent = parent
        self.diagnostics = diagnostics
        self.cuda_diagnostics = cuda_diagnostics

        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("CUDA Diagnostics & Troubleshooting")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"800x600+{x}+{y}")

        self._setup_ui()

    def _setup_ui(self):
        """Setup the diagnostics UI."""
        # Main frame with scrollable content
        main_frame = ctk.CTkScrollableFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="CUDA System Diagnostics",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Generate and display report
        report_text = self.cuda_diagnostics.format_diagnostic_report(self.diagnostics)

        # Text widget for report
        text_widget = ctk.CTkTextbox(main_frame, height=400, font=ctk.CTkFont(family="Consolas", size=11))
        text_widget.pack(fill="both", expand=True, pady=(0, 20))
        text_widget.insert("1.0", report_text)
        text_widget.configure(state="disabled")  # Make read-only

        # Suggested configuration
        suggestions = self.cuda_diagnostics.get_training_config_suggestions(self.diagnostics)

        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 20))

        ctk.CTkLabel(config_frame, text="Recommended Training Configuration",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))

        config_text = []
        config_text.append(f"Device: {suggestions['device'].upper()}")
        if 'vec_envs' in suggestions:
            config_text.append(f"Environments: {suggestions['vec_envs']}")
        if 'batch_size' in suggestions:
            config_text.append(f"Batch Size: {suggestions['batch_size']}")
        if 'n_steps' in suggestions:
            config_text.append(f"Steps per Update: {suggestions['n_steps']}")

        config_label = ctk.CTkLabel(config_frame, text=" | ".join(config_text),
                                   font=ctk.CTkFont(size=12))
        config_label.pack(pady=(0, 10))

        # Action buttons
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x")

        # Refresh button
        refresh_btn = ctk.CTkButton(button_frame, text="Refresh Diagnostics",
                                   command=self._refresh_diagnostics,
                                   **Theme.get_button_colors("success"))
        refresh_btn.pack(side="left", padx=(10, 5), pady=10)

        # Copy report button
        copy_btn = ctk.CTkButton(button_frame, text="Copy Report",
                                command=self._copy_report,
                                **Theme.get_button_colors("info"))
        copy_btn.pack(side="left", padx=5, pady=10)

        # Close button
        close_btn = ctk.CTkButton(button_frame, text="Close",
                                 command=self.dialog.destroy,
                                 **Theme.get_button_colors("secondary"))
        close_btn.pack(side="right", padx=(5, 10), pady=10)

    def _refresh_diagnostics(self):
        """Refresh the diagnostics information."""
        try:
            # Re-run diagnostics
            self.diagnostics = self.cuda_diagnostics.diagnose_system()

            # Close current dialog and show new one
            self.dialog.destroy()

            # Create new dialog with updated diagnostics
            new_dialog = CUDADiagnosticsDialog(self.parent, self.diagnostics, self.cuda_diagnostics)
            new_dialog.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh diagnostics: {e}")

    def _copy_report(self):
        """Copy the diagnostic report to clipboard."""
        try:
            report_text = self.cuda_diagnostics.format_diagnostic_report(self.diagnostics)
            self.dialog.clipboard_clear()
            self.dialog.clipboard_append(report_text)
            messagebox.showinfo("Success", "Diagnostic report copied to clipboard!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy report: {e}")

    def show(self):
        """Show the dialog."""
        self.dialog.wait_window()


class StartTrainingDialog:
    """Simplified dialog for configuring training."""

    def __init__(self, parent, presets: Dict, games: List[str], algorithms: List[str], app=None):
        self.parent = parent
        self.app = app  # Store reference to the main app for accessing process_manager
        self.presets = presets
        self.games = games
        self.algorithms = algorithms
        self.result = None
        
        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Start AI Training")
        self.dialog.geometry("900x800")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)
        self.dialog.minsize(800, 700)  # Ensure minimum size for button visibility

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (800 // 2)
        self.dialog.geometry(f"900x800+{x}+{y}")

        # Training video toggle (default off, controlled per run)
        self.training_video_var = tk.BooleanVar(value=False)
        # Fast Mode toggle (default off)
        self.fast_mode_var = tk.BooleanVar(value=False)

        self._create_dialog_ui()
    
    def _create_dialog_ui(self):
        """Create the dialog UI."""
        # Create scrollable frame for content
        scrollable_frame = ctk.CTkScrollableFrame(self.dialog)
        scrollable_frame.pack(fill="both", expand=True, padx=15, pady=15)

        # Main content frame
        main_frame = ctk.CTkFrame(scrollable_frame)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="Start New AI Training",
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 15))

        # === SECTION 1: Game Selection (2-column layout) ===
        game_selection_frame = ctk.CTkFrame(main_frame)
        game_selection_frame.pack(fill="x", pady=(0, 10))

        # Gaming System (left column)
        system_col = ctk.CTkFrame(game_selection_frame)
        system_col.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        ctk.CTkLabel(system_col, text="Gaming System:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5))

        self.system_var = tk.StringVar(value="Atari")
        system_options = ["Atari", "Classic Control", "Box2D"]
        system_combo = ctk.CTkOptionMenu(system_col, variable=self.system_var,
                                       values=system_options, command=self._on_system_changed)
        system_combo.pack(fill="x", pady=(0, 5))

        # Game Selection (right column)
        game_col = ctk.CTkFrame(game_selection_frame)
        game_col.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        ctk.CTkLabel(game_col, text="Choose Your Game:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5))

        # Initialize game options based on system
        self.game_var = tk.StringVar()
        self.game_combo = ctk.CTkOptionMenu(game_col, variable=self.game_var, values=[""],
                                           command=self._on_game_changed)
        self.game_combo.pack(fill="x", pady=(0, 5))

        # Separator
        ctk.CTkFrame(main_frame, height=2, fg_color="gray30").pack(fill="x", pady=5)

        # === SECTION 2: Experiment Configuration (2-column layout) ===
        config_frame = ctk.CTkFrame(main_frame)
        config_frame.pack(fill="x", pady=(0, 10))

        # Experiment Name (left column)
        name_col = ctk.CTkFrame(config_frame)
        name_col.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        ctk.CTkLabel(name_col, text="Experiment Name (Optional):",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5))

        self.custom_name_var = tk.StringVar(value="")
        self.name_entry = ctk.CTkEntry(name_col, textvariable=self.custom_name_var,
                                  placeholder_text="e.g., 'high-exploration-test'")
        self.name_entry.pack(fill="x", pady=(0, 2))

        name_info = ctk.CTkLabel(name_col,
                                text="Auto format: name - game - leg 1",
                                font=ctk.CTkFont(size=9),
                                text_color="gray")
        name_info.pack(anchor="w", pady=(0, 5))

        # Algorithm Selection (right column)
        algo_col = ctk.CTkFrame(config_frame)
        algo_col.pack(side="left", fill="both", expand=True, padx=(5, 10), pady=10)

        ctk.CTkLabel(algo_col, text="AI Algorithm:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(0, 5))

        self.algo_var = tk.StringVar(value="PPO")
        algorithm_combo = ctk.CTkOptionMenu(algo_col, variable=self.algo_var,
                                           values=["PPO", "DQN"])
        algorithm_combo.pack(fill="x", pady=(0, 2))

        # Algorithm info label
        algo_info = ctk.CTkLabel(algo_col,
                                text="PPO: Best for most | DQN: Breakout",
                                font=ctk.CTkFont(size=9),
                                text_color="gray")
        algo_info.pack(anchor="w", pady=(0, 5))

        # Separator
        ctk.CTkFrame(main_frame, height=2, fg_color="gray30").pack(fill="x", pady=5)

        # === SECTION 3: Training Mode ===
        run_mode_frame = ctk.CTkFrame(main_frame)
        run_mode_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(run_mode_frame, text="Training Mode:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 5), padx=10)

        self.run_mode_var = tk.StringVar(value="new")

        # Radio buttons for mode selection
        mode_container = ctk.CTkFrame(run_mode_frame)
        mode_container.pack(fill="x", padx=10, pady=(0, 10))

        new_run_radio = ctk.CTkRadioButton(mode_container, text="Start New Run",
                                           variable=self.run_mode_var, value="new",
                                           command=self._on_run_mode_changed)
        new_run_radio.pack(side="left", padx=(10, 20))

        continue_run_radio = ctk.CTkRadioButton(mode_container, text="Continue Previous Run",
                                               variable=self.run_mode_var, value="continue",
                                               command=self._on_run_mode_changed)
        continue_run_radio.pack(side="left", padx=(0, 10))

        # Previous runs dropdown (initially hidden)
        self.previous_runs_frame = ctk.CTkFrame(run_mode_frame)
        self.previous_runs_var = tk.StringVar()
        self.previous_runs_combo = ctk.CTkOptionMenu(self.previous_runs_frame,
                                                    variable=self.previous_runs_var,
                                                    values=["No previous runs available"],
                                                    command=self._on_previous_run_selected)
        self.previous_runs_combo.pack(fill="x", padx=10, pady=(5, 10))

        # Run info label
        self.run_info_label = ctk.CTkLabel(self.previous_runs_frame, text="",
                                          justify="left", font=ctk.CTkFont(size=11))
        self.run_info_label.pack(anchor="w", padx=10, pady=(0, 10))

        # Separator
        ctk.CTkFrame(main_frame, height=2, fg_color="gray30").pack(fill="x", pady=5)

        # === SECTION 4: Training Duration ===
        video_frame = ctk.CTkFrame(main_frame)
        video_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(video_frame, text="Target Video Length:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 5), padx=10)

        # Time input frame
        time_input_frame = ctk.CTkFrame(video_frame)
        time_input_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Hours input
        hours_frame = ctk.CTkFrame(time_input_frame)
        hours_frame.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(hours_frame, text="Hours:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 5))
        self.video_hours_var = tk.StringVar(value="4")
        hours_entry = ctk.CTkEntry(hours_frame, textvariable=self.video_hours_var, width=60)
        hours_entry.pack(side="left")

        # Minutes input
        minutes_frame = ctk.CTkFrame(time_input_frame)
        minutes_frame.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(minutes_frame, text="Minutes:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 5))
        self.video_minutes_var = tk.StringVar(value="0")
        minutes_entry = ctk.CTkEntry(minutes_frame, textvariable=self.video_minutes_var, width=60)
        minutes_entry.pack(side="left")

        # Quick preset buttons
        preset_frame = ctk.CTkFrame(time_input_frame)
        preset_frame.pack(side="left", fill="x", expand=True)

        ctk.CTkLabel(preset_frame, text="Quick Presets:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(10, 5))

        preset_buttons = [
            ("30m", 0, 30),
            ("1h", 1, 0),
            ("4h", 4, 0),
            ("10h", 10, 0)
        ]

        for label, hours, minutes in preset_buttons:
            btn = ctk.CTkButton(preset_frame, text=label, width=50, height=24,
                              command=lambda h=hours, m=minutes: self._set_video_length_preset(h, m))
            btn.pack(side="left", padx=2)

        # Info label
        self.video_length_info = ctk.CTkLabel(video_frame,
                                             text="Training will run long enough to generate this much video content",
                                             font=ctk.CTkFont(size=10),
                                             text_color="gray")
        self.video_length_info.pack(anchor="w", padx=10, pady=(0, 10))

        # Fast Mode (no live video)
        fast_mode_frame = ctk.CTkFrame(main_frame)
        fast_mode_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            fast_mode_frame,
            text="Training Speed Override:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)

        ctk.CTkCheckBox(
            fast_mode_frame,
            text="Fast Mode ⚡ (Disable live recording for max speed)",
            variable=self.fast_mode_var,
            command=self._on_fast_mode_changed
        ).pack(anchor="w", padx=10, pady=(0, 5))

        self.fast_mode_info = ctk.CTkLabel(
            fast_mode_frame,
            text="Videos will be auto-generated AFTER training completes.",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.fast_mode_info.pack(anchor="w", padx=10, pady=(0, 5))

        # Separator
        ctk.CTkFrame(main_frame, height=2, fg_color="gray30").pack(fill="x", pady=5)

        # Training video recording opt-in
        video_record_frame = ctk.CTkFrame(main_frame)
        video_record_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            video_record_frame,
            text="Training Video Recording:",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", pady=(10, 5), padx=10)

        ctk.CTkCheckBox(
            video_record_frame,
            text="Record training-time footage (saved to video/training/<run_id>/)",
            variable=self.training_video_var
        ).pack(anchor="w", padx=10, pady=(0, 5))

        ctk.CTkLabel(
            video_record_frame,
            text="Recommended: enable only for short runs or diagnostics.",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        ).pack(anchor="w", padx=10, pady=(0, 5))

        # Separator
        ctk.CTkFrame(main_frame, height=2, fg_color="gray30").pack(fill="x", pady=5)

        # === SECTION 5: Advanced Settings ===
        hyperparams_frame = ctk.CTkFrame(main_frame)
        hyperparams_frame.pack(fill="x", pady=(0, 10))

        # Header with toggle button
        header_frame = ctk.CTkFrame(hyperparams_frame)
        header_frame.pack(fill="x", padx=10, pady=(10, 0))

        self.show_hyperparams = tk.BooleanVar(value=False)
        toggle_btn = ctk.CTkButton(header_frame, text="▶", width=30, height=24,
                                   command=self._toggle_hyperparams,
                                   **Theme.get_button_colors("secondary"))
        toggle_btn.pack(side="left", padx=(0, 10))

        ctk.CTkLabel(header_frame, text="Training Hyperparameters (Optional)",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")

        # Hyperparameters container (initially hidden)
        self.hyperparams_container = ctk.CTkFrame(hyperparams_frame)

        # Create a grid layout for hyperparameters
        grid_frame = ctk.CTkFrame(self.hyperparams_container)
        grid_frame.pack(fill="x", padx=10, pady=10)

        # Initialize hyperparameter variables with defaults
        self.learning_rate_var = tk.StringVar(value="0.00025")  # 2.5e-4
        self.batch_size_var = tk.StringVar(value="512")
        self.n_steps_var = tk.StringVar(value="256")
        self.ent_coef_var = tk.StringVar(value="0.01")
        self.gamma_var = tk.StringVar(value="0.99")
        self.gae_lambda_var = tk.StringVar(value="0.95")
        self.clip_range_var = tk.StringVar(value="0.1")
        self.vf_coef_var = tk.StringVar(value="0.5")
        self.max_grad_norm_var = tk.StringVar(value="0.5")

        # Store entry widgets for enabling/disabling
        self.hyperparam_entries = []

        # Row 0: Learning Rate and Batch Size
        ctk.CTkLabel(grid_frame, text="Learning Rate:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=0, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.learning_rate_var, width=100)
        entry.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=3)
        self.hyperparam_entries.append(entry)

        ctk.CTkLabel(grid_frame, text="Batch Size:", font=ctk.CTkFont(size=11)).grid(
            row=0, column=2, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.batch_size_var, width=100)
        entry.grid(row=0, column=3, sticky="w", padx=(0, 5), pady=3)
        self.hyperparam_entries.append(entry)

        # Row 1: N Steps and Entropy Coefficient
        ctk.CTkLabel(grid_frame, text="N Steps:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.n_steps_var, width=100)
        entry.grid(row=1, column=1, sticky="w", padx=(0, 20), pady=3)
        self.hyperparam_entries.append(entry)

        ctk.CTkLabel(grid_frame, text="Entropy Coef:", font=ctk.CTkFont(size=11)).grid(
            row=1, column=2, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.ent_coef_var, width=100)
        entry.grid(row=1, column=3, sticky="w", padx=(0, 5), pady=3)
        self.hyperparam_entries.append(entry)

        # Row 2: Gamma and GAE Lambda
        ctk.CTkLabel(grid_frame, text="Gamma:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=0, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.gamma_var, width=100)
        entry.grid(row=2, column=1, sticky="w", padx=(0, 20), pady=3)
        self.hyperparam_entries.append(entry)

        ctk.CTkLabel(grid_frame, text="GAE Lambda:", font=ctk.CTkFont(size=11)).grid(
            row=2, column=2, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.gae_lambda_var, width=100)
        entry.grid(row=2, column=3, sticky="w", padx=(0, 5), pady=3)
        self.hyperparam_entries.append(entry)

        # Row 3: Clip Range and Value Coefficient
        ctk.CTkLabel(grid_frame, text="Clip Range:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=0, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.clip_range_var, width=100)
        entry.grid(row=3, column=1, sticky="w", padx=(0, 20), pady=3)
        self.hyperparam_entries.append(entry)

        ctk.CTkLabel(grid_frame, text="Value Coef:", font=ctk.CTkFont(size=11)).grid(
            row=3, column=2, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.vf_coef_var, width=100)
        entry.grid(row=3, column=3, sticky="w", padx=(0, 5), pady=3)
        self.hyperparam_entries.append(entry)

        # Row 4: Max Gradient Norm
        ctk.CTkLabel(grid_frame, text="Max Grad Norm:", font=ctk.CTkFont(size=11)).grid(
            row=4, column=0, sticky="w", padx=(5, 10), pady=3)
        entry = ctk.CTkEntry(grid_frame, textvariable=self.max_grad_norm_var, width=100)
        entry.grid(row=4, column=1, sticky="w", padx=(0, 20), pady=3)
        self.hyperparam_entries.append(entry)

        # Row 5: Preset buttons header
        preset_header = ctk.CTkLabel(grid_frame, text="Quick Presets:",
                                     font=ctk.CTkFont(size=12, weight="bold"))
        preset_header.grid(row=5, column=0, columnspan=4, sticky="w", padx=5, pady=(10, 3))

        # Row 6: Preset buttons
        preset_btn_container = ctk.CTkFrame(grid_frame)
        preset_btn_container.grid(row=6, column=0, columnspan=4, sticky="ew", padx=5, pady=(0, 5))

        presets_config = [
            ("Balanced", self._set_hyperparams_balanced, "Standard PPO settings"),
            ("Explore", self._set_hyperparams_explore, "High exploration for discovery"),
            ("Exploit", self._set_hyperparams_exploit, "Refine learned behaviors"),
            ("Fast", self._set_hyperparams_fast, "Aggressive learning speed"),
            ("Stable", self._set_hyperparams_stable, "Conservative & stable")
        ]

        self.hyperparam_preset_buttons = []
        for i, (label, command, description) in enumerate(presets_config):
            btn = ctk.CTkButton(preset_btn_container, text=label, width=100, height=28,
                               command=command,
                               **Theme.get_button_colors("secondary"))
            btn.grid(row=0, column=i, padx=3, pady=0)
            self.hyperparam_preset_buttons.append(btn)

        # Locked warning label (initially hidden)
        self.hyperparam_locked_label = ctk.CTkLabel(
            grid_frame,
            text="Hyperparameters locked to match original run",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#FFA500"
        )
        # Will be shown when continuing a run

        # Info labels
        info_frame = ctk.CTkFrame(self.hyperparams_container)
        info_frame.pack(fill="x", padx=10, pady=(3, 5))

        info_lines = [
            "Balanced: Standard settings | Explore: Discovery (fixes +20 plateau) | Exploit: Refine policy",
            "Fast: Quick iteration (unstable) | Stable: Conservative & reliable"
        ]

        for line in info_lines:
            label = ctk.CTkLabel(info_frame, text=line,
                               font=ctk.CTkFont(size=9),
                               text_color="gray",
                               anchor="w")
            label.pack(anchor="w", padx=5, pady=0)

        # Initialize variables
        self.resource_config = None
        self.run_id_var = tk.StringVar(value=generate_run_id())

        # Use project 'outputs' folder by default (user preference)
        if hasattr(self, 'app') and self.app and hasattr(self.app, 'project_root'):
             default_video_path = str(self.app.project_root / 'outputs')
        else:
             # Fallback if app reference missing, use generic outputs folder
             default_video_path = "outputs"

        self.output_path_var = tk.StringVar(value=default_video_path)

        # Action Buttons (right after video length)
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=(15, 10))

        # Button instructions
        button_info = ctk.CTkLabel(button_frame, text="Ready to start training?",
                                  font=ctk.CTkFont(size=14, weight="bold"))
        button_info.pack(pady=(10, 8))

        # Button container
        btn_container = ctk.CTkFrame(button_frame)
        btn_container.pack(fill="x", padx=20, pady=(0, 10))

        cancel_btn = ctk.CTkButton(btn_container, text="Cancel", command=self._cancel,
                                   height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                   width=140,
                                   **Theme.get_button_colors("secondary"))
        cancel_btn.pack(side="right", padx=(10, 0))

        start_btn = ctk.CTkButton(btn_container, text="Start AI Training", command=self._start,
                                  height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                  width=200,
                                  **Theme.get_button_colors("success"))
        start_btn.pack(side="right", padx=(0, 10))

        # Simple summary display
        self.summary_label = ctk.CTkLabel(main_frame, text="", justify="left",
                                         font=ctk.CTkFont(size=11))
        self.summary_label.pack(pady=(10, 0))

        # Initialize system and games (after all UI elements are created)
        self._on_system_changed("Atari")

    def _get_game_systems(self):
        """Get available gaming systems and their games."""
        return {
            "Atari": {
                # Popular/Classic Games
                "🎯 Breakout": "ALE/Breakout-v5",
                "🏓 Pong": "ALE/Pong-v5",
                "👾 Space Invaders": "ALE/SpaceInvaders-v5",
                "🎮 Ms. Pac-Man": "ALE/MsPacman-v5",
                "🚀 Asteroids": "ALE/Asteroids-v5",

                # Action/Shooter Games
                "🔫 Assault": "ALE/Assault-v5",
                "🎖️ Atlantis": "ALE/Atlantis-v5",
                "👾 Alien": "ALE/Alien-v5",
                "⭐ Asterix": "ALE/Asterix-v5",
                "🎯 Battlezone": "ALE/BattleZone-v5",
                "💥 BeamRider": "ALE/BeamRider-v5",
                "🔴 Berzerk": "ALE/Berzerk-v5",
                "🎰 Carnival": "ALE/Carnival-v5",
                "🐛 Centipede": "ALE/Centipede-v5",
                "🎪 Defender": "ALE/Defender-v5",
                "👾 DemonAttack": "ALE/DemonAttack-v5",
                "🚁 Gravitar": "ALE/Gravitar-v5",
                "🎯 Phoenix": "ALE/Phoenix-v5",
                "🎮 Robotank": "ALE/Robotank-v5",

                # Adventure/Platform Games
                "🏔️ Montezuma Revenge": "ALE/MontezumaRevenge-v5",
                "🎮 Pitfall": "ALE/Pitfall-v5",
                "🦘 Kangaroo": "ALE/Kangaroo-v5",
                "🏃 Krull": "ALE/Krull-v5",
                "🏰 Adventure": "ALE/Adventure-v5",

                # Sports Games
                "🏀 Basketball": "ALE/Basketball-v5",
                "🎳 Bowling": "ALE/Bowling-v5",
                "🏈 Football": "ALE/Football-v5",
                "🏒 Ice Hockey": "ALE/IceHockey-v5",
                "🎾 Tennis": "ALE/Tennis-v5",
                "🏐 Volleyball": "ALE/Volleyball-v5",
                "⚾ Video Pinball": "ALE/VideoPinball-v5",

                # Racing Games
                "🏎️ Enduro": "ALE/Enduro-v5",

                # Underwater/Sea Games
                "⚔️ Seaquest": "ALE/Seaquest-v5",
                "🐟 Fishing Derby": "ALE/FishingDerby-v5",

                # Maze/Puzzle Games
                "🎪 Freeway": "ALE/Freeway-v5",
                "🧩 Amidar": "ALE/Amidar-v5",
                "🎲 Venture": "ALE/Venture-v5",

                # Beat-em-up Games
                "🥊 Boxing": "ALE/Boxing-v5",
                "⚔️ Kung Fu Master": "ALE/KungFuMaster-v5",

                # Other Classic Games
                "🎮 Qbert": "ALE/Qbert-v5",
                "🌉 Riverraid": "ALE/Riverraid-v5",
                "🎯 Zaxxon": "ALE/Zaxxon-v5",
                "🎪 Solaris": "ALE/Solaris-v5",
                "🎮 Frostbite": "ALE/Frostbite-v5",
                "🎯 Gopher": "ALE/Gopher-v5",
                "🎮 James Bond": "ALE/Jamesbond-v5",
                "🎯 Name This Game": "ALE/NameThisGame-v5",
                "🎮 Road Runner": "ALE/RoadRunner-v5",
                "🎯 Star Gunner": "ALE/StarGunner-v5",
                "🎮 Time Pilot": "ALE/TimePilot-v5",
                "🎯 Tutankham": "ALE/Tutankham-v5",
                "🎮 Up N Down": "ALE/UpNDown-v5",
                "🎯 Wizard Of Wor": "ALE/WizardOfWor-v5",
                "🎮 Yars Revenge": "ALE/YarsRevenge-v5"
            },
            "Classic Control": {
                "🎯 CartPole": "CartPole-v1",
                "🏔️ Mountain Car": "MountainCar-v0",
                "🚁 Acrobot": "Acrobot-v1",
                "📐 Pendulum": "Pendulum-v1"
            },
            "Box2D": {
                "🌙 Lunar Lander": "LunarLander-v2",
                "🏎️ Car Racing": "CarRacing-v2",
                "🚁 Bipedal Walker": "BipedalWalker-v3"
            }
        }

    def _set_video_length_preset(self, hours, minutes):
        """Set video length from preset button."""
        self.video_hours_var.set(str(hours))
        self.video_minutes_var.set(str(minutes))
        self._update_summary()

    def _get_video_length_config(self):
        """Get training configuration based on video length input."""
        try:
            hours = float(self.video_hours_var.get()) if self.video_hours_var.get() else 0
            minutes = float(self.video_minutes_var.get()) if self.video_minutes_var.get() else 0

            # Convert to total hours
            total_hours = hours + (minutes / 60.0)

            # Ensure minimum of 1 minute (0.0167 hours) to prevent accidental zero-length training
            if total_hours < 0.0167:
                total_hours = 0.0167

            # Calculate timesteps based on realistic GPU training FPS
            # Assuming ~450 FPS average for GPU training (conservative estimate)
            # This matches the calculation in retro_ml/core/experiments/config.py
            estimated_training_fps = 450
            timesteps = int(total_hours * 3600 * estimated_training_fps)

            # Calculate number of milestones (one every 10% of progress, minimum 3)
            milestones = max(3, min(10, int(total_hours * 2)))

            return {
                "target_hours": total_hours,
                "timesteps": timesteps,
                "milestones": milestones
            }
        except (ValueError, AttributeError):
            # Default to 4 hours if there's an error
            return {
                "target_hours": 4,
                "timesteps": 4000000,
                "milestones": 8
            }

    def _toggle_hyperparams(self):
        """Toggle hyperparameters visibility."""
        if self.show_hyperparams.get():
            self.hyperparams_container.pack_forget()
            self.show_hyperparams.set(False)
            # Update toggle button text
            for child in self.dialog.winfo_children():
                self._update_toggle_button_recursive(child, "▶")
        else:
            self.hyperparams_container.pack(fill="x", padx=10, pady=(0, 10))
            self.show_hyperparams.set(True)
            # Update toggle button text
            for child in self.dialog.winfo_children():
                self._update_toggle_button_recursive(child, "▼")

    def _update_toggle_button_recursive(self, widget, text):
        """Recursively find and update the toggle button text."""
        try:
            if isinstance(widget, ctk.CTkButton) and widget.cget("width") == 30:
                widget.configure(text=text)
                return
            for child in widget.winfo_children():
                self._update_toggle_button_recursive(child, text)
        except:
            pass

    def _set_hyperparams_balanced(self):
        """⚖️ Balanced: Standard PPO settings for most games."""
        self.learning_rate_var.set("0.00025")
        self.batch_size_var.set("512")
        self.n_steps_var.set("256")
        self.ent_coef_var.set("0.01")
        self.gamma_var.set("0.99")
        self.gae_lambda_var.set("0.95")
        self.clip_range_var.set("0.1")
        self.vf_coef_var.set("0.5")
        self.max_grad_norm_var.set("0.5")

    def _set_hyperparams_explore(self):
        """🔍 Explore: High exploration for discovery (best for Breakout at +20 plateau)."""
        self.learning_rate_var.set("0.00025")
        self.batch_size_var.set("256")  # Smaller for more frequent updates
        self.n_steps_var.set("256")
        self.ent_coef_var.set("0.02")  # Doubled for more exploration
        self.gamma_var.set("0.99")
        self.gae_lambda_var.set("0.95")
        self.clip_range_var.set("0.1")
        self.vf_coef_var.set("0.5")
        self.max_grad_norm_var.set("0.5")

    def _set_hyperparams_exploit(self):
        """🎯 Exploit: Low exploration, refine learned behaviors."""
        self.learning_rate_var.set("0.0001")  # Lower for stability
        self.batch_size_var.set("1024")  # Larger for smoother gradients
        self.n_steps_var.set("256")
        self.ent_coef_var.set("0.005")  # Reduced exploration
        self.gamma_var.set("0.99")
        self.gae_lambda_var.set("0.95")
        self.clip_range_var.set("0.1")
        self.vf_coef_var.set("0.5")
        self.max_grad_norm_var.set("0.5")

    def _set_hyperparams_fast(self):
        """⚡ Fast: Aggressive learning for quick iteration."""
        self.learning_rate_var.set("0.0005")  # 2x higher
        self.batch_size_var.set("128")  # Small for rapid updates
        self.n_steps_var.set("128")  # Smaller rollouts
        self.ent_coef_var.set("0.01")
        self.gamma_var.set("0.99")
        self.gae_lambda_var.set("0.95")
        self.clip_range_var.set("0.2")  # Larger for bigger updates
        self.vf_coef_var.set("0.5")
        self.max_grad_norm_var.set("1.0")  # Allow larger gradients

    def _set_hyperparams_stable(self):
        """🛡️ Stable: Conservative settings for stable training."""
        self.learning_rate_var.set("0.0001")  # Conservative
        self.batch_size_var.set("512")
        self.n_steps_var.set("512")  # Longer rollouts
        self.ent_coef_var.set("0.01")
        self.gamma_var.set("0.99")
        self.gae_lambda_var.set("0.95")
        self.clip_range_var.set("0.05")  # Smaller for stability
        self.vf_coef_var.set("0.5")
        self.max_grad_norm_var.set("0.3")  # Clip gradients more aggressively

    def _get_hyperparameters(self):
        """Get hyperparameters from UI as a dictionary."""
        try:
            return {
                'learning_rate': float(self.learning_rate_var.get()),
                'batch_size': int(self.batch_size_var.get()),
                'n_steps': int(self.n_steps_var.get()),
                'ent_coef': float(self.ent_coef_var.get()),
                'gamma': float(self.gamma_var.get()),
                'gae_lambda': float(self.gae_lambda_var.get()),
                'clip_range': float(self.clip_range_var.get()),
                'vf_coef': float(self.vf_coef_var.get()),
                'max_grad_norm': float(self.max_grad_norm_var.get())
            }
        except ValueError as e:
            raise ValueError(f"Invalid hyperparameter value: {e}")

    def _set_hyperparameters(self, hyperparams: Dict):
        """Set hyperparameters in UI from a dictionary."""
        if not hyperparams:
            return

        if 'learning_rate' in hyperparams:
            self.learning_rate_var.set(str(hyperparams['learning_rate']))
        if 'batch_size' in hyperparams:
            self.batch_size_var.set(str(hyperparams['batch_size']))
        if 'n_steps' in hyperparams:
            self.n_steps_var.set(str(hyperparams['n_steps']))
        if 'ent_coef' in hyperparams:
            self.ent_coef_var.set(str(hyperparams['ent_coef']))
        if 'gamma' in hyperparams:
            self.gamma_var.set(str(hyperparams['gamma']))
        if 'gae_lambda' in hyperparams:
            self.gae_lambda_var.set(str(hyperparams['gae_lambda']))
        if 'clip_range' in hyperparams:
            self.clip_range_var.set(str(hyperparams['clip_range']))
        if 'vf_coef' in hyperparams:
            self.vf_coef_var.set(str(hyperparams['vf_coef']))
        if 'max_grad_norm' in hyperparams:
            self.max_grad_norm_var.set(str(hyperparams['max_grad_norm']))

    def _lock_hyperparameters(self, locked: bool):
        """Lock or unlock hyperparameter controls and custom name."""
        state = "disabled" if locked else "normal"

        # Disable/enable all hyperparameter entry fields
        for entry in self.hyperparam_entries:
            entry.configure(state=state)

        # Disable/enable preset buttons
        for btn in self.hyperparam_preset_buttons:
            btn.configure(state=state)

        # Disable/enable custom name field
        self.name_entry.configure(state=state)

        # Show/hide locked warning label
        if locked:
            self.hyperparam_locked_label.grid(row=7, column=0, columnspan=4, pady=(10, 5))
        else:
            self.hyperparam_locked_label.grid_forget()

    def _on_system_changed(self, system_name):
        """Handle gaming system selection change."""
        systems = self._get_game_systems()
        games = list(systems.get(system_name, {}).keys())

        # Update game dropdown
        self.game_combo.configure(values=games)
        if games:
            self.game_var.set(games[0])

        self._update_summary()

    def _on_game_changed(self, game_name):
        """Handle game selection change."""
        # Update available previous runs when game changes
        if self.run_mode_var.get() == "continue":
            self._update_previous_runs()
        self._update_summary()

    def _on_run_mode_changed(self):
        """Handle run mode selection change."""
        if self.run_mode_var.get() == "continue":
            # Show previous runs dropdown
            self.previous_runs_frame.pack(fill="x", padx=10, pady=(0, 10))
            self._update_previous_runs()
            # Lock hyperparameters to prevent accidental changes
            self._lock_hyperparameters(True)
        else:
            # Hide previous runs dropdown
            self.previous_runs_frame.pack_forget()
            # Generate new run ID
            self.run_id_var.set(generate_run_id())
            # Unlock hyperparameters for new runs
            self._lock_hyperparameters(False)

    def _update_previous_runs(self):
        """Update the list of previous runs for the selected game."""
        try:
            # Get the selected game environment ID
            system = self.system_var.get()
            game_display = self.game_var.get()
            systems = self._get_game_systems()
            game_env_id = systems.get(system, {}).get(game_display)

            if not game_env_id:
                self.previous_runs_combo.configure(values=["No game selected"])
                return

            # Get ML database from app
            if self.app and hasattr(self.app, 'process_manager') and hasattr(self.app.process_manager, 'ml_database'):
                ml_db = self.app.process_manager.ml_database

                # Get previous runs for this game, excluding active ones
                previous_runs = ml_db.get_runs_by_game(game_env_id, exclude_active=True)

                if previous_runs:
                    # Format run options to show human-friendly name plus run ID for disambiguation
                    run_options = []
                    self.run_data = {}  # Store run data for later use

                    for run in previous_runs:
                        # Calculate progress percentage
                        progress = 0
                        if run.config and run.config.total_timesteps:
                            progress = int((run.current_timestep / run.config.total_timesteps) * 100)

                        # Prefer custom/experiment name; fall back to run_id
                        name = getattr(run, 'display_name', None) or run.custom_name or run.experiment_name or run.run_id
                        run_id_short = run.run_id[:8] + "..."

                        # Format display string
                        reward_str = f"{run.best_reward:.1f}" if run.best_reward else "N/A"
                        display = f"{name} ({run_id_short}) - {run.status} - {progress}% - Reward: {reward_str}"
                        run_options.append(display)
                        self.run_data[display] = run

                    self.previous_runs_combo.configure(values=run_options)
                    if run_options:
                        self.previous_runs_var.set(run_options[0])
                        self._on_previous_run_selected(run_options[0])
                else:
                    self.previous_runs_combo.configure(values=["No previous runs for this game"])
                    self.run_info_label.configure(text="")
            else:
                self.previous_runs_combo.configure(values=["Database not available"])

        except Exception as e:
            print(f"Error updating previous runs: {e}")
            self.previous_runs_combo.configure(values=["Error loading runs"])

    def _on_previous_run_selected(self, run_display):
        """Handle previous run selection."""
        try:
            if hasattr(self, 'run_data') and run_display in self.run_data:
                run = self.run_data[run_display]

                # Update run ID
                self.run_id_var.set(run.run_id)

                # Load custom name from previous run
                if hasattr(run, 'custom_name') and run.custom_name:
                    self.custom_name_var.set(run.custom_name)

                # Load hyperparameters from previous run
                if run.config:
                    hyperparams = {
                        'learning_rate': run.config.learning_rate,
                        'batch_size': run.config.batch_size,
                        'n_steps': run.config.n_steps,
                        'ent_coef': run.config.ent_coef if run.config.ent_coef is not None else 0.01,
                        'gamma': run.config.gamma,
                        'gae_lambda': run.config.gae_lambda if run.config.gae_lambda is not None else 0.95,
                        'clip_range': run.config.clip_range if run.config.clip_range is not None else 0.1,
                        'vf_coef': run.config.vf_coef if run.config.vf_coef is not None else 0.5,
                        'max_grad_norm': run.config.max_grad_norm if run.config.max_grad_norm is not None else 0.5
                    }
                    self._set_hyperparameters(hyperparams)
                    print(f"✅ Loaded hyperparameters from previous run: {run.run_id}")

                # Display run information
                info_lines = []
                if run.start_time:
                    info_lines.append(f"Started: {run.start_time.strftime('%Y-%m-%d %H:%M')}")
                if run.current_timestep:
                    info_lines.append(f"Timesteps: {run.current_timestep:,}")
                if run.best_reward is not None:
                    info_lines.append(f"Best Reward: {run.best_reward:.2f}")

                info_text = " | ".join(info_lines) if info_lines else "No additional info"
                self.run_info_label.configure(text=info_text)
            else:
                self.run_info_label.configure(text="")
        except Exception as e:
            print(f"Error selecting previous run: {e}")
            self.run_info_label.configure(text="")



        self._update_summary()

    def _toggle_advanced(self):
        """Toggle advanced options visibility."""
        if self.show_advanced.get():
            self.advanced_container.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.advanced_container.pack_forget()

    def _update_summary(self):
        """Update the training summary display."""
        try:
            system = self.system_var.get()
            game_display = self.game_var.get()

            # Get video config
            video_config = self._get_video_length_config()

            # Format video length display
            hours = int(video_config['target_hours'])
            minutes = int((video_config['target_hours'] - hours) * 60)

            if hours > 0 and minutes > 0:
                length_str = f"{hours}h {minutes}m"
            elif hours > 0:
                length_str = f"{hours}h"
            else:
                length_str = f"{minutes}m"

            summary_text = (
                f"📋 {system} • {game_display} • {length_str} video • ~{video_config['timesteps']:,} steps"
            )

            self.summary_label.configure(text=summary_text)

        except Exception as e:
            self.summary_label.configure(text="Summary will appear here...")

    def _configure_resources(self):
        """Open resource configuration dialog."""
        try:
            # Get current resource configuration or defaults
            current_resources = self.resource_config or {
                'cpu_cores': 4,
                'memory_limit_gb': 16,
                'gpu_id': 'auto'
            }

            dialog = ResourceSelectorDialog(self.dialog, current_resources)
            result = dialog.show()

            if result:
                # Store resource configuration
                self.resource_config = result

                # Update status display
                cpu_cores = result.get('cpu_cores', 4)
                memory_gb = result.get('memory_limit_gb', 16)
                gpu_id = result.get('gpu_id', 'auto')

                status_text = f"CPU: {cpu_cores} cores, Memory: {memory_gb}GB, GPU: {gpu_id}"
                self.resource_status_label.configure(text=status_text)

                # Update summary
                self._update_summary()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open resource configuration: {e}")

    def _show_advanced_resources(self):
        """Show the advanced resource selection dialog."""
        current_resources = {
            'cpu_cores': int(self.cpu_cores_var.get()) if self.cpu_cores_var.get() else 4,
            'memory_limit_gb': float(self.memory_var.get()) if self.memory_var.get() else 16,
            'gpu_id': self.gpu_var.get()
        }

        dialog = ResourceSelectorDialog(self.dialog, current_resources)
        result = dialog.show()

        if result:
            # Update the dialog with selected values
            self.cpu_cores_var.set(str(result['cpu_cores']))
            self.memory_var.set(str(result.get('memory_limit_gb', 16)))
            self.gpu_var.set(str(result['gpu_id']))

    def _browse_output_path(self):
        """Browse for video output directory."""
        initial_dir = self.output_path_var.get() if self.output_path_var.get() else "D:\\"

        directory = filedialog.askdirectory(
            title="Select Video Output Directory",
            initialdir=initial_dir
        )

        if directory:
            self.output_path_var.set(directory)

    def _on_preset_changed(self, event=None):
        """Update default values when preset changes."""
        preset_name = self.preset_var.get()
        if preset_name in self.presets:
            preset = self.presets[preset_name]

            # Update training configuration
            if 'total_timesteps' in preset:
                self.timesteps_var.set(str(preset['total_timesteps']))
            if 'target_hours' in preset:
                self.target_hours_var.set(str(preset['target_hours']))
            if 'milestone_videos' in preset:
                self.milestone_videos_var.set(str(preset['milestone_videos']))

            # Update resource defaults
            defaults = preset.get('default_resources', {})
            if 'cpu_cores' in defaults:
                self.cpu_cores_var.set(str(defaults['cpu_cores']))
            if 'memory_limit_gb' in defaults:
                self.memory_var.set(str(defaults['memory_limit_gb']))
            if 'gpu_id' in defaults:
                self.gpu_var.set(defaults['gpu_id'])

    def _on_fast_mode_changed(self):
        """Handle Fast Mode toggle."""
        if self.fast_mode_var.get():
            # If Fast Mode is on, disable live recording since they are mutually exclusive
            self.training_video_var.set(False)
    
    def _start(self):
        """Start training with current configuration."""
        try:
            # Get selected system and game
            system = self.system_var.get()
            game_display = self.game_var.get()

            # Get actual game environment ID
            systems = self._get_game_systems()
            game_env_id = systems.get(system, {}).get(game_display, "ALE/Breakout-v5")

            # Get video length configuration
            video_config = self._get_video_length_config()

            # Get algorithm (clean up the display name)
            algorithm = self.algo_var.get().lower()  # "PPO" -> "ppo"

            # Determine if resuming from checkpoint and handle run naming
            run_mode = self.run_mode_var.get()
            resume_checkpoint = None
            custom_name = None
            leg_number = 1
            base_run_id = None
            branch_id = "main"
            root_name = None
            branch_token = "A"
            variant_index = 1
            old_run_id = None

            if run_mode == "continue":
                # Get the checkpoint path for the selected run
                old_run_id = self.run_id_var.get()
                checkpoint_path = Path(f"models/checkpoints/{old_run_id}/latest.zip")

                if checkpoint_path.exists():
                    resume_checkpoint = str(checkpoint_path)
                else:
                    messagebox.showerror("Error",
                                       f"Checkpoint not found for run {old_run_id}.\n"
                                       f"Expected: {checkpoint_path}")
                    return

                # Get previous run info to increment leg and track starting timestep
                leg_start_timestep = 0
                if self.app and hasattr(self.app, 'process_manager') and hasattr(self.app.process_manager, 'ml_database'):
                    ml_db = self.app.process_manager.ml_database
                    conn = ml_db.connection
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT custom_name, leg_number, base_run_id, current_timestep, branch_id, target_timestep,
                               root_name, branch_token
                        FROM experiment_runs
                        WHERE run_id = ?
                        """,
                        (old_run_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        custom_name = row[0]
                        old_leg = row[1] if row[1] else 1
                        leg_number = old_leg + 1
                        base_run_id = row[2] if row[2] else old_run_id  # Track original run
                        leg_start_timestep = row[3] if row[3] else 0  # Capture where this leg starts
                        branch_id = row[4] if row[4] else "main"
                        root_name = row[6] if row[6] else custom_name
                        branch_token = row[7] if row[7] else "A"
                    # Optional branch mode support (future): allocate next token
                    if run_mode == "branch" and self.app and hasattr(self.app, 'process_manager') and hasattr(self.app.process_manager, 'ml_database'):
                        existing_tokens = ml_db.get_branch_tokens(base_run_id or old_run_id)
                        branch_token = next_branch_token(existing_tokens)

                # Generate new run_id for this leg
                run_id = generate_run_id()
            else:
                # New run - use custom name if provided
                custom_name = self.custom_name_var.get().strip() if self.custom_name_var.get() else None
                root_name = custom_name or f"{algorithm}-{game_display}"
                leg_number = 1
                leg_start_timestep = 0  # First leg starts at 0
                run_id = generate_run_id()
                base_run_id = run_id  # First leg tracks itself
                branch_token = "A"

            # Format video length for display
            hours = int(video_config['target_hours'])
            minutes = int((video_config['target_hours'] - hours) * 60)
            if hours > 0 and minutes > 0:
                video_length_display = f"{hours}h {minutes}m"
            elif hours > 0:
                video_length_display = f"{hours}h"
            else:
                video_length_display = f"{minutes}m"

            # Get hyperparameters from UI
            hyperparameters = self._get_hyperparameters()

            # Compute absolute target timestep for lineage tracking
            target_timestep = leg_start_timestep + video_config['timesteps']

            # Resolve naming components and allocate variant index
            if not root_name:
                root_name = custom_name or base_run_id or run_id
            if not branch_token:
                branch_token = "A"

            variant_index = 1
            if self.app and hasattr(self.app, 'process_manager') and hasattr(self.app.process_manager, 'ml_database'):
                ml_db = self.app.process_manager.ml_database
                variant_index = ml_db.allocate_variant_index(
                    base_run_id=base_run_id or run_id,
                    branch_token=branch_token,
                    leg_number=leg_number
                )

            display_name = build_display_name(root_name, leg_number, branch_token, variant_index)

            # Build result configuration
            self.result = {
                'preset': 'custom',  # Always use custom for simple interface
                'game': game_env_id,
                'game_display': game_display,
                'system': system,
                'algorithm': algorithm,
                'run_id': run_id,
                'custom_name': custom_name,
                'leg_number': leg_number,
                'base_run_id': base_run_id,
                'root_name': root_name,
                'display_name': display_name,
                'branch_token': branch_token,
                'variant_index': variant_index,
                'leg_index': leg_number,  # keep leg index aligned with leg number (1-based)
                'branch_id': branch_id,
                'parent_run_id': old_run_id if run_mode == "continue" else None,
                'parent_checkpoint_path': resume_checkpoint if run_mode == "continue" else None,
                'leg_start_timestep': leg_start_timestep,  # Track where this leg starts
                'start_timestep': leg_start_timestep,
                'target_timestep': target_timestep,
                'total_timesteps': video_config['timesteps'],
                'target_hours': video_config['target_hours'],
                'milestone_videos': video_config['milestones'],
                'output_path': self.output_path_var.get(),
                'video_length_option': video_length_display,
                'priority': 'normal',
                'resume_checkpoint': resume_checkpoint,  # Add checkpoint path
                'run_mode': run_mode,  # Add run mode
                # Use optimized resource configuration for faster training
                'cpu_cores': 12,  # Increased from 4 to 12 cores
                'memory_limit_gb': 16,
                'gpu_id': 'auto',
                # Add hyperparameters
                'hyperparameters': hyperparameters,
                'training_video_enabled': bool(self.training_video_var.get()),
                'fast_mode': bool(self.fast_mode_var.get())
            }

            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
    
    def _cancel(self):
        """Cancel the dialog."""
        self.result = None
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result


if __name__ == "__main__":
    app = RetroMLSimple()
    app.run()
