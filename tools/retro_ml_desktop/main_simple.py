"""
Retro ML Desktop - Simple Process Manager (No Docker Required)
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import sys
import yaml
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.retro_ml_desktop.monitor import SystemMonitor, SystemMetrics, get_gpu_status_message
from tools.retro_ml_desktop.process_manager import ProcessManager, ProcessInfo, ResourceLimits, generate_run_id, get_available_cpus, get_available_gpus, get_detailed_cpu_info, get_detailed_gpu_info, get_recommended_resources
from tools.retro_ml_desktop.resource_selector import ResourceSelectorDialog
from tools.retro_ml_desktop.ram_cleaner import RAMCleanupDialog, get_memory_recommendations
from tools.retro_ml_desktop.video_player import VideoPlayerDialog, play_video_with_player, get_video_info
from tools.retro_ml_desktop.ml_database import MetricsDatabase
from tools.retro_ml_desktop.ml_collector import MetricsCollector
from tools.retro_ml_desktop.ml_dashboard import MLDashboard
from tools.retro_ml_desktop.cuda_diagnostics import CUDADiagnostics, create_user_friendly_error_message


class RetroMLSimple:
    """Simple ML training process manager - no Docker required."""
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize components
        self.project_root = project_root
        self.system_monitor = SystemMonitor(update_interval=2.0)
        self.process_manager = ProcessManager(str(project_root))

        # Initialize ML tracking system
        self.ml_database = MetricsDatabase(str(project_root / "ml_experiments.db"))
        self.ml_collector = MetricsCollector(self.ml_database)

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
            sidebar, text="🔍 CUDA Diagnostics", font=ctk.CTkFont(size=12),
            height=35, command=self._show_cuda_diagnostics,
            fg_color="#17a2b8", hover_color="#138496"
        )
        diagnostics_btn.pack(pady=(5, 20), padx=10, fill="x")
        
        # System info
        ctk.CTkLabel(sidebar, text="System Status:", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 5))
        
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
        
        # Available resources with detailed info
        ctk.CTkLabel(sidebar, text="System Resources:", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 5))

        # Get detailed resource information
        cpu_info = get_detailed_cpu_info()
        gpu_info = get_detailed_gpu_info()
        recommendations = get_recommended_resources()

        available_cpus = len([cpu for cpu in cpu_info if cpu.available])
        available_gpus = len([gpu for gpu in gpu_info if gpu.available])

        resources_text = (
            f"💻 CPU: {available_cpus}/{len(cpu_info)} cores available\n"
            f"🎮 GPU: {available_gpus}/{len(gpu_info)} devices available\n"
            f"🎯 Recommended: {recommendations['cpu_cores']} cores"
        )
        ctk.CTkLabel(sidebar, text=resources_text, justify="left").pack(pady=5, padx=10, anchor="w")

        # Advanced resource selector button
        advanced_resources_btn = ctk.CTkButton(
            sidebar, text="🔧 Advanced Resource Selection",
            command=self._show_resource_selector,
            font=ctk.CTkFont(size=12),
            height=30
        )
        advanced_resources_btn.pack(pady=5, padx=10, fill="x")

        # RAM cleanup button
        ram_cleanup_btn = ctk.CTkButton(
            sidebar, text="🧠 RAM Cleanup & Optimization",
            command=self._show_ram_cleanup,
            font=ctk.CTkFont(size=12),
            height=30
        )
        ram_cleanup_btn.pack(pady=5, padx=10, fill="x")

        # Training controls info
        controls_frame = ctk.CTkFrame(sidebar)
        controls_frame.pack(fill="x", padx=10, pady=(20, 5))

        ctk.CTkLabel(controls_frame, text="Training Controls",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))

        controls_text = (
            "• Stop: Gracefully terminate training\n"
            "• Pause: Suspend training (resume later)\n"
            "• Resume: Continue paused training\n"
            "• Clear Data: Delete training outputs\n"
            "• Remove: Remove from process list"
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
        self.ml_dashboard_tab = self.tabview.add("🧪 ML Dashboard")
        self.videos_tab = self.tabview.add("Video Gallery")
        self.logs_tab = self.tabview.add("Logs")

        # Setup each tab
        self._setup_processes_tab()
        self._setup_ml_dashboard_tab()
        self._setup_videos_tab()
        self._setup_logs_tab()
    
    def _setup_processes_tab(self):
        """Setup the processes tab with process list."""
        # Control buttons
        controls_frame = ctk.CTkFrame(self.processes_tab)
        controls_frame.pack(fill="x", padx=10, pady=5)

        refresh_btn = ctk.CTkButton(controls_frame, text="Refresh", command=self._refresh_processes)
        refresh_btn.pack(side="left", padx=5, pady=5)

        stop_btn = ctk.CTkButton(controls_frame, text="🛑 Stop Selected", command=self._stop_selected_process,
                                fg_color="#dc3545", hover_color="#c82333")
        stop_btn.pack(side="left", padx=5)

        pause_btn = ctk.CTkButton(controls_frame, text="⏸️ Pause Selected", command=self._pause_selected_process,
                                 fg_color="#ffc107", hover_color="#e0a800", text_color="black")
        pause_btn.pack(side="left", padx=5)

        resume_btn = ctk.CTkButton(controls_frame, text="▶️ Resume Selected", command=self._resume_selected_process,
                                  fg_color="#28a745", hover_color="#218838")
        resume_btn.pack(side="left", padx=5)

        remove_btn = ctk.CTkButton(controls_frame, text="Remove Selected", command=self._remove_selected_process)
        remove_btn.pack(side="left", padx=5, pady=5)

        # Clear data buttons
        clear_frame = ctk.CTkFrame(self.processes_tab)
        clear_frame.pack(fill="x", padx=10, pady=5)

        clear_selected_btn = ctk.CTkButton(clear_frame, text="Clear Selected Data",
                                         command=self._clear_selected_data, fg_color="orange")
        clear_selected_btn.pack(side="left", padx=5)

        clear_all_btn = ctk.CTkButton(clear_frame, text="Clear ALL Training Data",
                                    command=self._clear_all_data, fg_color="red")
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
        """Setup the comprehensive ML dashboard tab."""
        # Initialize the ML dashboard
        self.ml_dashboard = MLDashboard(
            parent_frame=self.ml_dashboard_tab,
            database=self.ml_database,
            collector=self.ml_collector
        )

    def _setup_videos_tab(self):
        """Setup the video gallery tab for viewing training videos."""
        # Main container
        main_frame = ctk.CTkFrame(self.videos_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="🎬 Training Video Gallery",
                                  font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(10, 20))

        # Controls frame
        controls_frame = ctk.CTkFrame(main_frame)
        controls_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Refresh videos button
        refresh_videos_btn = ctk.CTkButton(controls_frame, text="🔄 Refresh Videos",
                                         command=self._refresh_videos)
        refresh_videos_btn.pack(side="left", padx=5, pady=5)

        # Open video folder button
        open_folder_btn = ctk.CTkButton(controls_frame, text="📁 Open Video Folder",
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

        play_btn = ctk.CTkButton(action_frame, text="▶️ Play Video", command=self._play_selected_video,
                                fg_color="#28a745", hover_color="#218838")
        play_btn.pack(side="left", padx=5, pady=5)

        player_btn = ctk.CTkButton(action_frame, text="🎬 Video Player", command=self._open_video_player)
        player_btn.pack(side="left", padx=5, pady=5)

        preview_btn = ctk.CTkButton(action_frame, text="👁️ Quick Preview", command=self._preview_selected_video)
        preview_btn.pack(side="left", padx=5, pady=5)

        info_btn = ctk.CTkButton(action_frame, text="ℹ️ Video Info", command=self._show_video_info)
        info_btn.pack(side="left", padx=5, pady=5)

        delete_btn = ctk.CTkButton(action_frame, text="🗑️ Delete Video", command=self._delete_selected_video,
                                  fg_color="#dc3545", hover_color="#c82333")
        delete_btn.pack(side="right", padx=5, pady=5)

        # Auto-refresh videos when tab is opened
        self._refresh_videos()

    def _setup_logs_tab(self):
        """Setup the logs tab with log viewer."""
        # Log text area
        self.log_text = ctk.CTkTextbox(self.logs_tab, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Clear logs button
        clear_btn = ctk.CTkButton(self.logs_tab, text="Clear Logs", command=self._clear_logs)
        clear_btn.pack(pady=5)
    
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
                status_display = "🟢 Running"
            elif process.status == "paused":
                status_display = "⏸️ Paused"
            elif process.status == "stopped":
                status_display = "🔴 Stopped"
            elif process.status == "finished":
                status_display = "✅ Finished"
            elif process.status == "failed":
                status_display = "❌ Failed"

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

    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.delete("1.0", "end")

    def _append_log(self, message: str):
        """Append a message to the log display."""
        def update_ui():
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")

        self.root.after(0, update_ui)

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

        ctk.CTkLabel(training_frame, text="🧠 Training Progress",
                    font=ctk.CTkFont(weight="bold")).pack(pady=(5, 2))

        if progress_info['training']:
            training = progress_info['training']
            training_text = (
                f"📊 Timesteps: {training['current_steps']:,} / {training['total_steps']:,} "
                f"({training['progress_pct']:.1f}%)\n"
                f"⏱️ ETA: {training['eta']}\n"
                f"🎯 Current Reward: {training['current_reward']}\n"
                f"⚡ FPS: {training['fps']}\n"
                f"💾 Checkpoints: {training['checkpoints_saved']}"
            )
        else:
            training_text = "⏳ Training progress not available (starting up...)"

        ctk.CTkLabel(training_frame, text=training_text, justify="left").pack(pady=2, padx=10)

        # Video Generation Progress Section
        video_frame = ctk.CTkFrame(process_frame)
        video_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(video_frame, text="🎬 Video Generation Progress",
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

        ctk.CTkLabel(files_frame, text="📁 Output Files",
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

                # Calculate progress percentage
                if training_info['current_steps'] > 0 and training_info['total_steps'] > 0:
                    training_info['progress_pct'] = (training_info['current_steps'] / training_info['total_steps']) * 100

                    # Estimate time remaining (very rough)
                    if training_info['progress_pct'] > 1:  # At least 1% done
                        # This is a very rough estimate
                        training_info['eta'] = f"~{int((100 - training_info['progress_pct']) * 2)} minutes"

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
            self.video_tree.insert("", "end", values=(
                video['name'],
                video['type'],
                video['duration'],
                video['size'],
                video['created'],
                video['training_run']
            ), tags=(video['path'],))

    def _discover_videos(self):
        """Discover all video files from training runs."""
        videos = []

        try:
            # Get all processes to find their output paths
            processes = self.process_manager.get_processes()

            # Check each process's output directory
            for process in processes:
                output_paths = self.process_manager.get_process_output_paths(process.id)
                if output_paths:
                    videos.extend(self._scan_process_videos(process, output_paths))

            # Also check default outputs directory
            default_outputs = self.project_root / "outputs"
            if default_outputs.exists():
                for run_dir in default_outputs.iterdir():
                    if run_dir.is_dir():
                        videos.extend(self._scan_directory_videos(run_dir, run_dir.name))

        except Exception as e:
            print(f"Error discovering videos: {e}")

        return videos

    def _scan_process_videos(self, process, output_paths):
        """Scan videos for a specific process."""
        videos = []

        try:
            # Scan milestone videos
            milestones_path = output_paths.get('videos_milestones')
            if milestones_path and Path(milestones_path).exists():
                videos.extend(self._scan_video_directory(
                    Path(milestones_path), "Milestone", process.name
                ))

            # Scan evaluation videos
            eval_path = output_paths.get('videos_eval')
            if eval_path and Path(eval_path).exists():
                videos.extend(self._scan_video_directory(
                    Path(eval_path), "Evaluation", process.name
                ))

            # Scan hour/part videos
            parts_path = output_paths.get('videos_parts')
            if parts_path and Path(parts_path).exists():
                videos.extend(self._scan_video_directory(
                    Path(parts_path), "Hour", process.name
                ))

        except Exception as e:
            print(f"Error scanning process videos for {process.id}: {e}")

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

                        videos.append({
                            'name': file_path.name,
                            'path': str(file_path),
                            'type': video_type,
                            'duration': duration,
                            'size': f"{size_mb:.1f} MB",
                            'created': created.strftime("%Y-%m-%d %H:%M"),
                            'training_run': training_run
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

    def _play_selected_video(self):
        """Play the selected video in the default video player."""
        selection = self.video_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a video to play.")
            return

        item = selection[0]
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
    
    def _show_start_training_dialog(self):
        """Show the start training dialog."""
        dialog = StartTrainingDialog(self.root, self.presets, self.games, self.algorithms)
        result = dialog.show()
        
        if result:
            self._start_training_process(result)
    
    def _start_training_process(self, config: Dict):
        """Start a new training process with the given configuration."""
        try:
            # Handle both old preset-based config and new simple config
            if config.get('preset') == 'custom' or config.get('preset') not in self.presets:
                # Use simple interface configuration
                preset = {
                    'total_timesteps': config.get('total_timesteps', 4000000),
                    'vec_envs': 4,
                    'save_freq': 100000,
                    'extra_args': []
                }
            else:
                # Use traditional preset
                preset = self.presets[config['preset']]

            # Prepare resource limits
            resources = ResourceLimits(
                cpu_affinity=list(range(config.get('cpu_cores', 4))),
                memory_limit_gb=config.get('memory_limit_gb'),
                priority=config.get('priority', 'normal'),
                gpu_id=config.get('gpu_id') if config.get('gpu_id') != 'auto' else None
            )

            # Debug: Print configuration being used
            print(f"🔧 Starting training with config:")
            print(f"   Game: {config['game']}")
            print(f"   Algorithm: {config['algorithm']}")
            print(f"   Timesteps: {config.get('total_timesteps', preset['total_timesteps'])}")
            print(f"   Vec Envs: {preset['vec_envs']}")
            print(f"   Save Freq: {preset['save_freq']}")
            print(f"   Output Path: {config.get('output_path')}")

            # Create process
            process_id = self.process_manager.create_process(
                game=config['game'],
                algorithm=config['algorithm'],
                run_id=config.get('run_id'),
                total_timesteps=config.get('total_timesteps', preset['total_timesteps']),
                vec_envs=preset['vec_envs'],
                save_freq=preset['save_freq'],
                resources=resources,
                extra_args=preset.get('extra_args', []),
                custom_output_path=config.get('output_path')
            )

            # Create experiment run in ML database
            self._create_experiment_run(process_id, config, preset)

            # Start log streaming
            self.process_manager.start_log_stream(process_id, self._append_log)

            # Refresh process list
            self._refresh_processes()

            # Switch to logs tab
            self.tabview.set("Logs")

            # Create friendly log message
            game_display = config.get('game_display', config['game'])
            system = config.get('system', 'Unknown')
            video_length = config.get('video_length_option', 'Standard Training')

            self._append_log(f"🚀 Started AI Training:")
            self._append_log(f"   🕹️ System: {system}")
            self._append_log(f"   🎯 Game: {game_display}")
            self._append_log(f"   ⏱️ Length: {video_length}")
            self._append_log(f"   🤖 Algorithm: {config['algorithm'].upper()}")
            self._append_log(f"   🏷️ Run ID: {process_id}")
            self._append_log(f"   📁 Videos: {config.get('output_path', 'Default location')}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
            self._append_log(f"❌ Failed to start training: {e}")

    def _create_experiment_run(self, process_id: str, config: Dict, preset: Dict):
        """Create an experiment run in the ML database and start metrics collection."""
        try:
            from .ml_metrics import ExperimentRun, ExperimentConfig

            # Create experiment configuration
            experiment_config = ExperimentConfig(
                algorithm=config['algorithm'],
                policy_type="CnnPolicy",  # Default for Atari
                learning_rate=0.0003,  # Default PPO learning rate
                batch_size=256,  # Default batch size
                n_steps=128,  # Default n_steps
                gamma=0.99,  # Default discount factor
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

            # Create experiment run
            experiment_run = ExperimentRun(
                run_id=process_id,
                experiment_name=f"{config['algorithm']}-{config['game'].split('/')[-1]}",
                start_time=datetime.now(),
                status="running",
                config=experiment_config,
                description=f"Training {config['algorithm']} on {config['game']} for {config.get('total_timesteps', preset['total_timesteps']):,} timesteps",
                tags=[config['algorithm'], config['game'].split('/')[-1], "auto-generated"]
            )

            # Store in database
            success = self.ml_database.create_experiment_run(experiment_run)
            if success:
                print(f"✅ Created experiment run in ML database: {process_id}")

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
                print(f"✅ Started metrics collection for: {process_id}")
            else:
                print(f"❌ Failed to create experiment run in ML database: {process_id}")

        except Exception as e:
            print(f"❌ Error creating experiment run: {e}")
            self._append_log(f"Warning: Failed to create ML experiment tracking: {e}")

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
            # Show confirmation dialog
            from tools.retro_ml_desktop.close_confirmation_dialog import show_close_confirmation
            result = show_close_confirmation(self.root, running_processes)

            if result == "cancel":
                return  # Don't close
            elif result == "pause":
                self.pause_all_training()
            elif result == "stop":
                self.stop_all_training()

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
        self.dialog.title("🔍 CUDA Diagnostics & Troubleshooting")
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
        title_label = ctk.CTkLabel(main_frame, text="🔍 CUDA System Diagnostics",
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

        ctk.CTkLabel(config_frame, text="💡 Recommended Training Configuration",
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
        refresh_btn = ctk.CTkButton(button_frame, text="🔄 Refresh Diagnostics",
                                   command=self._refresh_diagnostics,
                                   fg_color="#28a745", hover_color="#218838")
        refresh_btn.pack(side="left", padx=(10, 5), pady=10)

        # Copy report button
        copy_btn = ctk.CTkButton(button_frame, text="📋 Copy Report",
                                command=self._copy_report,
                                fg_color="#17a2b8", hover_color="#138496")
        copy_btn.pack(side="left", padx=5, pady=10)

        # Close button
        close_btn = ctk.CTkButton(button_frame, text="✖️ Close",
                                 command=self.dialog.destroy,
                                 fg_color="#6c757d", hover_color="#5a6268")
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
    
    def __init__(self, parent, presets: Dict, games: List[str], algorithms: List[str]):
        self.parent = parent
        self.presets = presets
        self.games = games
        self.algorithms = algorithms
        self.result = None
        
        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("🎮 Start AI Training")
        self.dialog.geometry("600x750")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.resizable(True, True)
        self.dialog.minsize(550, 700)  # Ensure minimum size for button visibility

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (750 // 2)
        self.dialog.geometry(f"600x750+{x}+{y}")

        self._create_dialog_ui()
    
    def _create_dialog_ui(self):
        """Create the dialog UI."""
        # Main frame (simple, no scrolling for now)
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Title
        title_label = ctk.CTkLabel(main_frame, text="🎮 Start New AI Training",
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 20))

        # Gaming System Selection
        system_frame = ctk.CTkFrame(main_frame)
        system_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(system_frame, text="🕹️ Gaming System:",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(10, 5), padx=10)

        self.system_var = tk.StringVar(value="Atari")
        system_options = ["Atari", "Classic Control", "Box2D"]
        system_combo = ctk.CTkOptionMenu(system_frame, variable=self.system_var,
                                       values=system_options, command=self._on_system_changed)
        system_combo.pack(fill="x", padx=10, pady=(0, 10))

        # Game Selection
        game_frame = ctk.CTkFrame(main_frame)
        game_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(game_frame, text="🎯 Choose Your Game:",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(10, 5), padx=10)

        # Initialize game options based on system
        self.game_var = tk.StringVar()
        self.game_combo = ctk.CTkOptionMenu(game_frame, variable=self.game_var, values=[""])
        self.game_combo.pack(fill="x", padx=10, pady=(0, 10))

        # Video Length Selection (Compact)
        video_frame = ctk.CTkFrame(main_frame)
        video_frame.pack(fill="x", pady=(0, 15), padx=10)

        ctk.CTkLabel(video_frame, text="⏱️ Video Length:",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 5), padx=10)

        self.video_length_var = tk.StringVar(value="Standard Training (4 hours)")
        video_options = [
            "Quick Demo (30 min)",
            "Short Training (1 hour)",
            "Standard Training (4 hours)",
            "Epic Training (10 hours)",
            "Custom Length"
        ]
        video_combo = ctk.CTkOptionMenu(video_frame, variable=self.video_length_var,
                                      values=video_options, command=self._on_video_length_changed)
        video_combo.pack(fill="x", padx=10, pady=(0, 10))

        # Custom length entry (initially hidden)
        self.custom_length_frame = ctk.CTkFrame(video_frame)
        self.custom_hours_var = tk.StringVar(value="2")
        ctk.CTkLabel(self.custom_length_frame, text="Custom Hours:").pack(side="left", padx=(10, 5))
        custom_entry = ctk.CTkEntry(self.custom_length_frame, textvariable=self.custom_hours_var, width=100)
        custom_entry.pack(side="left", padx=(0, 10))

        # Initialize variables
        self.resource_config = None
        self.algo_var = tk.StringVar(value="PPO")
        self.run_id_var = tk.StringVar(value=generate_run_id())
        self.output_path_var = tk.StringVar(value="D:\\ML_Videos")

        # Action Buttons (right after video length)
        button_frame = ctk.CTkFrame(main_frame)
        button_frame.pack(fill="x", pady=(30, 20))

        # Button instructions
        button_info = ctk.CTkLabel(button_frame, text="Ready to start training?",
                                  font=ctk.CTkFont(size=14, weight="bold"))
        button_info.pack(pady=(15, 10))

        # Button container
        btn_container = ctk.CTkFrame(button_frame)
        btn_container.pack(fill="x", padx=20, pady=(0, 15))

        cancel_btn = ctk.CTkButton(btn_container, text="❌ Cancel", command=self._cancel,
                                  fg_color="#6c757d", hover_color="#5a6268",
                                  height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                  width=140)
        cancel_btn.pack(side="right", padx=(10, 0))

        start_btn = ctk.CTkButton(btn_container, text="🚀 Start AI Training", command=self._start,
                                 fg_color="#28a745", hover_color="#218838",
                                 height=50, font=ctk.CTkFont(size=14, weight="bold"),
                                 width=200)
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
                "🎯 Breakout": "ALE/Breakout-v5",
                "🏓 Pong": "ALE/Pong-v5",
                "👾 Space Invaders": "ALE/SpaceInvaders-v5",
                "🚀 Asteroids": "ALE/Asteroids-v5",
                "🎮 Pac-Man": "ALE/MsPacman-v5",
                "🏎️ Enduro": "ALE/Enduro-v5",
                "🎪 Freeway": "ALE/Freeway-v5",
                "⚔️ Seaquest": "ALE/Seaquest-v5"
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

    def _get_video_length_config(self, length_option):
        """Get training configuration for video length option."""
        configs = {
            "Quick Demo (30 min)": {
                "target_hours": 0.5,
                "timesteps": 1000000,
                "milestones": 3
            },
            "Short Training (1 hour)": {
                "target_hours": 1,
                "timesteps": 2000000,
                "milestones": 4
            },
            "Standard Training (4 hours)": {
                "target_hours": 4,
                "timesteps": 4000000,
                "milestones": 8
            },
            "Epic Training (10 hours)": {
                "target_hours": 10,
                "timesteps": 10000000,
                "milestones": 10
            },
            "Custom Length": {
                "target_hours": float(self.custom_hours_var.get()) if hasattr(self, 'custom_hours_var') else 2,
                "timesteps": int(float(self.custom_hours_var.get()) * 1000000) if hasattr(self, 'custom_hours_var') else 2000000,
                "milestones": max(3, int(float(self.custom_hours_var.get()))) if hasattr(self, 'custom_hours_var') else 4
            }
        }
        return configs.get(length_option, configs["Standard Training (4 hours)"])

    def _on_system_changed(self, system_name):
        """Handle gaming system selection change."""
        systems = self._get_game_systems()
        games = list(systems.get(system_name, {}).keys())

        # Update game dropdown
        self.game_combo.configure(values=games)
        if games:
            self.game_var.set(games[0])

        self._update_summary()

    def _on_video_length_changed(self, length_option):
        """Handle video length selection change."""
        if length_option == "Custom Length":
            self.custom_length_frame.pack(fill="x", padx=10, pady=(0, 10))
        else:
            self.custom_length_frame.pack_forget()

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
            length = self.video_length_var.get()

            # Get video config
            video_config = self._get_video_length_config(length)

            summary_text = (
                f"📋 {system} • {game_display} • {length} • {video_config['target_hours']} hours"
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
            video_config = self._get_video_length_config(self.video_length_var.get())

            # Get algorithm (clean up the display name)
            algorithm = self.algo_var.get().lower()  # "PPO" -> "ppo"

            # Build result configuration
            self.result = {
                'preset': 'custom',  # Always use custom for simple interface
                'game': game_env_id,
                'game_display': game_display,
                'system': system,
                'algorithm': algorithm,
                'run_id': self.run_id_var.get(),
                'total_timesteps': video_config['timesteps'],
                'target_hours': video_config['target_hours'],
                'milestone_videos': video_config['milestones'],
                'output_path': self.output_path_var.get(),
                'video_length_option': self.video_length_var.get(),
                'priority': 'normal',
                # Use default resource configuration
                'cpu_cores': 4,
                'memory_limit_gb': 16,
                'gpu_id': 'auto'
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
