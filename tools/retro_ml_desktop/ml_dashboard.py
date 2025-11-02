#!/usr/bin/env python3
"""
ML Progress Dashboard

Comprehensive dashboard for ML scientists with:
- Multi-run comparison
- Real-time training curves
- Advanced analytics
- Experiment management
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import customtkinter as ctk
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
import logging

from .ml_database import MetricsDatabase
from .ml_collector import MetricsCollector
from .ml_metrics import ExperimentRun, TrainingMetrics, MetricsAggregator
from .ml_plotting import MLPlotter, PlottingControls


class MLDashboard:
    """
    Comprehensive ML experiment tracking dashboard.
    
    Features:
    - Multi-run overview with status tracking
    - Real-time training curves and metrics
    - Experiment comparison and analysis
    - Advanced filtering and search
    - Export and reporting capabilities
    """
    
    def __init__(self, parent_frame, database: MetricsDatabase, collector: MetricsCollector, process_manager=None):
        """
        Initialize ML dashboard.

        Args:
            parent_frame: Parent CTk frame
            database: MetricsDatabase instance
            collector: MetricsCollector instance
            process_manager: ProcessManager instance for log access
        """
        self.parent = parent_frame
        self.database = database
        self.collector = collector
        self.process_manager = process_manager
        
        # UI components
        self.main_frame = None
        self.runs_tree = None
        self.metrics_frame = None
        self.comparison_frame = None
        self.plotter = None
        self.plotting_controls = None
        
        # Data
        self.selected_runs = set()
        self.refresh_interval = 5000  # 5 seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI
        self._setup_dashboard()
        
        # Start auto-refresh
        self._schedule_refresh()
    
    def _setup_dashboard(self):
        """Setup the main dashboard layout."""
        # Main container
        self.main_frame = ctk.CTkFrame(self.parent)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create paned window for layout
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Left panel: Experiment runs
        self._setup_runs_panel()
        
        # Right panel: Metrics and analysis
        self._setup_metrics_panel()
    
    def _setup_runs_panel(self):
        """Setup the experiment runs panel."""
        # Left panel frame
        left_frame = ctk.CTkFrame(self.paned_window)
        self.paned_window.add(left_frame, weight=1)
        
        # Header
        header_frame = ctk.CTkFrame(left_frame)
        header_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(header_frame, text="🧪 Experiment Runs", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10, pady=5)
        
        # Control buttons
        controls_frame = ctk.CTkFrame(header_frame)
        controls_frame.pack(side="right", padx=5)
        
        ctk.CTkButton(controls_frame, text="🔄 Refresh", width=80,
                     command=self._refresh_runs).pack(side="left", padx=2)
        
        ctk.CTkButton(controls_frame, text="📊 Compare", width=80,
                     command=self._compare_selected).pack(side="left", padx=2)
        
        ctk.CTkButton(controls_frame, text="🗑️ Delete", width=80,
                     command=self._delete_selected).pack(side="left", padx=2)
        
        # Filter frame
        filter_frame = ctk.CTkFrame(left_frame)
        filter_frame.pack(fill="x", padx=5, pady=2)
        
        ctk.CTkLabel(filter_frame, text="Filter:").pack(side="left", padx=5)
        
        self.status_filter = ctk.CTkOptionMenu(filter_frame, 
                                              values=["All", "Running", "Completed", "Failed", "Paused"],
                                              command=self._filter_runs)
        self.status_filter.pack(side="left", padx=5)
        self.status_filter.set("All")
        
        # Runs tree
        tree_frame = ctk.CTkFrame(left_frame)
        tree_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create treeview with scrollbar
        tree_container = tk.Frame(tree_frame)
        tree_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.runs_tree = ttk.Treeview(tree_container, 
                                     columns=("status", "algorithm", "game", "progress", "reward", "duration"),
                                     show="tree headings",
                                     selectmode="extended")
        
        # Configure columns
        self.runs_tree.heading("#0", text="Run ID")
        self.runs_tree.heading("status", text="Status")
        self.runs_tree.heading("algorithm", text="Algorithm")
        self.runs_tree.heading("game", text="Game")
        self.runs_tree.heading("progress", text="Progress")
        self.runs_tree.heading("reward", text="Best Reward")
        self.runs_tree.heading("duration", text="Duration")
        
        # Column widths
        self.runs_tree.column("#0", width=120)
        self.runs_tree.column("status", width=80)
        self.runs_tree.column("algorithm", width=80)
        self.runs_tree.column("game", width=100)
        self.runs_tree.column("progress", width=80)
        self.runs_tree.column("reward", width=80)
        self.runs_tree.column("duration", width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=self.runs_tree.yview)
        self.runs_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.runs_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind selection event
        self.runs_tree.bind("<<TreeviewSelect>>", self._on_run_selection)
        
        # Summary stats frame
        stats_frame = ctk.CTkFrame(left_frame)
        stats_frame.pack(fill="x", padx=5, pady=5)
        
        self.stats_label = ctk.CTkLabel(stats_frame, text="📈 Total Runs: 0 | Active: 0 | Completed: 0")
        self.stats_label.pack(pady=5)
    
    def _setup_metrics_panel(self):
        """Setup the metrics and analysis panel."""
        # Right panel frame
        right_frame = ctk.CTkFrame(self.paned_window)
        self.paned_window.add(right_frame, weight=2)
        
        # Create tabview for different views
        self.metrics_tabview = ctk.CTkTabview(right_frame)
        self.metrics_tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add tabs
        self.metrics_tabview.add("📊 Training Curves")
        self.metrics_tabview.add("📈 Performance")
        self.metrics_tabview.add("🔍 Analysis")
        self.metrics_tabview.add("📝 Logs")
        self.metrics_tabview.add("⚙️ Details")

        # Setup each tab
        self._setup_training_curves_tab()
        self._setup_performance_tab()
        self._setup_analysis_tab()
        self._setup_logs_tab()
        self._setup_details_tab()
    
    def _setup_training_curves_tab(self):
        """Setup training curves visualization tab."""
        tab = self.metrics_tabview.tab("📊 Training Curves")

        # Create plotting controls frame
        controls_frame = tk.Frame(tab)
        controls_frame.pack(fill="x", padx=5, pady=5)

        # Create plotting frame
        plotting_frame = tk.Frame(tab)
        plotting_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Initialize plotter
        self.plotter = MLPlotter(plotting_frame, self.database)

        # Initialize plotting controls
        self.plotting_controls = PlottingControls(controls_frame, self.plotter, self.database)
    
    def _setup_performance_tab(self):
        """Setup performance metrics tab."""
        tab = self.metrics_tabview.tab("📈 Performance")
        
        # Current metrics frame
        current_frame = ctk.CTkFrame(tab)
        current_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(current_frame, text="🎯 Current Performance", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Metrics grid
        self.metrics_grid = ctk.CTkFrame(current_frame)
        self.metrics_grid.pack(fill="x", padx=10, pady=5)
        
        # Performance summary frame
        summary_frame = ctk.CTkFrame(tab)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(summary_frame, text="📊 Performance Summary", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Summary text
        self.summary_text = ctk.CTkTextbox(summary_frame, height=200)
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=5)
    
    def _setup_analysis_tab(self):
        """Setup analysis and comparison tab."""
        tab = self.metrics_tabview.tab("🔍 Analysis")
        
        # Comparison frame
        comparison_frame = ctk.CTkFrame(tab)
        comparison_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(comparison_frame, text="🔍 Experiment Analysis", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Analysis controls
        controls_frame = ctk.CTkFrame(comparison_frame)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(controls_frame, text="📊 Statistical Analysis",
                     command=self._run_statistical_analysis).pack(side="left", padx=5)
        
        ctk.CTkButton(controls_frame, text="🎯 Convergence Analysis",
                     command=self._run_convergence_analysis).pack(side="left", padx=5)
        
        ctk.CTkButton(controls_frame, text="📈 Sample Efficiency",
                     command=self._analyze_sample_efficiency).pack(side="left", padx=5)
        
        # Analysis results
        self.analysis_text = ctk.CTkTextbox(comparison_frame, height=300)
        self.analysis_text.pack(fill="both", expand=True, padx=10, pady=5)

    def _setup_logs_tab(self):
        """Setup logs display tab."""
        tab = self.metrics_tabview.tab("📝 Logs")

        # Header frame
        header_frame = ctk.CTkFrame(tab)
        header_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(header_frame, text="📝 Training Logs",
                    font=ctk.CTkFont(size=14, weight="bold")).pack(side="left", pady=5)

        # Controls
        controls_frame = ctk.CTkFrame(header_frame)
        controls_frame.pack(side="right", padx=5)

        ctk.CTkButton(controls_frame, text="🔄 Refresh", width=80,
                     command=self._refresh_logs).pack(side="left", padx=2)

        ctk.CTkButton(controls_frame, text="📋 Copy", width=80,
                     command=self._copy_logs).pack(side="left", padx=2)

        ctk.CTkButton(controls_frame, text="💾 Save", width=80,
                     command=self._save_logs).pack(side="left", padx=2)

        # Run selector frame (shows which run's logs are displayed)
        selector_frame = ctk.CTkFrame(tab)
        selector_frame.pack(fill="x", padx=10, pady=2)

        ctk.CTkLabel(selector_frame, text="Showing logs for:").pack(side="left", padx=5)
        self.current_log_run_label = ctk.CTkLabel(selector_frame, text="No run selected",
                                                 font=ctk.CTkFont(weight="bold"))
        self.current_log_run_label.pack(side="left", padx=5)

        # Auto-refresh toggle
        self.auto_refresh_logs = ctk.CTkCheckBox(selector_frame, text="Auto-refresh")
        self.auto_refresh_logs.pack(side="right", padx=5)
        self.auto_refresh_logs.select()  # Enable by default

        # Logs display frame
        logs_frame = ctk.CTkFrame(tab)
        logs_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Logs text widget with scrollbar
        self.logs_text = ctk.CTkTextbox(logs_frame, height=400, font=ctk.CTkFont(family="Consolas", size=10))
        self.logs_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Status frame
        status_frame = ctk.CTkFrame(tab)
        status_frame.pack(fill="x", padx=10, pady=2)

        self.logs_status_label = ctk.CTkLabel(status_frame, text="📊 Ready to display logs")
        self.logs_status_label.pack(side="left", padx=5, pady=2)

        # Initialize
        self.current_log_run_id = None

    def _setup_details_tab(self):
        """Setup experiment details tab."""
        tab = self.metrics_tabview.tab("⚙️ Details")
        
        # Details frame
        details_frame = ctk.CTkFrame(tab)
        details_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(details_frame, text="⚙️ Experiment Details", 
                    font=ctk.CTkFont(size=14, weight="bold")).pack(pady=5)
        
        # Details text
        self.details_text = ctk.CTkTextbox(details_frame, height=400)
        self.details_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Export controls
        export_frame = ctk.CTkFrame(tab)
        export_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(export_frame, text="📄 Export JSON",
                     command=self._export_json).pack(side="left", padx=5)
        
        ctk.CTkButton(export_frame, text="📊 Export CSV",
                     command=self._export_csv).pack(side="left", padx=5)
        
        ctk.CTkButton(export_frame, text="📋 Copy Config",
                     command=self._copy_config).pack(side="left", padx=5)
    
    def _refresh_runs(self):
        """Refresh the experiment runs list."""
        try:
            logging.info("Refreshing ML Dashboard runs display...")

            # Clear existing items
            for item in self.runs_tree.get_children():
                self.runs_tree.delete(item)

            # Get filter
            status_filter = self.status_filter.get()
            filter_status = None if status_filter == "All" else status_filter.lower()

            # Get runs from database
            runs = self.database.get_experiment_runs(status=filter_status)
            logging.info(f"Found {len(runs)} experiment runs in database")

            # Debug: Check if there are any metrics in the database at all
            if runs:
                for run in runs:
                    metrics_count = len(self.database.get_training_metrics(run.run_id))
                    logging.info(f"Run {run.run_id} has {metrics_count} total metrics in database")

            # Populate tree
            total_runs = len(runs)
            active_runs = sum(1 for r in runs if r.is_active)
            completed_runs = sum(1 for r in runs if r.is_completed)
            
            for run in runs:
                # Format values
                status_icon = self._get_status_icon(run.status)
                algorithm = run.config.algorithm if run.config else "Unknown"
                game = run.config.env_id.split('/')[-1] if run.config and run.config.env_id else "Unknown"
                
                # Get latest metrics for progress
                logging.info(f"Getting latest metrics for run: {run.run_id}")
                latest_metrics = self.database.get_latest_metrics(run.run_id)
                logging.info(f"Latest metrics for {run.run_id}: {latest_metrics}")

                if latest_metrics:
                    progress = f"{latest_metrics.progress_pct:.1f}%"
                    reward = f"{latest_metrics.episode_reward_mean:.2f}" if latest_metrics.episode_reward_mean else "N/A"
                    logging.info(f"Metrics found - Progress: {progress}, Reward: {reward}")
                else:
                    progress = "0.0%"
                    reward = "N/A"
                    logging.info(f"No metrics found for run {run.run_id}")
                
                # Duration
                duration = self._format_duration(run.duration)
                
                # Insert into tree
                self.runs_tree.insert("", "end", 
                                     text=run.run_id[:12] + "...",
                                     values=(f"{status_icon} {run.status.title()}", 
                                            algorithm, game, progress, reward, duration),
                                     tags=(run.run_id,))
            
            # Update stats
            self.stats_label.configure(text=f"📈 Total Runs: {total_runs} | Active: {active_runs} | Completed: {completed_runs}")
            
        except Exception as e:
            self.logger.error(f"Failed to refresh runs: {e}")
            messagebox.showerror("Error", f"Failed to refresh runs: {e}")
    
    def _get_status_icon(self, status: str) -> str:
        """Get icon for run status."""
        icons = {
            "running": "🟢",
            "completed": "✅",
            "failed": "❌",
            "paused": "⏸️",
            "stopped": "🔴"
        }
        return icons.get(status, "⚪")
    
    def _format_duration(self, duration_seconds: Optional[float]) -> str:
        """Format duration in human-readable format."""
        if not duration_seconds:
            return "N/A"
        
        duration = timedelta(seconds=int(duration_seconds))
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h"
        elif duration.seconds >= 3600:
            return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        else:
            return f"{duration.seconds//60}m"
    
    def _on_run_selection(self, event):
        """Handle run selection in tree."""
        selection = self.runs_tree.selection()
        if not selection:
            return
        
        # Get selected run IDs
        self.selected_runs = set()
        for item in selection:
            run_id = self.runs_tree.item(item)["tags"][0]
            self.selected_runs.add(run_id)
        
        # Update details for first selected run
        if self.selected_runs:
            first_run_id = next(iter(self.selected_runs))
            self._update_run_details(first_run_id)
            self._update_logs_display(first_run_id)

            # Update plotter with selected runs
            if self.plotter:
                self.plotter.update_selected_runs(list(self.selected_runs))
    
    def _update_run_details(self, run_id: str):
        """Update the details panel for selected run."""
        try:
            # Get run details
            runs = self.database.get_experiment_runs()
            run = next((r for r in runs if r.run_id == run_id), None)
            
            if not run:
                return
            
            # Update details text
            details = self._format_run_details(run)
            self.details_text.delete("1.0", "end")
            self.details_text.insert("1.0", details)
            
            # Update performance metrics
            self._update_performance_metrics(run_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update run details: {e}")
    
    def _format_run_details(self, run: ExperimentRun) -> str:
        """Format run details for display."""
        details = f"""🧪 Experiment: {run.experiment_name}
🆔 Run ID: {run.run_id}
📅 Started: {run.start_time.strftime('%Y-%m-%d %H:%M:%S')}
⏱️ Duration: {self._format_duration(run.duration)}
📊 Status: {run.status.title()}

🤖 Algorithm Configuration:
"""
        
        if run.config:
            details += f"""  • Algorithm: {run.config.algorithm}
  • Learning Rate: {run.config.learning_rate}
  • Batch Size: {run.config.batch_size}
  • Environment: {run.config.env_id}
  • Total Timesteps: {run.config.total_timesteps:,}
  • Vectorized Envs: {run.config.n_envs}
"""
        
        if run.description:
            details += f"\n📝 Description:\n{run.description}\n"
        
        if run.tags:
            details += f"\n🏷️ Tags: {', '.join(run.tags)}\n"
        
        return details
    
    def _update_performance_metrics(self, run_id: str):
        """Update performance metrics display."""
        try:
            # Get summary stats
            stats = self.database.get_run_summary_stats(run_id)
            
            if not stats:
                return
            
            # Clear existing metrics
            for widget in self.metrics_grid.winfo_children():
                widget.destroy()
            
            # Create metric cards
            row = 0
            col = 0
            
            metrics = [
                ("📊 Current Step", f"{stats['current_timestep']:,}"),
                ("🎯 Best Reward", f"{stats['reward_stats']['max']:.2f}" if stats['reward_stats'] else "N/A"),
                ("📈 Mean Reward", f"{stats['reward_stats']['mean']:.2f}" if stats['reward_stats'] else "N/A"),
                ("⚡ Avg FPS", f"{stats['fps_stats']['mean']:.0f}" if stats['fps_stats'] else "N/A"),
                ("📏 Total Metrics", f"{stats['total_metrics']:,}"),
                ("🎮 Status", stats['status'].title())
            ]
            
            for label, value in metrics:
                metric_frame = ctk.CTkFrame(self.metrics_grid)
                metric_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
                
                ctk.CTkLabel(metric_frame, text=label, font=ctk.CTkFont(size=10)).pack()
                ctk.CTkLabel(metric_frame, text=value, font=ctk.CTkFont(size=14, weight="bold")).pack()
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
            
            # Configure grid weights
            for i in range(3):
                self.metrics_grid.grid_columnconfigure(i, weight=1)
            
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")
    
    def _filter_runs(self, value):
        """Filter runs by status."""
        self._refresh_runs()
    
    def _compare_selected(self):
        """Compare selected runs."""
        if len(self.selected_runs) < 2:
            messagebox.showwarning("Selection Required", "Please select at least 2 runs to compare.")
            return
        
        # TODO: Implement comparison logic
        messagebox.showinfo("Coming Soon", "Multi-run comparison will be implemented with matplotlib visualization.")
    
    def _delete_selected(self):
        """Delete selected runs."""
        if not self.selected_runs:
            messagebox.showwarning("Selection Required", "Please select runs to delete.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Delete {len(self.selected_runs)} selected runs?"):
            for run_id in self.selected_runs:
                self.database.delete_experiment_run(run_id)
            
            self._refresh_runs()
            messagebox.showinfo("Success", f"Deleted {len(self.selected_runs)} runs.")
    
    def _update_plot(self):
        """Update training curves plot."""
        # TODO: Implement matplotlib plotting
        messagebox.showinfo("Coming Soon", "Real-time plotting will be implemented with matplotlib.")
    
    def _run_statistical_analysis(self):
        """Run statistical analysis on selected runs."""
        # TODO: Implement statistical analysis
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "📊 Statistical Analysis\n\nComing soon: Comprehensive statistical analysis including:\n• Performance comparison\n• Significance testing\n• Confidence intervals\n• Effect size analysis")
    
    def _run_convergence_analysis(self):
        """Run convergence analysis."""
        # TODO: Implement convergence analysis
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "🎯 Convergence Analysis\n\nComing soon: Convergence detection including:\n• Convergence point identification\n• Stability analysis\n• Learning curve smoothness\n• Plateau detection")
    
    def _analyze_sample_efficiency(self):
        """Analyze sample efficiency."""
        # TODO: Implement sample efficiency analysis
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "📈 Sample Efficiency Analysis\n\nComing soon: Sample efficiency metrics including:\n• Learning speed comparison\n• Data efficiency curves\n• Performance milestones\n• Resource utilization")
    
    def _export_json(self):
        """Export selected run to JSON."""
        # TODO: Implement JSON export
        messagebox.showinfo("Coming Soon", "JSON export functionality will be implemented.")
    
    def _export_csv(self):
        """Export selected run metrics to CSV."""
        # TODO: Implement CSV export
        messagebox.showinfo("Coming Soon", "CSV export functionality will be implemented.")
    
    def _copy_config(self):
        """Copy run configuration to clipboard."""
        # TODO: Implement config copying
        messagebox.showinfo("Coming Soon", "Configuration copying will be implemented.")

    def _update_logs_display(self, run_id: str):
        """Update the logs display for the selected run."""
        try:
            self.current_log_run_id = run_id
            self.current_log_run_label.configure(text=f"run-{run_id[-8:]}")  # Show last 8 chars

            # Get logs from process manager if available
            if self.process_manager:
                logs = self.process_manager.get_recent_logs(run_id)
                if logs:
                    # Clear and update logs display
                    self.logs_text.delete("1.0", "end")
                    self.logs_text.insert("1.0", logs)

                    # Auto-scroll to bottom
                    self.logs_text.see("end")

                    # Update status
                    lines_count = len(logs.split('\n'))
                    chars_count = len(logs)
                    self.logs_status_label.configure(
                        text=f"📊 Showing {lines_count} lines ({chars_count} chars) for {run_id[-8:]}"
                    )
                else:
                    self.logs_text.delete("1.0", "end")
                    self.logs_text.insert("1.0", f"No logs available for run {run_id}\n\nThis could mean:\n- The training process hasn't started yet\n- The process has finished\n- Logs are not being captured")
                    self.logs_status_label.configure(text="📊 No logs available")
            else:
                self.logs_text.delete("1.0", "end")
                self.logs_text.insert("1.0", "Process manager not available - cannot display logs")
                self.logs_status_label.configure(text="❌ Process manager unavailable")

        except Exception as e:
            self.logger.error(f"Failed to update logs display: {e}")
            self.logs_text.delete("1.0", "end")
            self.logs_text.insert("1.0", f"Error loading logs: {e}")
            self.logs_status_label.configure(text="❌ Error loading logs")

    def _refresh_logs(self):
        """Manually refresh the logs display."""
        if self.current_log_run_id:
            self._update_logs_display(self.current_log_run_id)
        else:
            messagebox.showinfo("No Run Selected", "Please select a run first to view its logs.")

    def _copy_logs(self):
        """Copy current logs to clipboard."""
        try:
            logs_content = self.logs_text.get("1.0", "end-1c")
            if logs_content.strip():
                self.parent.clipboard_clear()
                self.parent.clipboard_append(logs_content)
                messagebox.showinfo("Copied", "Logs copied to clipboard!")
            else:
                messagebox.showwarning("No Logs", "No logs to copy.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy logs: {e}")

    def _save_logs(self):
        """Save current logs to file."""
        try:
            if not self.current_log_run_id:
                messagebox.showwarning("No Run Selected", "Please select a run first.")
                return

            logs_content = self.logs_text.get("1.0", "end-1c")
            if not logs_content.strip():
                messagebox.showwarning("No Logs", "No logs to save.")
                return

            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                title="Save Logs",
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
                initialvalue=f"logs_{self.current_log_run_id}.log"
            )

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(logs_content)
                messagebox.showinfo("Saved", f"Logs saved to {filename}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs: {e}")

    def _schedule_refresh(self):
        """Schedule automatic refresh."""
        self._refresh_runs()

        # Auto-refresh logs if enabled and a run is selected
        if (hasattr(self, 'auto_refresh_logs') and
            self.auto_refresh_logs.get() and
            self.current_log_run_id):
            self._refresh_logs()

        self.parent.after(self.refresh_interval, self._schedule_refresh)
