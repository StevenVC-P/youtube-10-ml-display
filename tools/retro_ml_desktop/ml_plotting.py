#!/usr/bin/env python3
"""
ML Plotting System

Real-time plotting capabilities for ML experiment visualization.
Integrates matplotlib with tkinter for interactive charts.

Enhanced with Sprint 1 features:
- Enhanced navigation toolbar with coordinate display
- Chart annotations with database persistence
- Chart state save/load functionality
- Multi-format export capabilities
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import logging

from .ml_database import MetricsDatabase
from .ml_metrics import TrainingMetrics, ExperimentRun
from .enhanced_navigation import EnhancedNavigationToolbar
from .chart_annotations import ChartAnnotationManager
from .chart_state import ChartStateManager
from .enhanced_export import EnhancedExportManager


class MLPlotter:
    """
    Advanced plotting system for ML experiment visualization.
    
    Features:
    - Real-time training curves
    - Multi-run comparison
    - Interactive zooming and panning
    - Customizable metrics selection
    - Export capabilities
    """
    
    def __init__(self, parent_frame, database: MetricsDatabase):
        """
        Initialize ML plotter.

        Args:
            parent_frame: Parent tkinter frame
            database: MetricsDatabase instance
        """
        self.parent = parent_frame
        self.database = database

        # Plotting configuration
        self.figure_size = (12, 8)
        self.dpi = 100
        self.style = 'dark_background'

        # Data
        self.selected_runs = set()
        self.current_metric = "episode_reward_mean"
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 seconds

        # UI components
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.axes = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Sprint 1 enhancements
        self.annotation_manager = None
        self.state_manager = None
        self.export_manager = None

        # Initialize plotting
        self._setup_plotting()
    
    def _setup_plotting(self):
        """Setup the matplotlib plotting interface."""
        # Set matplotlib style
        plt.style.use(self.style)

        # Create figure
        self.figure = Figure(figsize=self.figure_size, dpi=self.dpi, facecolor='#2b2b2b')

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self.parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Create toolbar frame
        toolbar_frame = tk.Frame(self.parent)
        toolbar_frame.pack(fill="x")

        # Create enhanced toolbar (Sprint 1 feature)
        self.toolbar = EnhancedNavigationToolbar(self.canvas, toolbar_frame, plotter=self)
        self.toolbar.update()

        # Initial plot setup
        self._create_subplots()

        # Initialize Sprint 1 managers
        self._init_sprint1_features()

        # Start auto-refresh if enabled
        if self.auto_refresh:
            self._schedule_refresh()

    def _init_sprint1_features(self):
        """Initialize Sprint 1 enhancement features."""
        try:
            # Initialize annotation manager
            self.annotation_manager = ChartAnnotationManager(self, self.database)

            # Initialize state manager
            self.state_manager = ChartStateManager(self, self.database)

            # Initialize export manager
            self.export_manager = EnhancedExportManager(self)

            self.logger.info("Sprint 1 features initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Sprint 1 features: {e}")
    
    def _create_subplots(self):
        """Create subplot layout for different metrics."""
        self.figure.clear()
        
        # Create 2x2 subplot grid
        self.axes = {
            'reward': self.figure.add_subplot(2, 2, 1),
            'loss': self.figure.add_subplot(2, 2, 2),
            'learning': self.figure.add_subplot(2, 2, 3),
            'system': self.figure.add_subplot(2, 2, 4)
        }
        
        # Configure axes
        self._configure_axes()
        
        # Adjust layout
        self.figure.tight_layout(pad=3.0)
        
        # Initial empty plots
        self._plot_empty_state()
    
    def _configure_axes(self):
        """Configure axes properties and styling."""
        # Reward plot
        self.axes['reward'].set_title('Episode Reward', fontsize=12, fontweight='bold', color='white')
        self.axes['reward'].set_xlabel('Timesteps', color='white')
        self.axes['reward'].set_ylabel('Reward', color='white')
        self.axes['reward'].grid(True, alpha=0.3)
        
        # Loss plot
        self.axes['loss'].set_title('Training Losses', fontsize=12, fontweight='bold', color='white')
        self.axes['loss'].set_xlabel('Timesteps', color='white')
        self.axes['loss'].set_ylabel('Loss', color='white')
        self.axes['loss'].grid(True, alpha=0.3)
        
        # Learning dynamics plot
        self.axes['learning'].set_title('Learning Dynamics', fontsize=12, fontweight='bold', color='white')
        self.axes['learning'].set_xlabel('Timesteps', color='white')
        self.axes['learning'].set_ylabel('Value', color='white')
        self.axes['learning'].grid(True, alpha=0.3)
        
        # System performance plot
        self.axes['system'].set_title('System Performance', fontsize=12, fontweight='bold', color='white')
        self.axes['system'].set_xlabel('Timesteps', color='white')
        self.axes['system'].set_ylabel('FPS / Utilization %', color='white')
        self.axes['system'].grid(True, alpha=0.3)
        
        # Style all axes
        for ax in self.axes.values():
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
    
    def _plot_empty_state(self):
        """Plot empty state with instructions."""
        for ax_name, ax in self.axes.items():
            ax.text(0.5, 0.5, f'No data to display\nSelect runs to visualize {ax_name} metrics',
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=10, color='gray',
                   bbox=dict(boxstyle='round', facecolor='#3b3b3b', alpha=0.8))
        
        self.canvas.draw()
    
    def update_selected_runs(self, run_ids: List[str]):
        """
        Update the selected runs for plotting.

        Args:
            run_ids: List of run IDs to plot
        """
        self.logger.info(f"MLPlotter: Updating selected runs: {run_ids}")
        self.selected_runs = set(run_ids)
        self.logger.info(f"MLPlotter: Selected runs set to: {self.selected_runs}")
        self._update_plots()
    
    def set_metric(self, metric: str):
        """
        Set the primary metric to display.
        
        Args:
            metric: Metric name to display
        """
        self.current_metric = metric
        self._update_plots()
    
    def _update_plots(self):
        """Update all plots with current data."""
        self.logger.info(f"MLPlotter: _update_plots called with selected_runs: {self.selected_runs}")

        if not self.selected_runs:
            self.logger.info("MLPlotter: No selected runs, showing empty state")
            self._plot_empty_state()
            return

        self.logger.info(f"MLPlotter: Plotting data for {len(self.selected_runs)} runs")
        try:
            # Clear all axes
            for ax in self.axes.values():
                ax.clear()
            
            # Reconfigure axes
            self._configure_axes()
            
            # Plot data for each selected run
            colors = plt.cm.Set1(np.linspace(0, 1, len(self.selected_runs)))
            
            for i, run_id in enumerate(self.selected_runs):
                color = colors[i]
                self._plot_run_data(run_id, color)

            # Set fixed axis limits that start from (0,0) and only expand
            self._set_fixed_axis_limits()

            # Add legends
            for ax in self.axes.values():
                if ax.get_lines():
                    ax.legend(loc='upper left', fontsize=8)

            # Display annotations for selected runs (Sprint 1 feature)
            if self.annotation_manager:
                self.annotation_manager.display_annotations_for_runs(list(self.selected_runs))

            # Adjust layout and refresh
            self.figure.tight_layout(pad=3.0)
            self.canvas.draw()

        except Exception as e:
            self.logger.error(f"Failed to update plots: {e}")

    def _set_fixed_axis_limits(self):
        """Set fixed axis limits that start from (0,0) and only expand as needed."""
        try:
            # Get all data to determine max values
            all_timesteps = []
            all_rewards = []
            all_losses = []
            all_fps = []
            all_cpu = []
            all_lr = []
            all_kl = []

            for run_id in self.selected_runs:
                metrics = self.database.get_training_metrics(run_id)
                if not metrics:
                    continue

                timesteps = [m.timestep for m in metrics]
                all_timesteps.extend(timesteps)

                # Collect reward data
                rewards = [m.episode_reward_mean for m in metrics if m.episode_reward_mean is not None]
                all_rewards.extend(rewards)

                # Collect loss data
                policy_losses = [m.policy_loss for m in metrics if m.policy_loss is not None]
                value_losses = [m.value_loss for m in metrics if m.value_loss is not None]
                all_losses.extend(policy_losses + value_losses)

                # Collect system data
                fps_data = [m.fps for m in metrics if m.fps is not None]
                cpu_data = [m.cpu_percent for m in metrics if m.cpu_percent is not None]
                all_fps.extend(fps_data)
                all_cpu.extend(cpu_data)

                # Collect learning data
                lr_data = [m.learning_rate for m in metrics if m.learning_rate is not None]
                kl_data = [m.kl_divergence * 1000 for m in metrics if m.kl_divergence is not None]  # Scaled
                all_lr.extend(lr_data)
                all_kl.extend(kl_data)

            # Set X-axis limits (timesteps) - always start from 0
            if all_timesteps:
                max_timestep = max(all_timesteps)
                for ax in self.axes.values():
                    ax.set_xlim(0, max_timestep * 1.05)  # Add 5% padding

            # Set Y-axis limits for each plot - always start from appropriate minimum
            if all_rewards:
                min_reward = min(0, min(all_rewards))  # Start from 0 or lowest reward
                max_reward = max(all_rewards)
                self.axes['reward'].set_ylim(min_reward, max_reward * 1.1)

            if all_losses:
                self.axes['loss'].set_ylim(0, max(all_losses) * 1.1)  # Losses start from 0

            if all_fps or all_cpu:
                max_system = max((max(all_fps) if all_fps else 0), (max(all_cpu) if all_cpu else 0))
                self.axes['system'].set_ylim(0, max_system * 1.1)  # System metrics start from 0

            if all_lr or all_kl:
                max_learning = max((max(all_lr) if all_lr else 0), (max(all_kl) if all_kl else 0))
                self.axes['learning'].set_ylim(0, max_learning * 1.1)  # Learning metrics start from 0

        except Exception as e:
            self.logger.error(f"Failed to set fixed axis limits: {e}")

    def _plot_run_data(self, run_id: str, color):
        """
        Plot data for a specific run.

        Args:
            run_id: Run ID to plot
            color: Color for this run's plots
        """
        self.logger.info(f"MLPlotter: Plotting data for run {run_id}")

        # Get run info
        runs = self.database.get_experiment_runs()
        run = next((r for r in runs if r.run_id == run_id), None)

        if not run:
            self.logger.warning(f"MLPlotter: Run {run_id} not found in database")
            return

        # Get metrics
        metrics = self.database.get_training_metrics(run_id)
        self.logger.info(f"MLPlotter: Retrieved {len(metrics) if metrics else 0} metrics for run {run_id}")

        if not metrics:
            self.logger.warning(f"MLPlotter: No metrics found for run {run_id}")
            return
        
        # Extract data
        timesteps = [m.timestep for m in metrics]
        run_label = f"{run.config.algorithm}-{run.config.env_id.split('/')[-1]}" if run.config else run_id[:8]
        
        # Plot reward data
        rewards = [m.episode_reward_mean for m in metrics if m.episode_reward_mean is not None]
        reward_steps = [m.timestep for m in metrics if m.episode_reward_mean is not None]
        
        if rewards:
            self.axes['reward'].plot(reward_steps, rewards, color=color, label=run_label, linewidth=2)
            
            # Add moving average
            if len(rewards) > 10:
                window = min(50, len(rewards) // 4)
                moving_avg = self._calculate_moving_average(rewards, window)
                self.axes['reward'].plot(reward_steps, moving_avg, color=color, alpha=0.7, 
                                       linestyle='--', label=f'{run_label} (MA)')
        
        # Plot loss data
        policy_losses = [m.policy_loss for m in metrics if m.policy_loss is not None]
        value_losses = [m.value_loss for m in metrics if m.value_loss is not None]
        loss_steps = [m.timestep for m in metrics if m.policy_loss is not None or m.value_loss is not None]
        
        if policy_losses:
            self.axes['loss'].plot(loss_steps[:len(policy_losses)], policy_losses, 
                                 color=color, label=f'{run_label} Policy', linewidth=1.5)
        
        if value_losses:
            self.axes['loss'].plot(loss_steps[:len(value_losses)], value_losses, 
                                 color=color, alpha=0.7, linestyle=':', 
                                 label=f'{run_label} Value', linewidth=1.5)
        
        # Plot learning dynamics
        learning_rates = [m.learning_rate for m in metrics if m.learning_rate is not None]
        kl_divergences = [m.kl_divergence for m in metrics if m.kl_divergence is not None]
        
        if learning_rates:
            lr_steps = [m.timestep for m in metrics if m.learning_rate is not None]
            self.axes['learning'].plot(lr_steps, learning_rates, color=color, 
                                     label=f'{run_label} LR', linewidth=1.5)
        
        if kl_divergences:
            kl_steps = [m.timestep for m in metrics if m.kl_divergence is not None]
            # Scale KL divergence for better visualization
            scaled_kl = [kl * 1000 for kl in kl_divergences]  # Scale by 1000
            self.axes['learning'].plot(kl_steps, scaled_kl, color=color, alpha=0.7,
                                     linestyle='--', label=f'{run_label} KL×1000', linewidth=1.5)
        
        # Plot system performance
        fps_data = [m.fps for m in metrics if m.fps is not None]
        cpu_data = [m.cpu_percent for m in metrics if m.cpu_percent is not None]
        
        if fps_data:
            fps_steps = [m.timestep for m in metrics if m.fps is not None]
            self.axes['system'].plot(fps_steps, fps_data, color=color, 
                                   label=f'{run_label} FPS', linewidth=1.5)
        
        if cpu_data:
            cpu_steps = [m.timestep for m in metrics if m.cpu_percent is not None]
            self.axes['system'].plot(cpu_steps, cpu_data, color=color, alpha=0.7,
                                   linestyle=':', label=f'{run_label} CPU%', linewidth=1.5)
    
    def _calculate_moving_average(self, data: List[float], window: int) -> List[float]:
        """Calculate moving average with specified window size."""
        if len(data) < window:
            return data
        
        result = []
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            window_data = data[start_idx:i + 1]
            result.append(np.mean(window_data))
        
        return result
    
    def export_plot(self, filename: str, dpi: int = 300, format: str = None):
        """
        Export current plot to file using enhanced export manager.

        Args:
            filename: Output filename
            dpi: Resolution for export
            format: File format (png, svg, pdf) - auto-detected if None
        """
        if self.export_manager:
            # Use enhanced export manager (Sprint 1 feature)
            success = self.export_manager.export_chart(filename, format=format, dpi=dpi)
            if success:
                self.logger.info(f"Plot exported to {filename}")
            else:
                self.logger.error(f"Failed to export plot to {filename}")
        else:
            # Fallback to basic export
            try:
                self.figure.savefig(filename, dpi=dpi, bbox_inches='tight',
                                  facecolor='#2b2b2b', edgecolor='none')
                self.logger.info(f"Plot exported to {filename}")
            except Exception as e:
                self.logger.error(f"Failed to export plot: {e}")
    
    def set_auto_refresh(self, enabled: bool):
        """
        Enable or disable auto-refresh.
        
        Args:
            enabled: Whether to enable auto-refresh
        """
        self.auto_refresh = enabled
        if enabled:
            self._schedule_refresh()
    
    def _schedule_refresh(self):
        """Schedule automatic plot refresh."""
        if self.auto_refresh and self.selected_runs:
            self._update_plots()
        
        # Schedule next refresh
        if self.auto_refresh:
            self.parent.after(self.refresh_interval, self._schedule_refresh)
    
    def clear_plots(self):
        """Clear all plots."""
        self.selected_runs.clear()
        self._plot_empty_state()

    # Sprint 1 Enhancement Methods

    def add_annotation(self, run_id: str, axes_name: str, x: float, y: float,
                      text: str, color: str = 'yellow') -> Optional[str]:
        """
        Add annotation to chart (Sprint 1 feature).

        Args:
            run_id: Run ID to associate with
            axes_name: Axes name (reward, loss, learning, system)
            x: X-coordinate (timestep)
            y: Y-coordinate (metric value)
            text: Annotation text
            color: Annotation color

        Returns:
            str: Annotation ID if successful, None otherwise
        """
        if self.annotation_manager:
            return self.annotation_manager.add_annotation(run_id, axes_name, x, y, text, color)
        return None

    def save_chart_state(self, name: str, description: str = "") -> Optional[str]:
        """
        Save current chart state (Sprint 1 feature).

        Args:
            name: User-friendly name for the state
            description: Optional description

        Returns:
            str: State ID if successful, None otherwise
        """
        if self.state_manager:
            return self.state_manager.save_current_state(name, description)
        return None

    def load_chart_state(self, state_id: str) -> bool:
        """
        Load a saved chart state (Sprint 1 feature).

        Args:
            state_id: ID of state to load

        Returns:
            bool: True if successful, False otherwise
        """
        if self.state_manager:
            return self.state_manager.load_state(state_id)
        return False

    def list_chart_states(self):
        """
        Get list of all saved chart states (Sprint 1 feature).

        Returns:
            List of ChartState objects
        """
        if self.state_manager:
            return self.state_manager.list_states()
        return []

    def export_with_preset(self, filename: str, preset_name: str) -> bool:
        """
        Export chart using a preset configuration (Sprint 1 feature).

        Args:
            filename: Output filename
            preset_name: Name of preset to use

        Returns:
            bool: True if successful, False otherwise
        """
        if self.export_manager:
            return self.export_manager.export_with_preset(filename, preset_name)
        return False

    def batch_export(self, base_filename: str, formats: List[str] = None,
                    dpi: int = 300) -> Dict[str, bool]:
        """
        Export chart in multiple formats (Sprint 1 feature).

        Args:
            base_filename: Base filename (without extension)
            formats: List of formats to export
            dpi: Resolution in DPI

        Returns:
            Dict mapping format to success status
        """
        if self.export_manager:
            return self.export_manager.batch_export(base_filename, formats, dpi)
        return {}


class PlottingControls:
    """
    Control panel for ML plotting configuration.
    
    Provides UI controls for metric selection, run filtering,
    and plot customization.
    """
    
    def __init__(self, parent_frame, plotter: MLPlotter, database: MetricsDatabase):
        """
        Initialize plotting controls.
        
        Args:
            parent_frame: Parent tkinter frame
            plotter: MLPlotter instance
            database: MetricsDatabase instance
        """
        self.parent = parent_frame
        self.plotter = plotter
        self.database = database
        
        # Setup controls
        self._setup_controls()
    
    def _setup_controls(self):
        """Setup the control panel UI."""
        # Main controls frame
        controls_frame = tk.Frame(self.parent, bg='#2b2b2b')
        controls_frame.pack(fill="x", padx=5, pady=5)
        
        # Metric selection
        metric_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        metric_frame.pack(side="left", padx=5)
        
        tk.Label(metric_frame, text="Primary Metric:", fg='white', bg='#2b2b2b').pack(side="left")
        
        self.metric_var = tk.StringVar(value="episode_reward_mean")
        metric_options = ["episode_reward_mean", "policy_loss", "value_loss", "learning_rate", "fps"]
        
        metric_menu = ttk.Combobox(metric_frame, textvariable=self.metric_var, 
                                  values=metric_options, state="readonly", width=15)
        metric_menu.pack(side="left", padx=5)
        metric_menu.bind("<<ComboboxSelected>>", self._on_metric_change)
        
        # Auto-refresh toggle
        refresh_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        refresh_frame.pack(side="left", padx=20)
        
        self.auto_refresh_var = tk.BooleanVar(value=True)
        refresh_check = tk.Checkbutton(refresh_frame, text="Auto Refresh", 
                                     variable=self.auto_refresh_var,
                                     command=self._on_auto_refresh_change,
                                     fg='white', bg='#2b2b2b', selectcolor='#2b2b2b')
        refresh_check.pack(side="left")
        
        # Manual refresh button
        refresh_btn = tk.Button(refresh_frame, text="🔄 Refresh", 
                              command=self._manual_refresh,
                              bg='#4a4a4a', fg='white', relief='flat')
        refresh_btn.pack(side="left", padx=5)
        
        # Sprint 1 Enhancement Buttons

        # Save State button
        save_state_btn = tk.Button(controls_frame, text="💾 Save State",
                                  command=self._save_chart_state,
                                  bg='#4a4a4a', fg='white', relief='flat')
        save_state_btn.pack(side="right", padx=5)

        # Load State button
        load_state_btn = tk.Button(controls_frame, text="📂 Load State",
                                  command=self._load_chart_state,
                                  bg='#4a4a4a', fg='white', relief='flat')
        load_state_btn.pack(side="right", padx=5)

        # Export button (enhanced)
        export_btn = tk.Button(controls_frame, text="📊 Export Plot",
                             command=self._export_plot,
                             bg='#4a4a4a', fg='white', relief='flat')
        export_btn.pack(side="right", padx=5)

        # Clear button
        clear_btn = tk.Button(controls_frame, text="🗑️ Clear",
                            command=self._clear_plots,
                            bg='#4a4a4a', fg='white', relief='flat')
        clear_btn.pack(side="right", padx=5)
    
    def _on_metric_change(self, event):
        """Handle metric selection change."""
        metric = self.metric_var.get()
        self.plotter.set_metric(metric)
    
    def _on_auto_refresh_change(self):
        """Handle auto-refresh toggle."""
        enabled = self.auto_refresh_var.get()
        self.plotter.set_auto_refresh(enabled)
    
    def _manual_refresh(self):
        """Manually refresh plots."""
        self.plotter._update_plots()
    
    def _export_plot(self):
        """Export current plot with enhanced options (Sprint 1 feature)."""
        from tkinter import filedialog, simpledialog

        # Ask for export format
        format_choice = simpledialog.askstring(
            "Export Format",
            "Choose format:\n1. PNG (default)\n2. PDF\n3. SVG\n4. All formats\n\nEnter number (1-4):",
            initialvalue="1"
        )

        if not format_choice:
            return

        # Determine file types based on choice
        if format_choice == "1":
            filetypes = [("PNG files", "*.png")]
            default_ext = ".png"
        elif format_choice == "2":
            filetypes = [("PDF files", "*.pdf")]
            default_ext = ".pdf"
        elif format_choice == "3":
            filetypes = [("SVG files", "*.svg")]
            default_ext = ".svg"
        elif format_choice == "4":
            # Batch export
            filename = filedialog.asksaveasfilename(
                title="Export Plot (base filename)",
                defaultextension=""
            )
            if filename:
                # Remove extension if present
                from pathlib import Path
                base_filename = str(Path(filename).with_suffix(''))
                results = self.plotter.batch_export(base_filename)

                success_count = sum(1 for v in results.values() if v)
                from tkinter import messagebox
                messagebox.showinfo(
                    "Batch Export Complete",
                    f"Exported {success_count}/{len(results)} formats successfully"
                )
            return
        else:
            filetypes = [("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
            default_ext = ".png"

        filename = filedialog.asksaveasfilename(
            defaultextension=default_ext,
            filetypes=filetypes,
            title="Export Plot"
        )

        if filename:
            self.plotter.export_plot(filename)

    def _save_chart_state(self):
        """Save current chart state (Sprint 1 feature)."""
        from tkinter import simpledialog, messagebox

        name = simpledialog.askstring("Save Chart State", "Enter a name for this state:")
        if not name:
            return

        description = simpledialog.askstring("Save Chart State", "Enter a description (optional):")

        state_id = self.plotter.save_chart_state(name, description or "")
        if state_id:
            messagebox.showinfo("Success", f"Chart state '{name}' saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to save chart state")

    def _load_chart_state(self):
        """Load a saved chart state (Sprint 1 feature)."""
        from tkinter import simpledialog, messagebox

        # Get list of saved states
        states = self.plotter.list_chart_states()

        if not states:
            messagebox.showinfo("No States", "No saved chart states found")
            return

        # Create selection dialog
        state_list = "\n".join([f"{i+1}. {s.name} - {s.description}"
                               for i, s in enumerate(states)])

        choice = simpledialog.askstring(
            "Load Chart State",
            f"Available states:\n{state_list}\n\nEnter number to load:"
        )

        if not choice:
            return

        try:
            index = int(choice) - 1
            if 0 <= index < len(states):
                state = states[index]
                if self.plotter.load_chart_state(state.state_id):
                    messagebox.showinfo("Success", f"Loaded chart state '{state.name}'")
                else:
                    messagebox.showerror("Error", "Failed to load chart state")
            else:
                messagebox.showerror("Error", "Invalid selection")
        except ValueError:
            messagebox.showerror("Error", "Invalid input")

    def _clear_plots(self):
        """Clear all plots."""
        self.plotter.clear_plots()
