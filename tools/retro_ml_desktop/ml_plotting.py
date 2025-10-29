#!/usr/bin/env python3
"""
ML Plotting System

Real-time plotting capabilities for ML experiment visualization.
Integrates matplotlib with tkinter for interactive charts.
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
        
        # Create toolbar
        toolbar_frame = tk.Frame(self.parent)
        toolbar_frame.pack(fill="x")
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Initial plot setup
        self._create_subplots()
        
        # Start auto-refresh if enabled
        if self.auto_refresh:
            self._schedule_refresh()
    
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
        self.selected_runs = set(run_ids)
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
        if not self.selected_runs:
            self._plot_empty_state()
            return
        
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
            
            # Add legends
            for ax in self.axes.values():
                if ax.get_lines():
                    ax.legend(loc='upper left', fontsize=8)
            
            # Adjust layout and refresh
            self.figure.tight_layout(pad=3.0)
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to update plots: {e}")
    
    def _plot_run_data(self, run_id: str, color):
        """
        Plot data for a specific run.
        
        Args:
            run_id: Run ID to plot
            color: Color for this run's plots
        """
        # Get run info
        runs = self.database.get_experiment_runs()
        run = next((r for r in runs if r.run_id == run_id), None)
        
        if not run:
            return
        
        # Get metrics
        metrics = self.database.get_training_metrics(run_id)
        
        if not metrics:
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
    
    def export_plot(self, filename: str, dpi: int = 300):
        """
        Export current plot to file.
        
        Args:
            filename: Output filename
            dpi: Resolution for export
        """
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
        
        # Export button
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
        """Export current plot."""
        from tkinter import filedialog
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
            title="Export Plot"
        )
        
        if filename:
            self.plotter.export_plot(filename)
    
    def _clear_plots(self):
        """Clear all plots."""
        self.plotter.clear_plots()
