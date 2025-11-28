"""
Live Training Progress Widget - Real-time training status and metrics display.

Subscribes to TRAINING_PROGRESS events and displays:
- Progress bar with percentage
- Current timestep / total timesteps
- ETA to completion
- Latest metrics (reward, loss, speed)
"""

import customtkinter as ctk
from typing import Optional, Dict, Any
from datetime import timedelta

from tools.retro_ml_desktop.metric_event_bus import get_event_bus, EventTypes


class LiveProgressWidget(ctk.CTkFrame):
    """Widget displaying live training progress and metrics."""

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.event_bus = get_event_bus()
        self.current_experiment_id: Optional[str] = None
        self._latest_metrics: Dict[str, Any] = {}

        self._init_ui()
        self._subscribe_to_events()

    def _init_ui(self):
        """Initialize the user interface."""
        # Header
        header = ctk.CTkLabel(
            self,
            text="Live Training Progress",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        header.pack(pady=(10, 5), padx=10, anchor="w")

        # Progress section
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.pack(fill="x", padx=10, pady=5)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=400)
        self.progress_bar.pack(fill="x", pady=5)
        self.progress_bar.set(0)

        # Progress label (percentage and steps)
        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="No active training",
            font=ctk.CTkFont(size=12)
        )
        self.progress_label.pack(anchor="w")

        # ETA label
        self.eta_label = ctk.CTkLabel(
            progress_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.eta_label.pack(anchor="w")

        # Metrics section
        metrics_frame = ctk.CTkFrame(self, fg_color="transparent")
        metrics_frame.pack(fill="x", padx=10, pady=(10, 5))

        metrics_title = ctk.CTkLabel(
            metrics_frame,
            text="Latest Metrics:",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        metrics_title.pack(anchor="w", pady=(0, 5))

        # Metrics display (4 columns grid)
        self.metrics_grid = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        self.metrics_grid.pack(fill="x")

        # Create metric labels
        self.reward_label = self._create_metric_row("Mean Reward:", "--")
        self.policy_loss_label = self._create_metric_row("Policy Loss:", "--")
        self.value_loss_label = self._create_metric_row("Value Loss:", "--")
        self.speed_label = self._create_metric_row("Speed:", "--")

    def _create_metric_row(self, label_text: str, value_text: str):
        """Create a metric label row."""
        frame = ctk.CTkFrame(self.metrics_grid, fg_color="transparent")
        frame.pack(fill="x", pady=2)

        label = ctk.CTkLabel(
            frame,
            text=label_text,
            font=ctk.CTkFont(size=11),
            width=100,
            anchor="w"
        )
        label.pack(side="left", padx=(0, 10))

        value = ctk.CTkLabel(
            frame,
            text=value_text,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w"
        )
        value.pack(side="left")

        return value

    def _subscribe_to_events(self):
        """Subscribe to Event Bus events."""
        self.event_bus.subscribe(EventTypes.TRAINING_STARTED, self._on_training_started)
        self.event_bus.subscribe(EventTypes.TRAINING_PROGRESS, self._on_training_progress)
        self.event_bus.subscribe(EventTypes.TRAINING_COMPLETE, self._on_training_complete)
        self.event_bus.subscribe(EventTypes.TRAINING_FAILED, self._on_training_failed)
        self.event_bus.subscribe(EventTypes.TRAINING_STOPPED, self._on_training_stopped)

    def _on_training_started(self, data: dict):
        """Handle TRAINING_STARTED event."""
        self.current_experiment_id = data.get('experiment_id')
        self._latest_metrics = {}

        # Reset UI
        self.after(0, lambda: self._update_ui(
            progress_pct=0,
            timestep=0,
            total_timesteps=data.get('total_timesteps', 0),
            eta=None
        ))

    def _on_training_progress(self, data: dict):
        """Handle TRAINING_PROGRESS event."""
        experiment_id = data.get('experiment_id')

        # Only update if it's for the current experiment
        if experiment_id != self.current_experiment_id:
            return

        # Extract metrics
        metrics = data.get('metrics', {})
        self._latest_metrics.update(metrics)

        # Extract progress info
        progress_pct = data.get('progress_pct', 0)
        timestep = data.get('timestep', metrics.get('timesteps', 0))
        total_timesteps = data.get('total_timesteps', 0)
        eta = data.get('estimated_time_remaining')

        # Update UI on main thread
        self.after(0, lambda: self._update_ui(
            progress_pct=progress_pct,
            timestep=timestep,
            total_timesteps=total_timesteps,
            eta=eta
        ))

    def _on_training_complete(self, data: dict):
        """Handle TRAINING_COMPLETE event."""
        if data.get('experiment_id') == self.current_experiment_id:
            self.after(0, lambda: self._set_complete())

    def _on_training_failed(self, data: dict):
        """Handle TRAINING_FAILED event."""
        if data.get('experiment_id') == self.current_experiment_id:
            self.after(0, lambda: self._set_failed())

    def _on_training_stopped(self, data: dict):
        """Handle TRAINING_STOPPED event."""
        if data.get('experiment_id') == self.current_experiment_id:
            self.after(0, lambda: self._set_stopped())

    def _update_ui(self, progress_pct: float, timestep: int, total_timesteps: int, eta: Optional[float]):
        """Update the UI with new progress data."""
        # Update progress bar
        self.progress_bar.set(progress_pct / 100)

        # Update progress label
        if total_timesteps > 0:
            self.progress_label.configure(
                text=f"{progress_pct:.1f}% ({timestep:,} / {total_timesteps:,} timesteps)"
            )
        else:
            self.progress_label.configure(text=f"{progress_pct:.1f}%")

        # Update ETA
        if eta is not None and eta > 0:
            eta_str = self._format_time(eta)
            self.eta_label.configure(text=f"ETA: {eta_str}")
        else:
            self.eta_label.configure(text="")

        # Update metrics
        self._update_metrics()

    def _update_metrics(self):
        """Update metric displays from latest metrics."""
        # Mean reward
        if 'reward_mean' in self._latest_metrics:
            self.reward_label.configure(
                text=f"{self._latest_metrics['reward_mean']:.2f}"
            )
        else:
            self.reward_label.configure(text="--")

        # Policy loss
        if 'policy_loss' in self._latest_metrics:
            self.policy_loss_label.configure(
                text=f"{self._latest_metrics['policy_loss']:.4f}"
            )
        else:
            self.policy_loss_label.configure(text="--")

        # Value loss
        if 'value_loss' in self._latest_metrics:
            self.value_loss_label.configure(
                text=f"{self._latest_metrics['value_loss']:.2f}"
            )
        else:
            self.value_loss_label.configure(text="--")

        # Speed (FPS)
        if 'fps' in self._latest_metrics:
            self.speed_label.configure(
                text=f"{self._latest_metrics['fps']:.0f} steps/sec"
            )
        else:
            self.speed_label.configure(text="--")

    def _set_complete(self):
        """Set UI to completed state."""
        self.progress_bar.set(1.0)
        self.progress_label.configure(text="Training Complete!")
        self.eta_label.configure(text="")

    def _set_failed(self):
        """Set UI to failed state."""
        self.progress_label.configure(text="Training Failed", text_color="red")
        self.eta_label.configure(text="")

    def _set_stopped(self):
        """Set UI to stopped state."""
        self.progress_label.configure(text="Training Stopped", text_color="orange")
        self.eta_label.configure(text="")

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to human-readable string."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def cleanup(self):
        """Cleanup resources before widget is destroyed."""
        try:
            self.event_bus.unsubscribe(EventTypes.TRAINING_STARTED, self._on_training_started)
            self.event_bus.unsubscribe(EventTypes.TRAINING_PROGRESS, self._on_training_progress)
            self.event_bus.unsubscribe(EventTypes.TRAINING_COMPLETE, self._on_training_complete)
            self.event_bus.unsubscribe(EventTypes.TRAINING_FAILED, self._on_training_failed)
            self.event_bus.unsubscribe(EventTypes.TRAINING_STOPPED, self._on_training_stopped)
        except:
            pass  # Event bus might already be gone
