"""
Recent Activity Widget - Displays recent training experiments with real-time updates.

This widget subscribes to the Event Bus and automatically updates when training
events occur (started, completed, failed, etc.).
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from typing import Optional

from tools.retro_ml_desktop.experiment_manager import ExperimentManager
from tools.retro_ml_desktop.metric_event_bus import get_event_bus, EventTypes


class RecentActivityWidget(ctk.CTkFrame):
    """Widget displaying recent training experiments with real-time updates."""

    def __init__(self, experiment_manager: Optional[ExperimentManager] = None, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.experiment_manager = experiment_manager
        self.event_bus = get_event_bus()
        self._experiment_rows = {}  # Maps experiment_id to tree item

        self._init_ui()
        self._subscribe_to_events()

        # Initial load
        if self.experiment_manager:
            self.refresh_experiments()

        # Auto-refresh every 30 seconds
        self._schedule_refresh()

    def _init_ui(self):
        """Initialize the user interface."""
        # Header frame
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(10, 5))

        # Title
        title_label = ctk.CTkLabel(
            header_frame,
            text="Recent Activity",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(side="left")

        # Filter dropdown
        filter_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        filter_frame.pack(side="right")

        ctk.CTkLabel(filter_frame, text="Filter:").pack(side="left", padx=(0, 5))
        self.filter_combo = ctk.CTkComboBox(
            filter_frame,
            values=["All", "Running", "Completed", "Failed", "Paused"],
            command=self._on_filter_changed,
            width=120
        )
        self.filter_combo.set("All")
        self.filter_combo.pack(side="left", padx=(0, 10))

        # Refresh button
        self.refresh_btn = ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self.refresh_experiments,
            width=80
        )
        self.refresh_btn.pack(side="left")

        # Table frame with scrollbar
        table_frame = ctk.CTkFrame(self)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Create Treeview for table
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Activity.Treeview",
            background="#2b2b2b",
            foreground="white",
            fieldbackground="#2b2b2b",
            borderwidth=0
        )
        style.configure(
            "Activity.Treeview.Heading",
            background="#1f538d",
            foreground="white",
            relief="flat"
        )
        style.map("Activity.Treeview", background=[('selected', '#144870')])

        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Tree widget
        self.tree = ttk.Treeview(
            table_frame,
            columns=("status", "game", "algorithm", "created", "preset"),
            show="headings",
            style="Activity.Treeview",
            yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.tree.yview)

        # Configure columns
        self.tree.heading("status", text="Status")
        self.tree.heading("game", text="Game")
        self.tree.heading("algorithm", text="Algorithm")
        self.tree.heading("created", text="Created")
        self.tree.heading("preset", text="Preset")

        self.tree.column("status", width=100, anchor="center")
        self.tree.column("game", width=200)
        self.tree.column("algorithm", width=100, anchor="center")
        self.tree.column("created", width=150, anchor="center")
        self.tree.column("preset", width=100, anchor="center")

        self.tree.pack(fill="both", expand=True)

        # Configure tags for status colors
        self.tree.tag_configure("running", background="#90EE90", foreground="black")
        self.tree.tag_configure("completed", background="#D3D3D3", foreground="black")
        self.tree.tag_configure("failed", background="#FFB6C1", foreground="black")
        self.tree.tag_configure("paused", background="#FFDAB9", foreground="black")
        self.tree.tag_configure("pending", background="#C8C8C8", foreground="black")

        # Bind right-click for context menu
        self.tree.bind("<Button-3>", self._show_context_menu)
        self.tree.bind("<Double-Button-1>", self._on_double_click)

    def set_experiment_manager(self, experiment_manager: ExperimentManager):
        """Set the experiment manager and refresh data."""
        self.experiment_manager = experiment_manager
        self.refresh_experiments()

    def _subscribe_to_events(self):
        """Subscribe to relevant Event Bus events."""
        # Subscribe to training lifecycle events
        self.event_bus.subscribe(EventTypes.TRAINING_STARTED, self._on_training_started)
        self.event_bus.subscribe(EventTypes.TRAINING_COMPLETE, self._on_training_complete)
        self.event_bus.subscribe(EventTypes.TRAINING_FAILED, self._on_training_failed)
        self.event_bus.subscribe(EventTypes.TRAINING_STOPPED, self._on_training_stopped)
        self.event_bus.subscribe(EventTypes.TRAINING_PAUSED, self._on_training_paused)
        self.event_bus.subscribe(EventTypes.TRAINING_RESUMED, self._on_training_resumed)

    def _on_training_started(self, data: dict):
        """Handle TRAINING_STARTED event."""
        # Schedule refresh on main thread
        self.after(100, self.refresh_experiments)

    def _on_training_complete(self, data: dict):
        """Handle TRAINING_COMPLETE event."""
        experiment_id = data.get('experiment_id')
        if experiment_id:
            self.after(100, lambda: self._update_experiment_status(experiment_id, 'completed'))

    def _on_training_failed(self, data: dict):
        """Handle TRAINING_FAILED event."""
        experiment_id = data.get('experiment_id')
        if experiment_id:
            self.after(100, lambda: self._update_experiment_status(experiment_id, 'failed'))

    def _on_training_stopped(self, data: dict):
        """Handle TRAINING_STOPPED event."""
        experiment_id = data.get('experiment_id')
        if experiment_id:
            self.after(100, lambda: self._update_experiment_status(experiment_id, 'paused'))

    def _on_training_paused(self, data: dict):
        """Handle TRAINING_PAUSED event."""
        experiment_id = data.get('experiment_id')
        if experiment_id:
            self.after(100, lambda: self._update_experiment_status(experiment_id, 'paused'))

    def _on_training_resumed(self, data: dict):
        """Handle TRAINING_RESUMED event."""
        experiment_id = data.get('experiment_id')
        if experiment_id:
            self.after(100, lambda: self._update_experiment_status(experiment_id, 'running'))

    def _update_experiment_status(self, experiment_id: str, new_status: str):
        """Update the status of an experiment in the table."""
        if experiment_id in self._experiment_rows:
            item_id = self._experiment_rows[experiment_id]
            # Update status column
            values = list(self.tree.item(item_id, "values"))
            if len(values) >= 5:
                values[0] = self._get_status_display(new_status)
                self.tree.item(item_id, values=values, tags=(new_status,))
        else:
            # Experiment not in table, refresh to add it
            self.refresh_experiments()

    def refresh_experiments(self):
        """Refresh the experiments list from the database."""
        if not self.experiment_manager:
            return

        # Get filter
        filter_text = self.filter_combo.get()
        status_filter = None if filter_text == "All" else filter_text.lower()

        # Query experiments
        experiments = self.experiment_manager.list_experiments(
            status=status_filter,
            limit=50
        )

        # Clear tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._experiment_rows.clear()

        # Populate tree
        for exp in experiments:
            values = (
                self._get_status_display(exp.status),
                self._format_game_name(exp.game),
                exp.algorithm,
                self._format_datetime(exp.created),
                exp.preset.capitalize()
            )

            item_id = self.tree.insert("", "end", values=values, tags=(exp.status,))
            self._experiment_rows[exp.id] = item_id

    def _get_status_display(self, status: str) -> str:
        """Get display text for status."""
        status_map = {
            'pending': 'Pending',
            'running': 'Running',
            'completed': 'Completed',
            'failed': 'Failed',
            'paused': 'Paused'
        }
        return status_map.get(status, status.capitalize())

    def _format_game_name(self, game: str) -> str:
        """Format game name for display."""
        # Remove ALE/ prefix and -v5 suffix
        name = game.replace('ALE/', '').replace('NoFrameskip-v4', '').replace('-v5', '')
        return name.strip()

    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime for display."""
        now = datetime.now()
        diff = now - dt

        if diff.days == 0:
            if diff.seconds < 60:
                return "Just now"
            elif diff.seconds < 3600:
                mins = diff.seconds // 60
                return f"{mins}m ago"
            else:
                hours = diff.seconds // 3600
                return f"{hours}h ago"
        elif diff.days == 1:
            return "Yesterday"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")

    def _on_filter_changed(self, value):
        """Handle filter dropdown change."""
        self.refresh_experiments()

    def _show_context_menu(self, event):
        """Show context menu for tree item."""
        # Select item under cursor
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)

            # Create context menu
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="View Details", command=self._view_details)
            menu.add_separator()
            menu.add_command(label="Refresh", command=self.refresh_experiments)

            # Show menu
            menu.post(event.x_root, event.y_root)

    def _on_double_click(self, event):
        """Handle double-click on tree item."""
        self._view_details()

    def _view_details(self):
        """View details of selected experiment."""
        selection = self.tree.selection()
        if selection:
            # TODO: Implement experiment details dialog
            print("View details not yet implemented")

    def _schedule_refresh(self):
        """Schedule next auto-refresh."""
        # Refresh every 30 seconds
        self.after(30000, self._auto_refresh)

    def _auto_refresh(self):
        """Auto-refresh callback."""
        self.refresh_experiments()
        self._schedule_refresh()

    def cleanup(self):
        """Cleanup resources before widget is destroyed."""
        # Unsubscribe from events
        try:
            self.event_bus.unsubscribe(EventTypes.TRAINING_STARTED, self._on_training_started)
            self.event_bus.unsubscribe(EventTypes.TRAINING_COMPLETE, self._on_training_complete)
            self.event_bus.unsubscribe(EventTypes.TRAINING_FAILED, self._on_training_failed)
            self.event_bus.unsubscribe(EventTypes.TRAINING_STOPPED, self._on_training_stopped)
            self.event_bus.unsubscribe(EventTypes.TRAINING_PAUSED, self._on_training_paused)
            self.event_bus.unsubscribe(EventTypes.TRAINING_RESUMED, self._on_training_resumed)
        except:
            pass  # Event bus might already be gone
