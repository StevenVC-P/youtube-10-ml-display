"""
Status Badge Widget for CustomTkinter

Displays training status with color coding and optional pulsing animation.
"""

import customtkinter as ctk
from typing import Literal


StatusType = Literal["running", "paused", "stopped", "completed", "failed", "idle"]


class StatusBadge(ctk.CTkFrame):
    """
    A status badge widget with color coding and optional animation.
    
    Features:
    - Color-coded status indicators
    - Status icons
    - Pulsing animation for "running" status
    - Compact design
    """
    
    # Status configurations
    STATUS_CONFIG = {
        "running": {
            "text": "● Running",
            "color": "#4CAF50",  # Green
            "text_color": "white",
            "pulse": True
        },
        "paused": {
            "text": "⏸ Paused",
            "color": "#FFC107",  # Yellow/Amber
            "text_color": "black",
            "pulse": False
        },
        "stopped": {
            "text": "⏹ Stopped",
            "color": "#9E9E9E",  # Gray
            "text_color": "white",
            "pulse": False
        },
        "completed": {
            "text": "✓ Completed",
            "color": "#2196F3",  # Blue
            "text_color": "white",
            "pulse": False
        },
        "failed": {
            "text": "✗ Failed",
            "color": "#F44336",  # Red
            "text_color": "white",
            "pulse": False
        },
        "idle": {
            "text": "○ Idle",
            "color": "#607D8B",  # Blue Gray
            "text_color": "white",
            "pulse": False
        }
    }
    
    def __init__(
        self,
        parent,
        status: StatusType = "idle",
        **kwargs
    ):
        """
        Initialize status badge.
        
        Args:
            parent: Parent widget
            status: Initial status
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(parent, **kwargs)
        
        self.current_status = status
        self.pulse_job = None
        self.pulse_alpha = 1.0
        self.pulse_direction = -1
        
        self._setup_ui()
        self.set_status(status)
    
    def _setup_ui(self):
        """Setup the UI components."""
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=6
        )
        self.status_label.pack(padx=8, pady=4)
    
    def set_status(self, status: StatusType):
        """
        Update the status.
        
        Args:
            status: New status to display
        """
        if status not in self.STATUS_CONFIG:
            status = "idle"
        
        self.current_status = status
        config = self.STATUS_CONFIG[status]
        
        # Update label
        self.status_label.configure(
            text=config["text"],
            fg_color=config["color"],
            text_color=config["text_color"]
        )
        
        # Handle pulsing animation
        if config["pulse"]:
            self._start_pulse()
        else:
            self._stop_pulse()
    
    def _start_pulse(self):
        """Start pulsing animation."""
        if self.pulse_job is None:
            self._pulse_step()
    
    def _stop_pulse(self):
        """Stop pulsing animation."""
        if self.pulse_job is not None:
            self.after_cancel(self.pulse_job)
            self.pulse_job = None
            self.pulse_alpha = 1.0
    
    def _pulse_step(self):
        """Execute one step of the pulse animation."""
        # Update alpha
        self.pulse_alpha += self.pulse_direction * 0.05
        
        # Reverse direction at bounds
        if self.pulse_alpha <= 0.6:
            self.pulse_alpha = 0.6
            self.pulse_direction = 1
        elif self.pulse_alpha >= 1.0:
            self.pulse_alpha = 1.0
            self.pulse_direction = -1
        
        # Apply alpha to color (simplified - just adjust brightness)
        config = self.STATUS_CONFIG[self.current_status]
        base_color = config["color"]
        
        # Schedule next step
        self.pulse_job = self.after(50, self._pulse_step)
    
    def get_status(self) -> StatusType:
        """
        Get the current status.
        
        Returns:
            Current status
        """
        return self.current_status
    
    def cleanup(self):
        """Cleanup resources before widget is destroyed."""
        self._stop_pulse()

