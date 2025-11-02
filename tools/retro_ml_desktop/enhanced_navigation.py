#!/usr/bin/env python3
"""
Enhanced Navigation Toolbar for ML Plotting

Provides advanced navigation features including:
- Real-time coordinate display
- Enhanced zoom and pan controls
- Custom navigation tools
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import logging


class EnhancedNavigationToolbar(NavigationToolbar2Tk):
    """
    Enhanced navigation toolbar with coordinate display and custom tools.
    
    Features:
    - Real-time coordinate display on mouse move
    - Enhanced zoom controls with coordinate feedback
    - Custom navigation tools
    - Coordinate format customization
    """
    
    def __init__(self, canvas, window, plotter=None):
        """
        Initialize enhanced navigation toolbar.
        
        Args:
            canvas: Matplotlib canvas
            window: Parent window
            plotter: MLPlotter instance for accessing axes
        """
        super().__init__(canvas, window)
        self.plotter = plotter
        self.logger = logging.getLogger(__name__)
        
        # Coordinate display
        self.coord_label = None
        self.current_axes = None
        
        # Setup enhancements
        self._setup_coordinate_display()
        self._setup_mouse_tracking()
        
        self.logger.info("Enhanced navigation toolbar initialized")
    
    def _setup_coordinate_display(self):
        """Add coordinate display widget to toolbar."""
        # Create separator
        separator = ttk.Separator(self, orient='vertical')
        separator.pack(side=tk.LEFT, fill='y', padx=5)
        
        # Create coordinate label
        self.coord_label = tk.Label(
            self,
            text="Coordinates: (-, -)",
            fg='white',
            bg='#2b2b2b',
            font=('Consolas', 9),
            relief='sunken',
            padx=10,
            pady=2
        )
        self.coord_label.pack(side=tk.LEFT, padx=5)
        
        # Create axes indicator
        self.axes_label = tk.Label(
            self,
            text="Plot: -",
            fg='#888888',
            bg='#2b2b2b',
            font=('Consolas', 9),
            padx=5
        )
        self.axes_label.pack(side=tk.LEFT, padx=5)
    
    def _setup_mouse_tracking(self):
        """Setup mouse tracking for coordinate display."""
        # Connect to canvas mouse events
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('axes_enter_event', self._on_axes_enter)
        self.canvas.mpl_connect('axes_leave_event', self._on_axes_leave)
    
    def _on_mouse_move(self, event):
        """
        Handle mouse move event to update coordinate display.
        
        Args:
            event: Matplotlib mouse event
        """
        if event.inaxes:
            self.current_axes = event.inaxes
            x, y = event.xdata, event.ydata
            
            # Format coordinates based on magnitude
            if abs(x) < 0.01 or abs(x) > 10000:
                x_str = f"{x:.2e}"
            else:
                x_str = f"{x:.2f}"
            
            if abs(y) < 0.01 or abs(y) > 10000:
                y_str = f"{y:.2e}"
            else:
                y_str = f"{y:.2f}"
            
            # Update coordinate label
            self.coord_label.config(
                text=f"Coordinates: ({x_str}, {y_str})",
                fg='white'
            )
            
            # Update axes label
            axes_name = self._get_axes_name(event.inaxes)
            self.axes_label.config(
                text=f"Plot: {axes_name}",
                fg='#00ff00'
            )
        else:
            self.coord_label.config(
                text="Coordinates: (-, -)",
                fg='#888888'
            )
            self.axes_label.config(
                text="Plot: -",
                fg='#888888'
            )
    
    def _on_axes_enter(self, event):
        """
        Handle axes enter event.
        
        Args:
            event: Matplotlib axes event
        """
        self.current_axes = event.inaxes
        axes_name = self._get_axes_name(event.inaxes)
        self.axes_label.config(
            text=f"Plot: {axes_name}",
            fg='#00ff00'
        )
    
    def _on_axes_leave(self, event):
        """
        Handle axes leave event.
        
        Args:
            event: Matplotlib axes event
        """
        self.current_axes = None
        self.coord_label.config(
            text="Coordinates: (-, -)",
            fg='#888888'
        )
        self.axes_label.config(
            text="Plot: -",
            fg='#888888'
        )
    
    def _get_axes_name(self, axes):
        """
        Get the name of the axes from the plotter.
        
        Args:
            axes: Matplotlib axes object
            
        Returns:
            str: Name of the axes or 'Unknown'
        """
        if not self.plotter or not hasattr(self.plotter, 'axes'):
            return "Unknown"
        
        for name, ax in self.plotter.axes.items():
            if ax == axes:
                return name.capitalize()
        
        return "Unknown"
    
    def get_current_coordinates(self):
        """
        Get current mouse coordinates.
        
        Returns:
            tuple: (x, y, axes_name) or None if not over axes
        """
        if self.current_axes:
            axes_name = self._get_axes_name(self.current_axes)
            # Note: We don't have the exact coordinates here without an event
            # This would need to be called from within an event handler
            return (None, None, axes_name)
        return None
    
    def set_coordinate_format(self, format_func):
        """
        Set custom coordinate formatting function.
        
        Args:
            format_func: Function that takes (x, y) and returns formatted string
        """
        self.coordinate_formatter = format_func
        self.logger.info("Custom coordinate formatter set")

