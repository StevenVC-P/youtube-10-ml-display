#!/usr/bin/env python3
"""
Interactive Chart Tooltips Module

Provides interactive tooltip functionality for ML charts including:
- Hover tooltips showing exact metric values
- Crosshair cursor for precise reading
- Data point highlighting on hover
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


class InteractiveTooltip:
    """
    Manages interactive tooltips for matplotlib charts.

    Features:
    - Shows exact values on hover
    - Crosshair cursor for precise reading
    - Highlights nearest data point
    - Multi-line support for multiple series
    """

    def __init__(self, plotter):
        """
        Initialize interactive tooltip manager.

        Args:
            plotter: MLPlotter instance
        """
        self.plotter = plotter
        self.logger = logging.getLogger(__name__)

        # Tooltip state
        self.tooltip_annotation = None
        self.crosshair_lines = {}  # ax -> (vline, hline)
        self.highlight_points = {}  # ax -> Circle
        self.current_hover_data = {}  # ax -> (x, y, label)

        # Configuration
        self.enabled = True
        self.show_crosshair = True
        self.show_highlight = True

        # Connect events
        self._connect_events()

        self.logger.info("Interactive tooltip manager initialized")

    def _connect_events(self):
        """Connect matplotlib events for tooltip functionality."""
        if self.plotter.canvas:
            self.plotter.canvas.mpl_connect('motion_notify_event', self._on_hover)
            self.plotter.canvas.mpl_connect('axes_leave_event', self._on_leave)

    def _on_hover(self, event):
        """
        Handle mouse hover event.

        Args:
            event: Matplotlib mouse event
        """
        if not self.enabled or not event.inaxes:
            return

        ax = event.inaxes

        # Find nearest data point
        nearest_point = self._find_nearest_point(ax, event.xdata, event.ydata)

        if nearest_point:
            x, y, label, distance = nearest_point

            # Only show tooltip if mouse is close enough to a data point
            if distance < 50:  # pixels
                self._show_tooltip(ax, x, y, label, event)
                if self.show_crosshair:
                    self._show_crosshair(ax, x, y)
                if self.show_highlight:
                    self._highlight_point(ax, x, y)
            else:
                self._hide_tooltip(ax)
        else:
            self._hide_tooltip(ax)

    def _on_leave(self, event):
        """
        Handle mouse leave event.

        Args:
            event: Matplotlib axes leave event
        """
        if event.inaxes:
            self._hide_tooltip(event.inaxes)

    def _find_nearest_point(self, ax, x_mouse, y_mouse) -> Optional[Tuple[float, float, str, float]]:
        """
        Find the nearest data point to the mouse cursor.

        Args:
            ax: Matplotlib axes
            x_mouse: Mouse x coordinate in data space
            y_mouse: Mouse y coordinate in data space

        Returns:
            Tuple of (x, y, label, distance) for nearest point, or None
        """
        if not ax or x_mouse is None or y_mouse is None:
            return None

        lines = ax.get_lines()
        if not lines:
            return None

        min_distance = float('inf')
        nearest_point = None

        # Get display coordinates for mouse position
        mouse_display = ax.transData.transform([[x_mouse, y_mouse]])[0]

        for line in lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            label = line.get_label()

            if len(xdata) == 0 or label.startswith('_'):
                continue

            # Convert all data points to display coordinates
            points_display = ax.transData.transform(np.column_stack([xdata, ydata]))

            # Calculate distances to all points
            distances = np.sqrt(np.sum((points_display - mouse_display) ** 2, axis=1))

            # Find minimum distance for this line
            min_idx = np.argmin(distances)
            min_dist = distances[min_idx]

            if min_dist < min_distance:
                min_distance = min_dist
                nearest_point = (xdata[min_idx], ydata[min_idx], label, min_dist)

        return nearest_point


    def _hide_tooltip(self, ax):
        """
        Hide tooltip and crosshair.

        Args:
            ax: Matplotlib axes
        """
        # Remove annotation
        if self.tooltip_annotation:
            self.tooltip_annotation.remove()
            self.tooltip_annotation = None

        # Remove crosshair
        if ax in self.crosshair_lines:
            vline, hline = self.crosshair_lines[ax]
            vline.remove()
            hline.remove()
            del self.crosshair_lines[ax]

        # Remove highlight
        if ax in self.highlight_points:
            self.highlight_points[ax].remove()
            del self.highlight_points[ax]

        # Redraw canvas
        self.plotter.canvas.draw_idle()

    def _show_crosshair(self, ax, x, y):
        """
        Show crosshair lines at the specified position.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
        """
        # Remove old crosshair if exists
        if ax in self.crosshair_lines:
            vline, hline = self.crosshair_lines[ax]
            vline.remove()
            hline.remove()

        # Create new crosshair lines
        vline = ax.axvline(x, color='#808080', linestyle='--', linewidth=0.8, alpha=0.6, zorder=999)
        hline = ax.axhline(y, color='#808080', linestyle='--', linewidth=0.8, alpha=0.6, zorder=999)

        self.crosshair_lines[ax] = (vline, hline)

    def _highlight_point(self, ax, x, y):
        """
        Highlight the data point at the specified position.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
        """
        # Remove old highlight if exists
        if ax in self.highlight_points:
            self.highlight_points[ax].remove()

        # Create highlight circle
        highlight = Circle((x, y), radius=0.01, transform=ax.transData,
                          facecolor='yellow', edgecolor='orange',
                          linewidth=2, alpha=0.8, zorder=1001,
                          transform_rotates_text=False)

        # Add to axes
        ax.add_patch(highlight)
        self.highlight_points[ax] = highlight

    def _format_value(self, value: float) -> str:
        """
        Format a numeric value for display.

        Args:
            value: Numeric value to format

        Returns:
            Formatted string
        """
        if abs(value) >= 1_000_000:
            return f'{value/1_000_000:.2f}M'
        elif abs(value) >= 1_000:
            return f'{value/1_000:.2f}K'
        elif abs(value) >= 1:
            return f'{value:.2f}'
        elif abs(value) >= 0.01:
            return f'{value:.4f}'
        else:
            return f'{value:.6f}'

    def enable(self):
        """Enable tooltip functionality."""
        self.enabled = True
        self.logger.info("Tooltips enabled")

    def disable(self):
        """Disable tooltip functionality."""
        self.enabled = False
        # Clear all tooltips
        for ax in list(self.crosshair_lines.keys()):
            self._hide_tooltip(ax)
        self.logger.info("Tooltips disabled")

    def toggle_crosshair(self):
        """Toggle crosshair visibility."""
        self.show_crosshair = not self.show_crosshair
        self.logger.info(f"Crosshair {'enabled' if self.show_crosshair else 'disabled'}")

    def toggle_highlight(self):
        """Toggle point highlighting."""
        self.show_highlight = not self.show_highlight
        self.logger.info(f"Point highlighting {'enabled' if self.show_highlight else 'disabled'}")

    def _show_tooltip(self, ax, x, y, label, event):
        """
        Show tooltip with exact values.

        Args:
            ax: Matplotlib axes
            x: X coordinate
            y: Y coordinate
            label: Data series label
            event: Mouse event
        """
        # Remove old annotation if exists
        if self.tooltip_annotation:
            self.tooltip_annotation.remove()

        # Format values
        x_str = self._format_value(x)
        y_str = self._format_value(y)

        # Create tooltip text
        tooltip_text = f"{label}\nX: {x_str}\nY: {y_str}"

        # Create annotation
        self.tooltip_annotation = ax.annotate(
            tooltip_text,
            xy=(x, y),
            xytext=(20, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2a2a2a',
                     edgecolor='#505050', alpha=0.95, linewidth=1.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                          color='#808080', linewidth=1.5),
            fontsize=9,
            color='#e0e0e0',
            zorder=1000
        )

        # Redraw canvas
        self.plotter.canvas.draw_idle()

