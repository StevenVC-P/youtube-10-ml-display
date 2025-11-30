"""
Centralized theme configuration for Retro ML Desktop.

This module provides a consistent color palette and styling configuration
for the entire application, supporting both light and dark modes.
"""

from typing import Tuple, Union

# Type alias for colors that support light/dark mode
ColorTuple = Union[str, Tuple[str, str]]


class Theme:
    """Centralized theme configuration with semantic color names."""
    
    # ============================================================================
    # PRIMARY COLOR PALETTE
    # ============================================================================
    
    # Primary colors (Info/Action)
    PRIMARY = "#1976D2"  # Material Blue 700
    PRIMARY_HOVER = "#1565C0"  # Material Blue 800
    PRIMARY_LIGHT = "#42A5F5"  # Material Blue 400
    
    # Success colors
    SUCCESS = "#2E7D32"  # Material Green 800
    SUCCESS_HOVER = "#1B5E20"  # Material Green 900
    SUCCESS_LIGHT = "#4CAF50"  # Material Green 500
    
    # Warning colors
    WARNING = "#F57C00"  # Material Orange 700
    WARNING_HOVER = "#E65100"  # Material Orange 900
    WARNING_LIGHT = "#FF9800"  # Material Orange 500
    
    # Danger colors
    DANGER = "#C62828"  # Material Red 800
    DANGER_HOVER = "#B71C1C"  # Material Red 900
    DANGER_LIGHT = "#F44336"  # Material Red 500
    
    # Info colors (Cyan/Teal)
    INFO = "#0097A7"  # Material Cyan 700
    INFO_HOVER = "#00838F"  # Material Cyan 800
    INFO_LIGHT = "#00BCD4"  # Material Cyan 500
    
    # Secondary/Neutral colors
    SECONDARY = "#546E7A"  # Material Blue Gray 600
    SECONDARY_HOVER = "#455A64"  # Material Blue Gray 700
    SECONDARY_LIGHT = "#78909C"  # Material Blue Gray 400
    
    # ============================================================================
    # STATUS COLORS (for StatusBadge widget)
    # ============================================================================
    
    STATUS_RUNNING = "#4CAF50"  # Green
    STATUS_PAUSED = "#FF9800"  # Orange
    STATUS_STOPPED = "#9E9E9E"  # Gray
    STATUS_COMPLETED = "#2196F3"  # Blue
    STATUS_FAILED = "#F44336"  # Red
    STATUS_IDLE = "#607D8B"  # Blue Gray
    
    # ============================================================================
    # BACKGROUND COLORS (Light, Dark)
    # ============================================================================
    
    BG_PRIMARY = ("gray90", "gray17")  # Main background
    BG_SECONDARY = ("gray85", "gray20")  # Secondary background
    BG_TERTIARY = ("gray80", "gray23")  # Tertiary background
    
    # ============================================================================
    # TEXT COLORS (Light, Dark)
    # ============================================================================
    
    TEXT_PRIMARY = ("gray10", "gray90")  # Primary text
    TEXT_SECONDARY = ("gray30", "gray70")  # Secondary text
    TEXT_DISABLED = ("gray50", "gray50")  # Disabled text
    TEXT_INVERSE = ("white", "black")  # Inverse text (for colored backgrounds)
    
    # ============================================================================
    # BORDER & SEPARATOR COLORS (Light, Dark)
    # ============================================================================
    
    BORDER_DEFAULT = ("gray70", "gray30")  # Default borders
    BORDER_FOCUS = (PRIMARY, PRIMARY)  # Focus indicator
    SEPARATOR = ("gray70", "gray30")  # Visual separators
    
    # ============================================================================
    # SEMANTIC BUTTON COLORS
    # ============================================================================
    
    @staticmethod
    def get_button_colors(variant: str = "primary") -> dict:
        """
        Get button color configuration for a specific variant.
        
        Args:
            variant: Button variant (primary, success, warning, danger, info, secondary)
            
        Returns:
            Dictionary with fg_color, hover_color, and text_color
        """
        variants = {
            "primary": {
                "fg_color": Theme.PRIMARY,
                "hover_color": Theme.PRIMARY_HOVER,
                "text_color": "white"
            },
            "success": {
                "fg_color": Theme.SUCCESS,
                "hover_color": Theme.SUCCESS_HOVER,
                "text_color": "white"
            },
            "warning": {
                "fg_color": Theme.WARNING,
                "hover_color": Theme.WARNING_HOVER,
                "text_color": "white"
            },
            "danger": {
                "fg_color": Theme.DANGER,
                "hover_color": Theme.DANGER_HOVER,
                "text_color": "white"
            },
            "info": {
                "fg_color": Theme.INFO,
                "hover_color": Theme.INFO_HOVER,
                "text_color": "white"
            },
            "secondary": {
                "fg_color": Theme.SECONDARY,
                "hover_color": Theme.SECONDARY_HOVER,
                "text_color": "white"
            }
        }
        
        return variants.get(variant, variants["primary"])
    
    @staticmethod
    def get_status_colors(status: str) -> dict:
        """
        Get color configuration for a specific status.
        
        Args:
            status: Status type (running, paused, stopped, completed, failed, idle)
            
        Returns:
            Dictionary with color and text_color
        """
        statuses = {
            "running": {"color": Theme.STATUS_RUNNING, "text_color": "white"},
            "paused": {"color": Theme.STATUS_PAUSED, "text_color": "black"},
            "stopped": {"color": Theme.STATUS_STOPPED, "text_color": "white"},
            "completed": {"color": Theme.STATUS_COMPLETED, "text_color": "white"},
            "failed": {"color": Theme.STATUS_FAILED, "text_color": "white"},
            "idle": {"color": Theme.STATUS_IDLE, "text_color": "white"}
        }
        
        return statuses.get(status, statuses["idle"])

