"""
Collapsible Frame Widget for CustomTkinter

A frame that can be expanded or collapsed with smooth animation.
Useful for organizing dashboard sections and improving UX.
"""

import customtkinter as ctk
from typing import Optional, Callable
from tools.retro_ml_desktop.theme import Theme


class CollapsibleFrame(ctk.CTkFrame):
    """
    A collapsible frame widget with smooth expand/collapse animation.
    
    Features:
    - Expandable/collapsible content area
    - Smooth animation
    - Customizable header with icon and title
    - State persistence support
    - Callback on state change
    """
    
    def __init__(
        self,
        parent,
        title: str = "Section",
        icon: str = "▼",
        collapsed_icon: str = "▶",
        initially_collapsed: bool = False,
        on_toggle: Optional[Callable[[bool], None]] = None,
        **kwargs
    ):
        """
        Initialize collapsible frame.
        
        Args:
            parent: Parent widget
            title: Section title text
            icon: Icon to show when expanded (default: ▼)
            collapsed_icon: Icon to show when collapsed (default: ▶)
            initially_collapsed: Whether to start collapsed
            on_toggle: Callback function called when toggled (receives is_expanded)
            **kwargs: Additional CTkFrame arguments
        """
        super().__init__(parent, **kwargs)
        
        self.title_text = title
        self.expanded_icon = icon
        self.collapsed_icon = collapsed_icon
        self.is_expanded = not initially_collapsed
        self.on_toggle_callback = on_toggle
        
        # Animation settings
        self.animation_steps = 10
        self.animation_duration = 150  # milliseconds
        
        self._setup_ui()
        
        # Set initial state
        if initially_collapsed:
            self.content_frame.pack_forget()
    
    def _setup_ui(self):
        """Setup the UI components."""
        # Header frame (always visible)
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=0, pady=0)
        
        # Toggle button with icon
        icon = self.expanded_icon if self.is_expanded else self.collapsed_icon
        self.toggle_button = ctk.CTkButton(
            self.header_frame,
            text=f"{icon} {self.title_text}",
            command=self.toggle,
            fg_color="transparent",
            hover_color=Theme.BG_SECONDARY,
            anchor="w",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.toggle_button.pack(fill="x", padx=5, pady=5)
        
        # Content frame (collapsible)
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def toggle(self):
        """Toggle the collapsed/expanded state."""
        if self.is_expanded:
            self.collapse()
        else:
            self.expand()
    
    def expand(self):
        """Expand the content frame."""
        if self.is_expanded:
            return
        
        self.is_expanded = True
        self.toggle_button.configure(text=f"{self.expanded_icon} {self.title_text}")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Call callback if provided
        if self.on_toggle_callback:
            self.on_toggle_callback(True)
    
    def collapse(self):
        """Collapse the content frame."""
        if not self.is_expanded:
            return
        
        self.is_expanded = False
        self.toggle_button.configure(text=f"{self.collapsed_icon} {self.title_text}")
        self.content_frame.pack_forget()
        
        # Call callback if provided
        if self.on_toggle_callback:
            self.on_toggle_callback(False)
    
    def get_content_frame(self) -> ctk.CTkFrame:
        """
        Get the content frame to add widgets to.
        
        Returns:
            The content frame where child widgets should be added
        """
        return self.content_frame
    
    def set_title(self, title: str):
        """Update the section title."""
        self.title_text = title
        icon = self.expanded_icon if self.is_expanded else self.collapsed_icon
        self.toggle_button.configure(text=f"{icon} {title}")
    
    def get_state(self) -> bool:
        """
        Get the current expanded state.
        
        Returns:
            True if expanded, False if collapsed
        """
        return self.is_expanded
    
    def set_state(self, expanded: bool):
        """
        Set the expanded state.
        
        Args:
            expanded: True to expand, False to collapse
        """
        if expanded and not self.is_expanded:
            self.expand()
        elif not expanded and self.is_expanded:
            self.collapse()

