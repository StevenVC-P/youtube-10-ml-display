"""
Dashboard widgets for the Retro ML Desktop application.
"""

from .recent_activity_widget import RecentActivityWidget
from .live_progress_widget import LiveProgressWidget
from .resource_monitor_widget import ResourceMonitorWidget
from .collapsible_frame import CollapsibleFrame
from .status_badge import StatusBadge

__all__ = [
    'RecentActivityWidget',
    'LiveProgressWidget',
    'ResourceMonitorWidget',
    'CollapsibleFrame',
    'StatusBadge'
]
