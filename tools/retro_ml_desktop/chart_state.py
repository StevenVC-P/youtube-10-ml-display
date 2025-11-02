#!/usr/bin/env python3
"""
Chart State Management System

Provides save/load functionality for chart view states including:
- Axis limits and zoom levels
- Selected runs and metrics
- View configurations
- Persistent storage in database
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pickle
import base64


class ChartState:
    """
    Represents a saved chart state.
    
    Attributes:
        state_id: Unique identifier
        name: User-friendly name
        description: Optional description
        created_at: Creation timestamp
        axis_limits: Dictionary of axis limits for each subplot
        selected_runs: List of selected run IDs
        current_metric: Currently selected metric
        view_config: Additional view configuration
    """
    
    def __init__(self, state_id: str, name: str, description: str = "",
                 created_at: str = None, axis_limits: Dict = None,
                 selected_runs: List[str] = None, current_metric: str = None,
                 view_config: Dict = None):
        self.state_id = state_id
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.now().isoformat()
        self.axis_limits = axis_limits or {}
        self.selected_runs = selected_runs or []
        self.current_metric = current_metric or "episode_reward_mean"
        self.view_config = view_config or {}
    
    def to_dict(self):
        """Convert state to dictionary."""
        return {
            'state_id': self.state_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'axis_limits': self.axis_limits,
            'selected_runs': self.selected_runs,
            'current_metric': self.current_metric,
            'view_config': self.view_config
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create state from dictionary."""
        return cls(**data)


class ChartStateManager:
    """
    Manages chart state save/load operations.
    
    Features:
    - Save current chart view state
    - Load saved states
    - List all saved states
    - Delete states
    - Export/import states
    """
    
    def __init__(self, plotter, database):
        """
        Initialize chart state manager.
        
        Args:
            plotter: MLPlotter instance
            database: MetricsDatabase instance
        """
        self.plotter = plotter
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # State storage
        self.states = {}  # state_id -> ChartState
        
        # Initialize database table
        self._init_database()
        
        # Load existing states
        self._load_states()
        
        self.logger.info("Chart state manager initialized")
    
    def _init_database(self):
        """Initialize chart states table in database."""
        try:
            conn = self.database._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chart_states (
                    state_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    state_data TEXT NOT NULL
                )
            """)
            conn.commit()
            self.logger.info("Chart states table initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize chart states table: {e}")
    
    def _load_states(self):
        """Load all saved states from database."""
        try:
            conn = self.database._get_connection()
            cursor = conn.execute("""
                SELECT state_id, name, description, created_at, state_data
                FROM chart_states
            """)
            
            for row in cursor.fetchall():
                try:
                    state_data = json.loads(row[4])
                    state = ChartState(
                        state_id=row[0],
                        name=row[1],
                        description=row[2] or "",
                        created_at=row[3],
                        axis_limits=state_data.get('axis_limits', {}),
                        selected_runs=state_data.get('selected_runs', []),
                        current_metric=state_data.get('current_metric', 'episode_reward_mean'),
                        view_config=state_data.get('view_config', {})
                    )
                    self.states[state.state_id] = state
                except Exception as e:
                    self.logger.error(f"Failed to parse state {row[0]}: {e}")
            
            self.logger.info(f"Loaded {len(self.states)} chart states from database")
        except Exception as e:
            self.logger.error(f"Failed to load chart states: {e}")
    
    def save_current_state(self, name: str, description: str = "") -> Optional[str]:
        """
        Save current chart state.
        
        Args:
            name: User-friendly name for the state
            description: Optional description
            
        Returns:
            str: State ID if successful, None otherwise
        """
        try:
            # Generate unique ID
            state_id = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Capture current axis limits
            axis_limits = {}
            for axes_name, ax in self.plotter.axes.items():
                axis_limits[axes_name] = {
                    'xlim': list(ax.get_xlim()),
                    'ylim': list(ax.get_ylim())
                }
            
            # Create state object
            state = ChartState(
                state_id=state_id,
                name=name,
                description=description,
                axis_limits=axis_limits,
                selected_runs=list(self.plotter.selected_runs),
                current_metric=self.plotter.current_metric,
                view_config={
                    'auto_refresh': self.plotter.auto_refresh,
                    'refresh_interval': self.plotter.refresh_interval
                }
            )
            
            # Serialize state data
            state_data = {
                'axis_limits': state.axis_limits,
                'selected_runs': state.selected_runs,
                'current_metric': state.current_metric,
                'view_config': state.view_config
            }
            state_data_json = json.dumps(state_data)
            
            # Save to database
            conn = self.database._get_connection()
            conn.execute("""
                INSERT INTO chart_states 
                (state_id, name, description, created_at, state_data)
                VALUES (?, ?, ?, ?, ?)
            """, (state_id, name, description, state.created_at, state_data_json))
            conn.commit()
            
            # Store in memory
            self.states[state_id] = state
            
            self.logger.info(f"Saved chart state '{name}' with ID {state_id}")
            return state_id
            
        except Exception as e:
            self.logger.error(f"Failed to save chart state: {e}")
            return None
    
    def load_state(self, state_id: str) -> bool:
        """
        Load a saved chart state.
        
        Args:
            state_id: ID of state to load
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if state_id not in self.states:
                self.logger.error(f"State {state_id} not found")
                return False
            
            state = self.states[state_id]
            
            # Restore selected runs
            self.plotter.selected_runs = set(state.selected_runs)
            
            # Restore current metric
            self.plotter.current_metric = state.current_metric
            
            # Restore view config
            if 'auto_refresh' in state.view_config:
                self.plotter.auto_refresh = state.view_config['auto_refresh']
            if 'refresh_interval' in state.view_config:
                self.plotter.refresh_interval = state.view_config['refresh_interval']
            
            # Update plots to apply selected runs
            self.plotter._update_plots()
            
            # Restore axis limits
            for axes_name, limits in state.axis_limits.items():
                if axes_name in self.plotter.axes:
                    ax = self.plotter.axes[axes_name]
                    if 'xlim' in limits:
                        ax.set_xlim(limits['xlim'])
                    if 'ylim' in limits:
                        ax.set_ylim(limits['ylim'])
            
            # Refresh canvas
            self.plotter.canvas.draw()
            
            self.logger.info(f"Loaded chart state '{state.name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load chart state: {e}")
            return False
    
    def delete_state(self, state_id: str) -> bool:
        """
        Delete a saved state.
        
        Args:
            state_id: ID of state to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove from database
            conn = self.database._get_connection()
            conn.execute("DELETE FROM chart_states WHERE state_id = ?", (state_id,))
            conn.commit()
            
            # Remove from memory
            if state_id in self.states:
                del self.states[state_id]
            
            self.logger.info(f"Deleted chart state {state_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete chart state: {e}")
            return False
    
    def list_states(self) -> List[ChartState]:
        """
        Get list of all saved states.
        
        Returns:
            List of ChartState objects
        """
        return list(self.states.values())
    
    def get_state(self, state_id: str) -> Optional[ChartState]:
        """
        Get a specific state by ID.
        
        Args:
            state_id: State ID
            
        Returns:
            ChartState object or None if not found
        """
        return self.states.get(state_id)

