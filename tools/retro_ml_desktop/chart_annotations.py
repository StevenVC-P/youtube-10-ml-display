#!/usr/bin/env python3
"""
Chart Annotation System

Provides annotation capabilities for ML training charts with:
- Interactive annotation placement
- Persistent storage in database
- Annotation management (add, edit, delete)
- Visual display on charts
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import json


class ChartAnnotation:
    """
    Represents a single chart annotation.
    
    Attributes:
        annotation_id: Unique identifier
        run_id: Associated experiment run ID
        axes_name: Name of the axes (reward, loss, learning, system)
        x: X-coordinate (timestep)
        y: Y-coordinate (metric value)
        text: Annotation text
        created_at: Creation timestamp
        color: Annotation color
        style: Annotation style
    """
    
    def __init__(self, annotation_id: str, run_id: str, axes_name: str,
                 x: float, y: float, text: str, created_at: str = None,
                 color: str = 'yellow', style: str = 'round'):
        self.annotation_id = annotation_id
        self.run_id = run_id
        self.axes_name = axes_name
        self.x = x
        self.y = y
        self.text = text
        self.created_at = created_at or datetime.now().isoformat()
        self.color = color
        self.style = style
    
    def to_dict(self):
        """Convert annotation to dictionary."""
        return {
            'annotation_id': self.annotation_id,
            'run_id': self.run_id,
            'axes_name': self.axes_name,
            'x': self.x,
            'y': self.y,
            'text': self.text,
            'created_at': self.created_at,
            'color': self.color,
            'style': self.style
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Create annotation from dictionary."""
        return cls(**data)


class ChartAnnotationManager:
    """
    Manages chart annotations with database persistence.
    
    Features:
    - Add annotations to charts
    - Store annotations in database
    - Load and display annotations
    - Edit and delete annotations
    - Annotation filtering by run
    """
    
    def __init__(self, plotter, database):
        """
        Initialize annotation manager.
        
        Args:
            plotter: MLPlotter instance
            database: MetricsDatabase instance
        """
        self.plotter = plotter
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Annotation storage
        self.annotations = {}  # annotation_id -> ChartAnnotation
        self.matplotlib_annotations = {}  # annotation_id -> matplotlib annotation object
        
        # Annotation mode
        self.annotation_mode = False
        self.current_axes = None
        
        # Initialize database table
        self._init_database()
        
        # Load existing annotations
        self._load_annotations()
        
        self.logger.info("Chart annotation manager initialized")
    
    def _init_database(self):
        """Initialize annotations table in database."""
        try:
            conn = self.database._get_connection()
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chart_annotations (
                    annotation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    axes_name TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    text TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    color TEXT DEFAULT 'yellow',
                    style TEXT DEFAULT 'round',
                    FOREIGN KEY (run_id) REFERENCES experiment_runs (run_id)
                )
            """)
            conn.commit()
            self.logger.info("Annotations table initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize annotations table: {e}")
    
    def _load_annotations(self):
        """Load all annotations from database."""
        try:
            conn = self.database._get_connection()
            cursor = conn.execute("""
                SELECT annotation_id, run_id, axes_name, x, y, text, 
                       created_at, color, style
                FROM chart_annotations
            """)
            
            for row in cursor.fetchall():
                annotation = ChartAnnotation(
                    annotation_id=row[0],
                    run_id=row[1],
                    axes_name=row[2],
                    x=row[3],
                    y=row[4],
                    text=row[5],
                    created_at=row[6],
                    color=row[7] or 'yellow',
                    style=row[8] or 'round'
                )
                self.annotations[annotation.annotation_id] = annotation
            
            self.logger.info(f"Loaded {len(self.annotations)} annotations from database")
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {e}")
    
    def add_annotation(self, run_id: str, axes_name: str, x: float, y: float, 
                      text: str, color: str = 'yellow') -> Optional[str]:
        """
        Add a new annotation.
        
        Args:
            run_id: Run ID to associate with
            axes_name: Axes name (reward, loss, learning, system)
            x: X-coordinate (timestep)
            y: Y-coordinate (metric value)
            text: Annotation text
            color: Annotation color
            
        Returns:
            str: Annotation ID if successful, None otherwise
        """
        try:
            # Generate unique ID
            annotation_id = f"ann_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Create annotation object
            annotation = ChartAnnotation(
                annotation_id=annotation_id,
                run_id=run_id,
                axes_name=axes_name,
                x=x,
                y=y,
                text=text,
                color=color
            )
            
            # Save to database
            conn = self.database._get_connection()
            conn.execute("""
                INSERT INTO chart_annotations 
                (annotation_id, run_id, axes_name, x, y, text, created_at, color, style)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (annotation_id, run_id, axes_name, x, y, text, 
                  annotation.created_at, color, 'round'))
            conn.commit()
            
            # Store in memory
            self.annotations[annotation_id] = annotation
            
            # Display on chart
            self._display_annotation(annotation)
            
            self.logger.info(f"Added annotation {annotation_id} at ({x}, {y})")
            return annotation_id
            
        except Exception as e:
            self.logger.error(f"Failed to add annotation: {e}")
            return None
    
    def _display_annotation(self, annotation: ChartAnnotation):
        """
        Display annotation on chart.
        
        Args:
            annotation: ChartAnnotation object
        """
        try:
            # Get the appropriate axes
            if annotation.axes_name not in self.plotter.axes:
                self.logger.warning(f"Axes {annotation.axes_name} not found")
                return
            
            ax = self.plotter.axes[annotation.axes_name]
            
            # Create matplotlib annotation
            mpl_annotation = ax.annotate(
                annotation.text,
                xy=(annotation.x, annotation.y),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(
                    boxstyle=f'{annotation.style},pad=0.3',
                    facecolor=annotation.color,
                    alpha=0.7,
                    edgecolor='white'
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0',
                    color='white',
                    lw=1.5
                ),
                fontsize=8,
                color='black',
                weight='bold'
            )
            
            # Store matplotlib annotation
            self.matplotlib_annotations[annotation.annotation_id] = mpl_annotation
            
            # Refresh canvas
            self.plotter.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to display annotation: {e}")
    
    def display_annotations_for_runs(self, run_ids: List[str]):
        """
        Display annotations for specific runs.
        
        Args:
            run_ids: List of run IDs to display annotations for
        """
        # Clear existing matplotlib annotations
        self._clear_matplotlib_annotations()
        
        # Display annotations for selected runs
        for annotation in self.annotations.values():
            if annotation.run_id in run_ids:
                self._display_annotation(annotation)
    
    def _clear_matplotlib_annotations(self):
        """Clear all matplotlib annotations from charts."""
        for ann_id, mpl_ann in self.matplotlib_annotations.items():
            try:
                mpl_ann.remove()
            except:
                pass
        
        self.matplotlib_annotations.clear()
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """
        Delete an annotation.
        
        Args:
            annotation_id: ID of annotation to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove from database
            conn = self.database._get_connection()
            conn.execute("DELETE FROM chart_annotations WHERE annotation_id = ?", 
                        (annotation_id,))
            conn.commit()
            
            # Remove from memory
            if annotation_id in self.annotations:
                del self.annotations[annotation_id]
            
            # Remove matplotlib annotation
            if annotation_id in self.matplotlib_annotations:
                self.matplotlib_annotations[annotation_id].remove()
                del self.matplotlib_annotations[annotation_id]
                self.plotter.canvas.draw()
            
            self.logger.info(f"Deleted annotation {annotation_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete annotation: {e}")
            return False
    
    def get_annotations_for_run(self, run_id: str) -> List[ChartAnnotation]:
        """
        Get all annotations for a specific run.
        
        Args:
            run_id: Run ID
            
        Returns:
            List of ChartAnnotation objects
        """
        return [ann for ann in self.annotations.values() if ann.run_id == run_id]

