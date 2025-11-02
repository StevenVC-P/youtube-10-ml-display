#!/usr/bin/env python3
"""
Test Suite for Sprint 1: Enhanced Chart Interactivity

Tests for:
- Enhanced navigation toolbar with coordinate display
- Chart annotations with database persistence
- Chart state save/load functionality
- Multi-format export capabilities
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import sqlite3
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.retro_ml_desktop.ml_database import MetricsDatabase
from tools.retro_ml_desktop.chart_annotations import ChartAnnotation, ChartAnnotationManager
from tools.retro_ml_desktop.chart_state import ChartState, ChartStateManager
from tools.retro_ml_desktop.enhanced_export import ExportPreset, EnhancedExportManager


class MockPlotter:
    """Mock plotter for testing."""

    def __init__(self):
        self.axes = {
            'reward': MockAxes(),
            'loss': MockAxes(),
            'learning': MockAxes(),
            'system': MockAxes()
        }
        self.selected_runs = set()
        self.current_metric = "episode_reward_mean"
        self.auto_refresh = True
        self.refresh_interval = 5000
        self.figure = MockFigure()
        self.canvas = MockCanvas()

    def _update_plots(self):
        """Mock update plots method."""
        pass


class MockAxes:
    """Mock matplotlib axes."""
    
    def __init__(self):
        self.xlim = (0, 100)
        self.ylim = (0, 100)
        self.annotations = []
    
    def get_xlim(self):
        return self.xlim
    
    def get_ylim(self):
        return self.ylim
    
    def set_xlim(self, lim):
        self.xlim = lim
    
    def set_ylim(self, lim):
        self.ylim = lim
    
    def annotate(self, *args, **kwargs):
        annotation = MockAnnotation()
        self.annotations.append(annotation)
        return annotation


class MockAnnotation:
    """Mock matplotlib annotation."""
    
    def remove(self):
        pass


class MockFigure:
    """Mock matplotlib figure."""
    
    def savefig(self, *args, **kwargs):
        pass
    
    def get_facecolor(self):
        return '#2b2b2b'


class MockCanvas:
    """Mock matplotlib canvas."""
    
    def draw(self):
        pass


class TestChartAnnotations(unittest.TestCase):
    """Test chart annotation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.database = MetricsDatabase(self.temp_db.name)
        self.plotter = MockPlotter()
        self.annotation_manager = ChartAnnotationManager(self.plotter, self.database)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_annotation_creation(self):
        """Test creating a chart annotation."""
        annotation = ChartAnnotation(
            annotation_id="test_ann_1",
            run_id="run_1",
            axes_name="reward",
            x=100.0,
            y=50.0,
            text="Test annotation"
        )
        
        self.assertEqual(annotation.annotation_id, "test_ann_1")
        self.assertEqual(annotation.run_id, "run_1")
        self.assertEqual(annotation.axes_name, "reward")
        self.assertEqual(annotation.x, 100.0)
        self.assertEqual(annotation.y, 50.0)
        self.assertEqual(annotation.text, "Test annotation")
    
    def test_add_annotation(self):
        """Test adding annotation to database."""
        ann_id = self.annotation_manager.add_annotation(
            run_id="run_1",
            axes_name="reward",
            x=100.0,
            y=50.0,
            text="Test annotation"
        )
        
        self.assertIsNotNone(ann_id)
        self.assertIn(ann_id, self.annotation_manager.annotations)
    
    def test_annotation_persistence(self):
        """Test annotation persistence across sessions."""
        # Add annotation
        ann_id = self.annotation_manager.add_annotation(
            run_id="run_1",
            axes_name="reward",
            x=100.0,
            y=50.0,
            text="Persistent annotation"
        )
        
        # Create new manager (simulates restart)
        new_manager = ChartAnnotationManager(self.plotter, self.database)
        
        # Check annotation was loaded
        self.assertIn(ann_id, new_manager.annotations)
        annotation = new_manager.annotations[ann_id]
        self.assertEqual(annotation.text, "Persistent annotation")
    
    def test_delete_annotation(self):
        """Test deleting an annotation."""
        ann_id = self.annotation_manager.add_annotation(
            run_id="run_1",
            axes_name="reward",
            x=100.0,
            y=50.0,
            text="To be deleted"
        )
        
        success = self.annotation_manager.delete_annotation(ann_id)
        self.assertTrue(success)
        self.assertNotIn(ann_id, self.annotation_manager.annotations)
    
    def test_get_annotations_for_run(self):
        """Test retrieving annotations for a specific run."""
        self.annotation_manager.add_annotation("run_1", "reward", 100, 50, "Ann 1")
        self.annotation_manager.add_annotation("run_1", "loss", 200, 30, "Ann 2")
        self.annotation_manager.add_annotation("run_2", "reward", 150, 60, "Ann 3")
        
        run1_annotations = self.annotation_manager.get_annotations_for_run("run_1")
        self.assertEqual(len(run1_annotations), 2)


class TestChartState(unittest.TestCase):
    """Test chart state save/load functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.database = MetricsDatabase(self.temp_db.name)
        self.plotter = MockPlotter()
        self.state_manager = ChartStateManager(self.plotter, self.database)
    
    def tearDown(self):
        """Clean up test fixtures."""
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_state_creation(self):
        """Test creating a chart state."""
        state = ChartState(
            state_id="state_1",
            name="Test State",
            description="Test description",
            axis_limits={'reward': {'xlim': [0, 100], 'ylim': [0, 50]}},
            selected_runs=["run_1", "run_2"],
            current_metric="episode_reward_mean"
        )
        
        self.assertEqual(state.state_id, "state_1")
        self.assertEqual(state.name, "Test State")
        self.assertEqual(len(state.selected_runs), 2)
    
    def test_save_current_state(self):
        """Test saving current chart state."""
        self.plotter.selected_runs = {"run_1", "run_2"}
        
        state_id = self.state_manager.save_current_state(
            name="My State",
            description="Test state"
        )
        
        self.assertIsNotNone(state_id)
        self.assertIn(state_id, self.state_manager.states)
    
    def test_state_persistence(self):
        """Test state persistence across sessions."""
        self.plotter.selected_runs = {"run_1", "run_2"}
        
        state_id = self.state_manager.save_current_state("Persistent State")
        
        # Create new manager (simulates restart)
        new_manager = ChartStateManager(self.plotter, self.database)
        
        # Check state was loaded
        self.assertIn(state_id, new_manager.states)
        state = new_manager.states[state_id]
        self.assertEqual(state.name, "Persistent State")
    
    def test_load_state(self):
        """Test loading a saved state."""
        self.plotter.selected_runs = {"run_1", "run_2"}
        state_id = self.state_manager.save_current_state("Test State")
        
        # Change plotter state
        self.plotter.selected_runs = set()
        
        # Load state
        success = self.state_manager.load_state(state_id)
        self.assertTrue(success)
        self.assertEqual(len(self.plotter.selected_runs), 2)
    
    def test_delete_state(self):
        """Test deleting a saved state."""
        state_id = self.state_manager.save_current_state("To Delete")
        
        success = self.state_manager.delete_state(state_id)
        self.assertTrue(success)
        self.assertNotIn(state_id, self.state_manager.states)
    
    def test_list_states(self):
        """Test listing all saved states."""
        self.state_manager.save_current_state("State 1")
        self.state_manager.save_current_state("State 2")
        self.state_manager.save_current_state("State 3")
        
        states = self.state_manager.list_states()
        self.assertEqual(len(states), 3)


class TestEnhancedExport(unittest.TestCase):
    """Test enhanced export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plotter = MockPlotter()
        self.export_manager = EnhancedExportManager(self.plotter)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_export_presets(self):
        """Test export preset functionality."""
        preset_names = self.export_manager.get_preset_names()
        
        self.assertIn('web', preset_names)
        self.assertIn('print', preset_names)
        self.assertIn('publication', preset_names)
    
    def test_add_custom_preset(self):
        """Test adding a custom export preset."""
        custom_preset = ExportPreset(
            name="Custom",
            format="png",
            dpi=450,
            transparent=True
        )
        
        self.export_manager.add_preset(custom_preset)
        self.assertIn('custom', self.export_manager.get_preset_names())
    
    def test_export_history(self):
        """Test export history tracking."""
        initial_count = len(self.export_manager.get_export_history())
        
        filename = os.path.join(self.temp_dir, "test.png")
        self.export_manager.export_chart(filename, format='png', dpi=300)
        
        history = self.export_manager.get_export_history()
        self.assertEqual(len(history), initial_count + 1)


if __name__ == '__main__':
    unittest.main()

