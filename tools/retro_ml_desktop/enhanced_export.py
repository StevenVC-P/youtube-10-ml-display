#!/usr/bin/env python3
"""
Enhanced Export System for ML Charts

Provides advanced export capabilities including:
- Multiple format support (PNG, SVG, PDF)
- Configurable resolution and quality
- Batch export functionality
- Export presets
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import matplotlib.pyplot as plt


class ExportPreset:
    """
    Represents an export preset configuration.
    
    Attributes:
        name: Preset name
        format: File format (png, svg, pdf)
        dpi: Resolution in DPI
        transparent: Whether to use transparent background
        bbox_inches: Bounding box setting
        facecolor: Background color
        edgecolor: Edge color
    """
    
    def __init__(self, name: str, format: str = 'png', dpi: int = 300,
                 transparent: bool = False, bbox_inches: str = 'tight',
                 facecolor: str = '#2b2b2b', edgecolor: str = 'none'):
        self.name = name
        self.format = format
        self.dpi = dpi
        self.transparent = transparent
        self.bbox_inches = bbox_inches
        self.facecolor = facecolor
        self.edgecolor = edgecolor
    
    def to_dict(self):
        """Convert preset to dictionary."""
        return {
            'name': self.name,
            'format': self.format,
            'dpi': self.dpi,
            'transparent': self.transparent,
            'bbox_inches': self.bbox_inches,
            'facecolor': self.facecolor,
            'edgecolor': self.edgecolor
        }


class EnhancedExportManager:
    """
    Manages advanced export operations for ML charts.
    
    Features:
    - Multi-format export (PNG, SVG, PDF)
    - Configurable resolution and quality
    - Export presets
    - Batch export
    - Export history tracking
    """
    
    # Default export presets
    DEFAULT_PRESETS = {
        'web': ExportPreset('Web', 'png', 150, False),
        'print': ExportPreset('Print', 'pdf', 300, False),
        'presentation': ExportPreset('Presentation', 'png', 300, True),
        'publication': ExportPreset('Publication', 'pdf', 600, False),
        'vector': ExportPreset('Vector', 'svg', 300, False),
        'high_res': ExportPreset('High Resolution', 'png', 600, False)
    }
    
    def __init__(self, plotter):
        """
        Initialize enhanced export manager.
        
        Args:
            plotter: MLPlotter instance
        """
        self.plotter = plotter
        self.logger = logging.getLogger(__name__)
        
        # Export presets
        self.presets = self.DEFAULT_PRESETS.copy()
        
        # Export history
        self.export_history = []
        
        self.logger.info("Enhanced export manager initialized")
    
    def export_chart(self, filename: str, format: str = None, dpi: int = 300,
                    transparent: bool = False, bbox_inches: str = 'tight',
                    facecolor: str = None, edgecolor: str = 'none') -> bool:
        """
        Export chart with advanced options.
        
        Args:
            filename: Output filename (with or without extension)
            format: File format (png, svg, pdf) - auto-detected from filename if None
            dpi: Resolution in DPI
            transparent: Whether to use transparent background
            bbox_inches: Bounding box setting ('tight' or None)
            facecolor: Background color (None uses current)
            edgecolor: Edge color
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Auto-detect format from filename if not specified
            if format is None:
                format = Path(filename).suffix.lstrip('.')
                if not format:
                    format = 'png'
                    filename = f"{filename}.png"
            
            # Ensure filename has correct extension
            if not filename.endswith(f'.{format}'):
                filename = f"{filename}.{format}"
            
            # Use current facecolor if not specified
            if facecolor is None:
                facecolor = self.plotter.figure.get_facecolor()
            
            # Prepare export parameters
            export_params = {
                'dpi': dpi,
                'bbox_inches': bbox_inches,
                'facecolor': facecolor,
                'edgecolor': edgecolor,
                'transparent': transparent
            }
            
            # Format-specific adjustments
            if format == 'svg':
                export_params['format'] = 'svg'
            elif format == 'pdf':
                export_params['format'] = 'pdf'
            elif format == 'png':
                export_params['format'] = 'png'
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False
            
            # Save figure
            self.plotter.figure.savefig(filename, **export_params)
            
            # Record export
            self._record_export(filename, format, dpi)
            
            self.logger.info(f"Chart exported to {filename} ({format.upper()}, {dpi} DPI)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export chart: {e}")
            return False
    
    def export_with_preset(self, filename: str, preset_name: str) -> bool:
        """
        Export chart using a preset configuration.
        
        Args:
            filename: Output filename
            preset_name: Name of preset to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        if preset_name not in self.presets:
            self.logger.error(f"Preset '{preset_name}' not found")
            return False
        
        preset = self.presets[preset_name]
        
        return self.export_chart(
            filename=filename,
            format=preset.format,
            dpi=preset.dpi,
            transparent=preset.transparent,
            bbox_inches=preset.bbox_inches,
            facecolor=preset.facecolor,
            edgecolor=preset.edgecolor
        )
    
    def batch_export(self, base_filename: str, formats: List[str] = None,
                    dpi: int = 300) -> Dict[str, bool]:
        """
        Export chart in multiple formats.
        
        Args:
            base_filename: Base filename (without extension)
            formats: List of formats to export (default: ['png', 'pdf', 'svg'])
            dpi: Resolution in DPI
            
        Returns:
            Dict mapping format to success status
        """
        if formats is None:
            formats = ['png', 'pdf', 'svg']
        
        results = {}
        
        for format in formats:
            filename = f"{base_filename}.{format}"
            success = self.export_chart(filename, format=format, dpi=dpi)
            results[format] = success
        
        successful = sum(1 for v in results.values() if v)
        self.logger.info(f"Batch export completed: {successful}/{len(formats)} successful")
        
        return results
    
    def export_individual_subplots(self, base_filename: str, format: str = 'png',
                                   dpi: int = 300) -> Dict[str, bool]:
        """
        Export each subplot as a separate file.
        
        Args:
            base_filename: Base filename (without extension)
            format: File format
            dpi: Resolution in DPI
            
        Returns:
            Dict mapping subplot name to success status
        """
        results = {}
        
        try:
            for axes_name, ax in self.plotter.axes.items():
                # Create a new figure for this subplot
                fig = plt.figure(figsize=(8, 6), dpi=dpi, facecolor='#2b2b2b')
                
                # Copy the subplot to the new figure
                new_ax = fig.add_subplot(111)
                
                # Copy lines
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(),
                              color=line.get_color(), label=line.get_label(),
                              linewidth=line.get_linewidth(),
                              linestyle=line.get_linestyle())
                
                # Copy properties
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                new_ax.grid(True, alpha=0.3)
                
                # Copy legend if exists
                if ax.get_legend():
                    new_ax.legend(loc='upper left', fontsize=8)
                
                # Style
                new_ax.tick_params(colors='white')
                for spine in new_ax.spines.values():
                    spine.set_color('white')
                
                # Save
                filename = f"{base_filename}_{axes_name}.{format}"
                fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                           facecolor='#2b2b2b', edgecolor='none')
                plt.close(fig)
                
                results[axes_name] = True
                self.logger.info(f"Exported subplot '{axes_name}' to {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to export individual subplots: {e}")
            results[axes_name] = False
        
        return results
    
    def add_preset(self, preset: ExportPreset):
        """
        Add a custom export preset.
        
        Args:
            preset: ExportPreset object
        """
        self.presets[preset.name.lower().replace(' ', '_')] = preset
        self.logger.info(f"Added export preset '{preset.name}'")
    
    def get_preset_names(self) -> List[str]:
        """
        Get list of available preset names.
        
        Returns:
            List of preset names
        """
        return list(self.presets.keys())
    
    def _record_export(self, filename: str, format: str, dpi: int):
        """
        Record export in history.
        
        Args:
            filename: Exported filename
            format: File format
            dpi: Resolution
        """
        self.export_history.append({
            'filename': filename,
            'format': format,
            'dpi': dpi,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 exports
        if len(self.export_history) > 100:
            self.export_history = self.export_history[-100:]
    
    def get_export_history(self) -> List[Dict]:
        """
        Get export history.
        
        Returns:
            List of export records
        """
        return self.export_history.copy()

