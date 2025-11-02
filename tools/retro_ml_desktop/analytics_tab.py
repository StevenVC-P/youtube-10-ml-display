#!/usr/bin/env python3
"""
Analytics Tab for ML Dashboard

Provides advanced analytics features from Sprint 2:
- Convergence detection
- Statistical analysis
- Sample efficiency analysis
- Hyperparameter sensitivity analysis

Part of Sprint 2: Advanced Metrics & Analytics
"""

import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List, Dict, Optional
import customtkinter as ctk

from .ml_database import MetricsDatabase
from .convergence_detector import ConvergenceDetector
from .statistical_analyzer import StatisticalAnalyzer
from .sample_efficiency import SampleEfficiencyAnalyzer
from .hyperparameter_analyzer import HyperparameterAnalyzer


class AnalyticsTab:
    """
    Analytics tab for advanced metrics and analysis.
    
    Provides UI for:
    - Convergence detection
    - Statistical analysis
    - Sample efficiency metrics
    - Hyperparameter sensitivity
    """
    
    def __init__(self, parent_frame, database: MetricsDatabase):
        """
        Initialize analytics tab.
        
        Args:
            parent_frame: Parent tkinter frame
            database: MetricsDatabase instance
        """
        self.parent = parent_frame
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        # Initialize analyzers
        self.convergence_detector = ConvergenceDetector()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.efficiency_analyzer = SampleEfficiencyAnalyzer()
        self.hyperparameter_analyzer = HyperparameterAnalyzer()
        
        # Selected runs for analysis
        self.selected_runs = []
        
        # Setup UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the analytics tab UI."""
        # Main container with scrollbar
        main_container = ctk.CTkScrollableFrame(self.parent)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Run selection section
        self._create_run_selection(main_container)
        
        # Analysis sections
        self._create_convergence_section(main_container)
        self._create_statistical_section(main_container)
        self._create_efficiency_section(main_container)
        self._create_hyperparameter_section(main_container)
        
        # Results display
        self._create_results_display(main_container)
    
    def _create_run_selection(self, parent):
        """Create run selection section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(section, text="📊 Run Selection", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Run listbox
        listbox_frame = tk.Frame(section, bg='#2b2b2b')
        listbox_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(listbox_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.run_listbox = tk.Listbox(
            listbox_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
            bg='#3b3b3b',
            fg='white',
            height=6
        )
        self.run_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.run_listbox.yview)
        
        # Refresh button
        ctk.CTkButton(section, text="🔄 Refresh Runs", 
                     command=self._refresh_runs).pack(padx=10, pady=5)
        
        # Load initial runs
        self._refresh_runs()
    
    def _create_convergence_section(self, parent):
        """Create convergence detection section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(section, text="🎯 Convergence Detection", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Method selection
        method_frame = tk.Frame(section, bg='#2b2b2b')
        method_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(method_frame, text="Method:", bg='#2b2b2b', fg='white').pack(side="left", padx=5)
        
        self.convergence_method = tk.StringVar(value="auto")
        methods = ["auto", "moving_avg", "gradient", "variance", "plateau"]
        
        for method in methods:
            tk.Radiobutton(
                method_frame,
                text=method,
                variable=self.convergence_method,
                value=method,
                bg='#2b2b2b',
                fg='white',
                selectcolor='#4a4a4a'
            ).pack(side="left", padx=5)
        
        # Analyze button
        ctk.CTkButton(section, text="🔍 Detect Convergence", 
                     command=self._analyze_convergence).pack(padx=10, pady=5)
    
    def _create_statistical_section(self, parent):
        """Create statistical analysis section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(section, text="📈 Statistical Analysis", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        button_frame = tk.Frame(section, bg='#2b2b2b')
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="📊 Summary Statistics", 
                     command=self._analyze_statistics).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="🔬 Compare Runs", 
                     command=self._compare_runs).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="📉 Distribution Analysis", 
                     command=self._analyze_distribution).pack(side="left", padx=5)
    
    def _create_efficiency_section(self, parent):
        """Create sample efficiency section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(section, text="⚡ Sample Efficiency", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        button_frame = tk.Frame(section, bg='#2b2b2b')
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="📊 Efficiency Metrics", 
                     command=self._analyze_efficiency).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="🔄 Compare Efficiency", 
                     command=self._compare_efficiency).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="📉 Learning Phases", 
                     command=self._analyze_phases).pack(side="left", padx=5)
    
    def _create_hyperparameter_section(self, parent):
        """Create hyperparameter analysis section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(section, text="🎛️ Hyperparameter Analysis", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        button_frame = tk.Frame(section, bg='#2b2b2b')
        button_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(button_frame, text="📊 Correlation Analysis", 
                     command=self._analyze_correlations).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="🏆 Sensitivity Ranking", 
                     command=self._rank_sensitivity).pack(side="left", padx=5)
        
        ctk.CTkButton(button_frame, text="🎯 Optimal Ranges", 
                     command=self._find_optimal_ranges).pack(side="left", padx=5)
    
    def _create_results_display(self, parent):
        """Create results display section."""
        section = ctk.CTkFrame(parent)
        section.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(section, text="📋 Analysis Results", 
                    font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(
            section,
            wrap=tk.WORD,
            bg='#3b3b3b',
            fg='white',
            font=("Courier", 10),
            height=20
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Clear button
        ctk.CTkButton(section, text="🗑️ Clear Results", 
                     command=self._clear_results).pack(padx=10, pady=5)
    
    def _refresh_runs(self):
        """Refresh the list of available runs."""
        try:
            runs = self.database.get_all_runs()
            
            self.run_listbox.delete(0, tk.END)
            
            for run in runs:
                display_text = f"{run.run_id} - {run.algorithm} ({run.status})"
                self.run_listbox.insert(tk.END, display_text)
            
            self.logger.info(f"Loaded {len(runs)} runs")
        except Exception as e:
            self.logger.error(f"Failed to refresh runs: {e}")
            messagebox.showerror("Error", f"Failed to load runs: {e}")
    
    def _get_selected_runs(self) -> List[str]:
        """Get list of selected run IDs."""
        selected_indices = self.run_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one run")
            return []
        
        run_ids = []
        for idx in selected_indices:
            text = self.run_listbox.get(idx)
            run_id = text.split(" - ")[0]
            run_ids.append(run_id)
        
        return run_ids
    
    def _append_result(self, text: str):
        """Append text to results display."""
        self.results_text.insert(tk.END, text + "\n\n")
        self.results_text.see(tk.END)
    
    def _clear_results(self):
        """Clear results display."""
        self.results_text.delete(1.0, tk.END)
    
    # Analysis methods (placeholders - will be implemented with actual logic)
    
    def _analyze_convergence(self):
        """Analyze convergence for selected runs."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== CONVERGENCE ANALYSIS ===")
        self._append_result(f"Method: {self.convergence_method.get()}")
        self._append_result(f"Analyzing {len(run_ids)} run(s)...")
        
        # TODO: Implement actual convergence detection
        self._append_result("Feature implementation in progress...")
    
    def _analyze_statistics(self):
        """Analyze statistics for selected runs."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== STATISTICAL ANALYSIS ===")
        self._append_result(f"Analyzing {len(run_ids)} run(s)...")
        
        # TODO: Implement actual statistical analysis
        self._append_result("Feature implementation in progress...")
    
    def _compare_runs(self):
        """Compare selected runs statistically."""
        run_ids = self._get_selected_runs()
        if len(run_ids) < 2:
            messagebox.showwarning("Insufficient Selection", "Please select at least 2 runs to compare")
            return
        
        self._append_result("=== RUN COMPARISON ===")
        self._append_result(f"Comparing {len(run_ids)} runs...")
        
        # TODO: Implement actual comparison
        self._append_result("Feature implementation in progress...")
    
    def _analyze_distribution(self):
        """Analyze distribution of selected runs."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== DISTRIBUTION ANALYSIS ===")
        
        # TODO: Implement actual distribution analysis
        self._append_result("Feature implementation in progress...")
    
    def _analyze_efficiency(self):
        """Analyze sample efficiency."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== SAMPLE EFFICIENCY ANALYSIS ===")
        
        # TODO: Implement actual efficiency analysis
        self._append_result("Feature implementation in progress...")
    
    def _compare_efficiency(self):
        """Compare efficiency between runs."""
        run_ids = self._get_selected_runs()
        if len(run_ids) < 2:
            messagebox.showwarning("Insufficient Selection", "Please select at least 2 runs to compare")
            return
        
        self._append_result("=== EFFICIENCY COMPARISON ===")
        
        # TODO: Implement actual efficiency comparison
        self._append_result("Feature implementation in progress...")
    
    def _analyze_phases(self):
        """Analyze learning phases."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== LEARNING PHASES ANALYSIS ===")
        
        # TODO: Implement actual phase analysis
        self._append_result("Feature implementation in progress...")
    
    def _analyze_correlations(self):
        """Analyze hyperparameter correlations."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== HYPERPARAMETER CORRELATION ANALYSIS ===")
        
        # TODO: Implement actual correlation analysis
        self._append_result("Feature implementation in progress...")
    
    def _rank_sensitivity(self):
        """Rank hyperparameter sensitivity."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== HYPERPARAMETER SENSITIVITY RANKING ===")
        
        # TODO: Implement actual sensitivity ranking
        self._append_result("Feature implementation in progress...")
    
    def _find_optimal_ranges(self):
        """Find optimal hyperparameter ranges."""
        run_ids = self._get_selected_runs()
        if not run_ids:
            return
        
        self._append_result("=== OPTIMAL HYPERPARAMETER RANGES ===")
        
        # TODO: Implement actual optimal range finding
        self._append_result("Feature implementation in progress...")

