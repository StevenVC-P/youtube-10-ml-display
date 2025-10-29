"""
RAM cleanup utilities for optimizing system memory before ML training.
"""

import psutil
import gc
import os
import subprocess
import platform
from typing import List, Dict, Tuple
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk


class RAMCleanupDialog:
    """Dialog for cleaning up system RAM before training."""
    
    def __init__(self, parent):
        self.parent = parent
        self.result = None
        
        # Create dialog
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("🧠 RAM Cleanup & Optimization")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"700x600+{x}+{y}")
        
        self._create_widgets()
        self._refresh_memory_info()
    
    def _create_widgets(self):
        """Create the dialog widgets."""
        # Main scroll frame
        main_frame = ctk.CTkScrollableFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="🧠 RAM Cleanup & Optimization", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # Current memory status
        self._create_memory_status(main_frame)
        
        # High memory processes
        self._create_process_list(main_frame)
        
        # Cleanup actions
        self._create_cleanup_actions(main_frame)
        
        # Buttons
        self._create_buttons()
    
    def _create_memory_status(self, parent):
        """Create memory status section."""
        status_frame = ctk.CTkFrame(parent)
        status_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(status_frame, text="📊 Current Memory Status", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        self.memory_status_label = ctk.CTkLabel(status_frame, text="Loading...", justify="left")
        self.memory_status_label.pack(pady=(0, 10), padx=10)
    
    def _create_process_list(self, parent):
        """Create high memory process list."""
        process_frame = ctk.CTkFrame(parent)
        process_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(process_frame, text="🔍 High Memory Processes", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Process list with checkboxes
        self.process_list_frame = ctk.CTkScrollableFrame(process_frame, height=200)
        self.process_list_frame.pack(fill="x", padx=10, pady=5)
        
        self.process_vars = {}  # Store checkbox variables
        
        refresh_btn = ctk.CTkButton(process_frame, text="🔄 Refresh Process List", 
                                   command=self._refresh_process_list)
        refresh_btn.pack(pady=10)
    
    def _create_cleanup_actions(self, parent):
        """Create cleanup action buttons."""
        cleanup_frame = ctk.CTkFrame(parent)
        cleanup_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(cleanup_frame, text="🧹 Cleanup Actions", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Quick cleanup buttons
        quick_frame = ctk.CTkFrame(cleanup_frame)
        quick_frame.pack(fill="x", padx=10, pady=5)
        
        # Python garbage collection
        gc_btn = ctk.CTkButton(quick_frame, text="🐍 Python Garbage Collection", 
                              command=self._run_garbage_collection)
        gc_btn.pack(side="left", padx=5, pady=5)
        
        # Windows memory cleanup
        if platform.system() == "Windows":
            mem_cleanup_btn = ctk.CTkButton(quick_frame, text="🪟 Windows Memory Cleanup", 
                                           command=self._run_windows_cleanup)
            mem_cleanup_btn.pack(side="left", padx=5, pady=5)
        
        # Close selected processes
        close_processes_btn = ctk.CTkButton(cleanup_frame, text="❌ Close Selected Processes", 
                                           command=self._close_selected_processes)
        close_processes_btn.pack(pady=10)
        
        # Cleanup explanation
        explanation_text = (
            "💡 Cleanup Tips:\n"
            "• Python Garbage Collection: Frees unused Python objects\n"
            "• Windows Memory Cleanup: Clears system cache and temporary files\n"
            "• Close Processes: Terminates selected high-memory applications\n"
            "• Always save your work before closing processes!"
        )
        ctk.CTkLabel(cleanup_frame, text=explanation_text, justify="left", 
                    font=ctk.CTkFont(size=10)).pack(pady=(0, 10), padx=10)
    
    def _create_buttons(self):
        """Create dialog buttons."""
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        close_btn = ctk.CTkButton(button_frame, text="Close", command=self._close)
        close_btn.pack(side="right", padx=5)
        
        refresh_btn = ctk.CTkButton(button_frame, text="🔄 Refresh All", command=self._refresh_all)
        refresh_btn.pack(side="right", padx=5)
    
    def _refresh_memory_info(self):
        """Refresh memory information display."""
        try:
            memory = psutil.virtual_memory()
            
            status_text = (
                f"💾 Total RAM: {memory.total / (1024**3):.1f} GB\n"
                f"🟢 Available: {memory.available / (1024**3):.1f} GB ({memory.available / memory.total * 100:.1f}%)\n"
                f"🔴 Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)\n"
                f"📊 Status: {'✅ Good' if memory.percent < 75 else '⚠️ High' if memory.percent < 90 else '🚨 Critical'}"
            )
            
            self.memory_status_label.configure(text=status_text)
        except Exception as e:
            self.memory_status_label.configure(text=f"Error getting memory info: {e}")
    
    def _refresh_process_list(self):
        """Refresh the high memory process list."""
        try:
            # Clear existing widgets
            for widget in self.process_list_frame.winfo_children():
                widget.destroy()
            self.process_vars.clear()
            
            # Get top memory-consuming processes
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                    if memory_mb > 100:  # Only show processes using > 100MB
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory_mb': memory_mb
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by memory usage
            processes.sort(key=lambda x: x['memory_mb'], reverse=True)
            
            # Display top 15 processes
            for i, proc in enumerate(processes[:15]):
                frame = ctk.CTkFrame(self.process_list_frame)
                frame.pack(fill="x", pady=2)
                
                # Checkbox for selection
                var = tk.BooleanVar()
                self.process_vars[proc['pid']] = var
                
                checkbox = ctk.CTkCheckBox(frame, text="", variable=var, width=20)
                checkbox.pack(side="left", padx=5)
                
                # Process info
                info_text = f"{proc['name']} (PID: {proc['pid']}) - {proc['memory_mb']:.0f} MB"
                label = ctk.CTkLabel(frame, text=info_text, anchor="w")
                label.pack(side="left", fill="x", expand=True, padx=5)
                
        except Exception as e:
            error_label = ctk.CTkLabel(self.process_list_frame, text=f"Error loading processes: {e}")
            error_label.pack(pady=10)
    
    def _run_garbage_collection(self):
        """Run Python garbage collection."""
        try:
            collected = gc.collect()
            messagebox.showinfo("Garbage Collection", f"✅ Collected {collected} objects")
            self._refresh_memory_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run garbage collection: {e}")
    
    def _run_windows_cleanup(self):
        """Run Windows memory cleanup."""
        if platform.system() != "Windows":
            messagebox.showwarning("Warning", "Windows cleanup only available on Windows systems")
            return
        
        try:
            # Run Windows memory cleanup
            subprocess.run(["cleanmgr", "/sagerun:1"], check=False, timeout=30)
            messagebox.showinfo("Windows Cleanup", "✅ Windows cleanup initiated")
            self._refresh_memory_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run Windows cleanup: {e}")
    
    def _close_selected_processes(self):
        """Close selected processes."""
        selected_pids = [pid for pid, var in self.process_vars.items() if var.get()]
        
        if not selected_pids:
            messagebox.showwarning("Warning", "No processes selected")
            return
        
        # Confirm action
        if not messagebox.askyesno("Confirm", f"Close {len(selected_pids)} selected processes?\n\nMake sure to save your work first!"):
            return
        
        closed_count = 0
        for pid in selected_pids:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                closed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                continue
        
        messagebox.showinfo("Processes Closed", f"✅ Closed {closed_count}/{len(selected_pids)} processes")
        self._refresh_all()
    
    def _refresh_all(self):
        """Refresh all information."""
        self._refresh_memory_info()
        self._refresh_process_list()
    
    def _close(self):
        """Close the dialog."""
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog."""
        self._refresh_process_list()
        self.dialog.wait_window()


def get_memory_recommendations() -> Dict[str, str]:
    """Get memory optimization recommendations."""
    memory = psutil.virtual_memory()
    recommendations = []
    
    if memory.percent > 90:
        recommendations.append("🚨 Critical: Close unnecessary applications immediately")
    elif memory.percent > 75:
        recommendations.append("⚠️ High usage: Consider closing some applications")
    else:
        recommendations.append("✅ Memory usage is healthy")
    
    if memory.available < 4 * (1024**3):  # Less than 4GB available
        recommendations.append("💾 Low available memory: Close browser tabs and unused programs")
    
    return {
        'status': 'critical' if memory.percent > 90 else 'warning' if memory.percent > 75 else 'good',
        'recommendations': recommendations,
        'available_gb': memory.available / (1024**3),
        'used_percent': memory.percent
    }
