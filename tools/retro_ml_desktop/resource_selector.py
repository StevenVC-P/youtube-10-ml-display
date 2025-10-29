"""
Enhanced resource selection dialog with detailed CPU/GPU information and auto-detection.
"""

import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from typing import Dict, List, Optional
from .process_manager import get_detailed_cpu_info, get_detailed_gpu_info, get_recommended_resources, CPUResource, GPUResource


class ResourceSelectorDialog:
    """Advanced resource selection dialog with detailed system information."""
    
    def __init__(self, parent, current_resources: Dict = None):
        self.parent = parent
        self.current_resources = current_resources or {}
        self.result = None
        
        # Get system information
        self.cpu_info = get_detailed_cpu_info()
        self.gpu_info = get_detailed_gpu_info()
        self.recommendations = get_recommended_resources()
        
        # Create dialog
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("🔧 Advanced Resource Selection")
        self.dialog.geometry("800x700")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (800 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"800x700+{x}+{y}")
        
        self._create_widgets()
        self._load_current_values()
    
    def _create_widgets(self):
        """Create the dialog widgets."""
        # Main scroll frame
        main_frame = ctk.CTkScrollableFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(main_frame, text="🔧 System Resource Selection", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=(0, 20))
        
        # System Overview
        self._create_system_overview(main_frame)
        
        # CPU Selection
        self._create_cpu_selection(main_frame)
        
        # GPU Selection  
        self._create_gpu_selection(main_frame)
        
        # Memory Selection
        self._create_memory_selection(main_frame)
        
        # Recommendations
        self._create_recommendations(main_frame)
        
        # Buttons
        self._create_buttons()
    
    def _create_system_overview(self, parent):
        """Create system overview section."""
        overview_frame = ctk.CTkFrame(parent)
        overview_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(overview_frame, text="📊 System Overview", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # System stats
        total_cpus = len(self.cpu_info)
        available_cpus = len([cpu for cpu in self.cpu_info if cpu.available])
        total_gpus = len(self.gpu_info)
        available_gpus = len([gpu for gpu in self.gpu_info if gpu.available])
        
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        overview_text = (
            f"💻 CPU Cores: {available_cpus}/{total_cpus} available\n"
            f"🎮 GPUs: {available_gpus}/{total_gpus} available\n"
            f"🧠 System Memory: {memory_gb:.1f} GB total\n"
            f"⚡ Auto-detection: {'✅ Active' if self.cpu_info and self.gpu_info else '❌ Limited'}"
        )
        
        ctk.CTkLabel(overview_frame, text=overview_text, justify="left").pack(pady=(0, 10), padx=10)
    
    def _create_cpu_selection(self, parent):
        """Create CPU selection section."""
        cpu_frame = ctk.CTkFrame(parent)
        cpu_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(cpu_frame, text="💻 CPU Core Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # CPU cores input
        cores_input_frame = ctk.CTkFrame(cpu_frame)
        cores_input_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(cores_input_frame, text="Number of CPU Cores:").pack(side="left", padx=5)
        
        self.cpu_cores_var = tk.StringVar(value=str(self.recommendations['cpu_cores']))
        cpu_cores_entry = ctk.CTkEntry(cores_input_frame, textvariable=self.cpu_cores_var, width=100)
        cpu_cores_entry.pack(side="left", padx=5)
        
        auto_cpu_btn = ctk.CTkButton(cores_input_frame, text="Auto Select", 
                                    command=self._auto_select_cpu, width=100)
        auto_cpu_btn.pack(side="left", padx=5)
        
        # CPU details
        if self.cpu_info:
            details_text = f"Available cores: {[cpu.core_id for cpu in self.cpu_info if cpu.available]}\n"
            details_text += f"Physical cores: {len([cpu for cpu in self.cpu_info if cpu.is_physical])}\n"
            details_text += f"Recommended: {self.recommendations['cpu_cores']} cores"
            
            ctk.CTkLabel(cpu_frame, text=details_text, justify="left", 
                        font=ctk.CTkFont(size=11)).pack(pady=(0, 10), padx=10)
    
    def _create_gpu_selection(self, parent):
        """Create GPU selection section."""
        gpu_frame = ctk.CTkFrame(parent)
        gpu_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(gpu_frame, text="🎮 GPU Selection", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # GPU selection
        gpu_input_frame = ctk.CTkFrame(gpu_frame)
        gpu_input_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(gpu_input_frame, text="GPU:").pack(side="left", padx=5)
        
        # Build GPU options
        gpu_options = ["auto", "none"]
        if self.gpu_info:
            for gpu in self.gpu_info:
                status = "✅" if gpu.available else "❌"
                gpu_options.append(f"{gpu.gpu_id}: {gpu.name} {status}")
        
        self.gpu_var = tk.StringVar(value="auto")
        gpu_combo = ctk.CTkComboBox(gpu_input_frame, variable=self.gpu_var, values=gpu_options, width=300)
        gpu_combo.pack(side="left", padx=5)
        
        auto_gpu_btn = ctk.CTkButton(gpu_input_frame, text="Auto Select", 
                                    command=self._auto_select_gpu, width=100)
        auto_gpu_btn.pack(side="left", padx=5)
        
        # GPU details
        if self.gpu_info:
            for gpu in self.gpu_info:
                status_icon = "✅" if gpu.available else "❌"
                details_text = (
                    f"{status_icon} GPU {gpu.gpu_id}: {gpu.name}\n"
                    f"   Memory: {gpu.memory_free_mb:.0f}MB free / {gpu.memory_total_mb:.0f}MB total\n"
                    f"   Utilization: {gpu.utilization_percent:.1f}% | Temp: {gpu.temperature_c:.0f}°C"
                )
                
                ctk.CTkLabel(gpu_frame, text=details_text, justify="left", 
                            font=ctk.CTkFont(size=11)).pack(pady=2, padx=10, anchor="w")
    
    def _create_memory_selection(self, parent):
        """Create memory selection section."""
        memory_frame = ctk.CTkFrame(parent)
        memory_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(memory_frame, text="🧠 Memory Allocation", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        # Memory input
        memory_input_frame = ctk.CTkFrame(memory_frame)
        memory_input_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(memory_input_frame, text="Memory Limit (GB):").pack(side="left", padx=5)
        
        self.memory_var = tk.StringVar(value=str(self.recommendations['memory_gb']))
        memory_entry = ctk.CTkEntry(memory_input_frame, textvariable=self.memory_var, width=100)
        memory_entry.pack(side="left", padx=5)
        
        auto_memory_btn = ctk.CTkButton(memory_input_frame, text="Auto Select", 
                                       command=self._auto_select_memory, width=100)
        auto_memory_btn.pack(side="left", padx=5)
        
        # Memory details
        import psutil
        memory = psutil.virtual_memory()
        details_text = (
            f"Total System Memory: {memory.total / (1024**3):.1f} GB\n"
            f"Currently Available: {memory.available / (1024**3):.1f} GB\n"
            f"Recommended Limit: {self.recommendations['memory_gb']} GB (75% of total)"
        )
        
        ctk.CTkLabel(memory_frame, text=details_text, justify="left", 
                    font=ctk.CTkFont(size=11)).pack(pady=(0, 10), padx=10)
    
    def _create_recommendations(self, parent):
        """Create recommendations section."""
        rec_frame = ctk.CTkFrame(parent)
        rec_frame.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(rec_frame, text="💡 Smart Recommendations", 
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(10, 5))
        
        rec_text = (
            f"🎯 Optimal Configuration:\n"
            f"   • CPU Cores: {self.recommendations['cpu_cores']} (75% of available physical cores)\n"
            f"   • GPU: {'Auto-select best available' if self.recommendations['gpu_id'] is not None else 'None available'}\n"
            f"   • Memory: {self.recommendations['memory_gb']} GB (75% of system memory)\n\n"
            f"📈 Performance Tips:\n"
            f"   • Use physical cores for better performance\n"
            f"   • Leave 25% memory free for system stability\n"
            f"   • Monitor GPU temperature during training\n"
            f"   • Consider pausing other GPU-intensive applications"
        )
        
        ctk.CTkLabel(rec_frame, text=rec_text, justify="left", 
                    font=ctk.CTkFont(size=11)).pack(pady=(0, 10), padx=10)
        
        # Apply recommendations button
        apply_rec_btn = ctk.CTkButton(rec_frame, text="🎯 Apply Recommendations", 
                                     command=self._apply_recommendations)
        apply_rec_btn.pack(pady=5)
    
    def _create_buttons(self):
        """Create dialog buttons."""
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=self._cancel)
        cancel_btn.pack(side="right", padx=5)
        
        apply_btn = ctk.CTkButton(button_frame, text="Apply Selection", command=self._apply)
        apply_btn.pack(side="right", padx=5)
    
    def _auto_select_cpu(self):
        """Auto-select optimal CPU configuration."""
        self.cpu_cores_var.set(str(self.recommendations['cpu_cores']))
    
    def _auto_select_gpu(self):
        """Auto-select optimal GPU configuration."""
        if self.recommendations['gpu_id'] is not None:
            # Find the GPU option that matches the recommended GPU
            for option in self.gpu_var.master.children.values():
                if hasattr(option, 'get') and str(self.recommendations['gpu_id']) in option.get():
                    self.gpu_var.set(option.get())
                    break
        else:
            self.gpu_var.set("none")
    
    def _auto_select_memory(self):
        """Auto-select optimal memory configuration."""
        self.memory_var.set(str(self.recommendations['memory_gb']))
    
    def _apply_recommendations(self):
        """Apply all recommendations."""
        self._auto_select_cpu()
        self._auto_select_gpu()
        self._auto_select_memory()
    
    def _load_current_values(self):
        """Load current resource values if provided."""
        if 'cpu_cores' in self.current_resources:
            self.cpu_cores_var.set(str(self.current_resources['cpu_cores']))
        if 'gpu_id' in self.current_resources:
            self.gpu_var.set(str(self.current_resources['gpu_id']))
        if 'memory_limit_gb' in self.current_resources:
            self.memory_var.set(str(self.current_resources['memory_limit_gb']))
    
    def _apply(self):
        """Apply the selected configuration."""
        try:
            # Parse GPU selection
            gpu_selection = self.gpu_var.get()
            if gpu_selection == "auto":
                gpu_id = "auto"
            elif gpu_selection == "none":
                gpu_id = "none"
            else:
                # Extract GPU ID from "ID: Name ✅" format
                gpu_id = gpu_selection.split(":")[0]
            
            self.result = {
                'cpu_cores': int(self.cpu_cores_var.get()),
                'gpu_id': gpu_id,
                'memory_limit_gb': float(self.memory_var.get()) if self.memory_var.get() else None,
                'priority': 'normal'
            }
            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")
    
    def _cancel(self):
        """Cancel the dialog."""
        self.result = None
        self.dialog.destroy()
    
    def show(self) -> Optional[Dict]:
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result
