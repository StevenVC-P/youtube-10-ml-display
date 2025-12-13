"""
Storage cleanup utilities for managing ML training data and videos.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import customtkinter as ctk
from tkinter import messagebox
import json

UI_VERSION = "storage_cleaner v2"


class StorageCleanupDialog:
    """Dialog for cleaning up old training data and videos."""
    
    def __init__(self, parent, project_root: Path):
        self.parent = parent
        self.project_root = project_root
        self.result = None
        self.selected_runs = set()
        
        # Create dialog
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title(f"Storage Cleanup ({UI_VERSION})")
        self.dialog.geometry("900x700")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (700 // 2)
        self.dialog.geometry(f"900x700+{x}+{y}")
        
        self._create_widgets()
        self._scan_storage()
    
    def _create_widgets(self):
        """Create the dialog widgets."""
        main_frame = ctk.CTkFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Grid layout: header (0), runs scroll (1), bottom controls (2)
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_rowconfigure(2, weight=0)
        main_frame.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(main_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text=f"Storage Cleanup Manager ({UI_VERSION})",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0, 6))
        self.summary_label = ctk.CTkLabel(header, text="Scanning storage...",
                                          font=ctk.CTkFont(size=12))
        self.summary_label.pack(pady=(0, 6))

        # Runs list with scroll
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(list_frame, text="Select runs to delete:",
                     font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w", pady=(10, 5), padx=10)
        self.runs_frame = ctk.CTkScrollableFrame(list_frame)
        self.runs_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Bottom controls (fixed)
        bottom = ctk.CTkFrame(main_frame)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.grid_columnconfigure(0, weight=1)

        # Selection buttons
        select_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        select_frame.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(select_frame, text="Select All",
                      command=self._select_all, width=100).pack(side="left", padx=5)
        ctk.CTkButton(select_frame, text="Select None",
                      command=self._select_none, width=100).pack(side="left", padx=5)
        ctk.CTkButton(select_frame, text="Select Old (>7 days)",
                      command=self._select_old, width=150).pack(side="left", padx=5)

        # Delete options (always visible)
        options_frame = ctk.CTkFrame(bottom)
        options_frame.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(options_frame, text="Delete options (and/or):", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", pady=(5, 2), padx=10)
        self.delete_checkpoints_var = ctk.BooleanVar(value=True)
        self.delete_outputs_var = ctk.BooleanVar(value=True)
        self.delete_training_videos_var = ctk.BooleanVar(value=True)

        options_inner = ctk.CTkFrame(options_frame, fg_color="transparent")
        options_inner.pack(fill="x", padx=5, pady=(2, 6))
        ctk.CTkCheckBox(options_inner, text="Delete checkpoints/models", variable=self.delete_checkpoints_var).pack(anchor="w", padx=10, pady=2)
        ctk.CTkCheckBox(options_inner, text="Delete processed videos/outputs (outputs/, D:/ML_Videos)", variable=self.delete_outputs_var).pack(anchor="w", padx=10, pady=2)
        ctk.CTkCheckBox(options_inner, text="Delete training recordings (video/training)", variable=self.delete_training_videos_var).pack(anchor="w", padx=10, pady=2)

        # Action buttons
        button_frame = ctk.CTkFrame(bottom, fg_color="transparent")
        button_frame.pack(fill="x", pady=(0, 2))
        ctk.CTkButton(button_frame, text="Delete Selected",
                      command=self._delete_selected,
                      fg_color="#dc3545", hover_color="#c82333",
                      width=150).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Refresh",
                      command=self._scan_storage,
                      width=100).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Close",
                      command=self.dialog.destroy,
                      width=100).pack(side="right", padx=5)
    
    def _scan_storage(self):
        """Scan storage and populate the run list."""
        self.summary_label.configure(text="Scanning storage...")
        self.dialog.update()
        
        # Clear existing runs
        for widget in self.runs_frame.winfo_children():
            widget.destroy()
        
        self.run_data = []
        
        # Scan checkpoints directory
        checkpoints_dir = self.project_root / "models" / "checkpoints"
        if checkpoints_dir.exists():
            for run_dir in checkpoints_dir.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith("run-"):
                    run_info = self._get_run_info(run_dir)
                    if run_info:
                        self.run_data.append(run_info)
        
        # Sort by date (newest first)
        self.run_data.sort(key=lambda x: x['created'], reverse=True)
        
        # Calculate totals
        total_size = sum(r['total_size'] for r in self.run_data)
        total_runs = len(self.run_data)
        
        # Update summary
        self.summary_label.configure(
            text=f"Found {total_runs} training runs using {self._format_size(total_size)}"
        )
        
        # Create run entries
        for run_info in self.run_data:
            self._create_run_entry(run_info)
    
    def _get_run_info(self, run_dir: Path) -> Optional[Dict]:
        """Get information about a training run."""
        try:
            run_id = run_dir.name
            
            # Get metadata if available
            metadata_path = run_dir / "run_metadata.json"
            game = "Unknown"
            algorithm = "Unknown"
            created = datetime.fromtimestamp(run_dir.stat().st_ctime)
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    game = metadata.get('game', 'Unknown')
                    algorithm = metadata.get('algorithm', 'Unknown')
                    created_str = metadata.get('created')
                    if created_str:
                        created = datetime.fromisoformat(created_str)
            
            # Calculate sizes
            checkpoint_size = self._get_dir_size(run_dir / "milestones")
            video_size = self._get_dir_size(self.project_root / "outputs" / run_id)
            
            # Check D:\ML_Videos
            d_video_size = 0
            d_video_path = Path("D:/ML_Videos") / run_id
            if d_video_path.exists():
                d_video_size = self._get_dir_size(d_video_path)
            
            training_video_size = self._get_dir_size(self.project_root / "video" / "training" / run_id)
            total_size = checkpoint_size + video_size + d_video_size + training_video_size
            
            # Count checkpoints
            milestones_dir = run_dir / "milestones"
            checkpoint_count = 0
            if milestones_dir.exists():
                checkpoint_count = len(list(milestones_dir.glob("*.zip")))
            
            return {
                'run_id': run_id,
                'game': game,
                'algorithm': algorithm,
                'created': created,
                'checkpoint_size': checkpoint_size,
                'training_video_size': training_video_size,
                'video_size': video_size,
                'd_video_size': d_video_size,
                'total_size': total_size,
                'checkpoint_count': checkpoint_count,
                'run_dir': run_dir,
                'video_dir': self.project_root / "outputs" / run_id,
                'd_video_dir': d_video_path if d_video_path.exists() else None,
                'training_video_dir': self.project_root / "video" / "training" / run_id
            }
        except Exception as e:
            print(f"Error getting info for {run_dir}: {e}")
            return None
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        if not path.exists():
            return 0
        
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
        except Exception as e:
            print(f"Error calculating size for {path}: {e}")
        
        return total
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _create_run_entry(self, run_info: Dict):
        """Create a checkbox entry for a run."""
        frame = ctk.CTkFrame(self.runs_frame)
        frame.pack(fill="x", pady=2, padx=5)
        
        # Checkbox
        var = ctk.BooleanVar(value=False)
        checkbox = ctk.CTkCheckBox(frame, text="", variable=var, width=30,
                                   command=lambda: self._on_checkbox_changed(run_info['run_id'], var))
        checkbox.pack(side="left", padx=5)
        
        # Run info
        age_days = (datetime.now() - run_info['created']).days
        age_str = f"{age_days}d ago" if age_days > 0 else "Today"
        
        info_text = (
            f"{run_info['algorithm'].upper()}-{run_info['game'].split('/')[-1].replace('-v5', '')} "
            f"({run_info['run_id'][:12]}...) | "
            f"{run_info['checkpoint_count']} checkpoints | "
            f"{self._format_size(run_info['total_size'])} | "
            f"Created {age_str}"
        )
        
        label = ctk.CTkLabel(frame, text=info_text, anchor="w")
        label.pack(side="left", fill="x", expand=True, padx=5)
        
        # Store the variable
        run_info['var'] = var
    
    def _on_checkbox_changed(self, run_id: str, var: ctk.BooleanVar):
        """Handle checkbox state change."""
        if var.get():
            self.selected_runs.add(run_id)
        else:
            self.selected_runs.discard(run_id)
    
    def _select_all(self):
        """Select all runs."""
        for run_info in self.run_data:
            run_info['var'].set(True)
            self.selected_runs.add(run_info['run_id'])
    
    def _select_none(self):
        """Deselect all runs."""
        for run_info in self.run_data:
            run_info['var'].set(False)
        self.selected_runs.clear()
    
    def _select_old(self):
        """Select runs older than 7 days."""
        cutoff = datetime.now().timestamp() - (7 * 24 * 60 * 60)
        for run_info in self.run_data:
            if run_info['created'].timestamp() < cutoff:
                run_info['var'].set(True)
                self.selected_runs.add(run_info['run_id'])
    
    def _delete_selected(self):
        """Delete selected runs."""
        if not self.selected_runs:
            messagebox.showwarning("No Selection", "Please select runs to delete.")
            return
        
        # Calculate total size to delete based on options
        total_size = 0
        for r in self.run_data:
            if r['run_id'] in self.selected_runs:
                if self.delete_checkpoints_var.get():
                    total_size += r['checkpoint_size']
                if self.delete_outputs_var.get():
                    total_size += r['video_size'] + r['d_video_size']
                if self.delete_training_videos_var.get():
                    total_size += r['training_video_size']
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {len(self.selected_runs)} training runs?\n\n"
            f"This will free up {self._format_size(total_size)}.\n\n"
            f"Selected options:\n"
            f"- Checkpoints/models: {'Yes' if self.delete_checkpoints_var.get() else 'No'}\n"
            f"- Processed videos (outputs): {'Yes' if self.delete_outputs_var.get() else 'No'}\n"
            f"- Training recordings: {'Yes' if self.delete_training_videos_var.get() else 'No'}\n\n"
            f"This action cannot be undone!"
        )
        
        if not response:
            return
        
        # Delete runs
        deleted_count = 0
        freed_space = 0
        
        for run_info in self.run_data:
            if run_info['run_id'] in self.selected_runs:
                try:
                    run_freed = 0
                    # Delete checkpoint directory
                    if self.delete_checkpoints_var.get() and run_info['run_dir'].exists():
                        shutil.rmtree(run_info['run_dir'])
                        run_freed += run_info['checkpoint_size']
                    
                    # Delete processed video directory
                    if self.delete_outputs_var.get() and run_info['video_dir'].exists():
                        shutil.rmtree(run_info['video_dir'])
                        run_freed += run_info['video_size']
                    
                    # Delete D:\ML_Videos directory
                    if self.delete_outputs_var.get() and run_info['d_video_dir'] and run_info['d_video_dir'].exists():
                        shutil.rmtree(run_info['d_video_dir'])
                        run_freed += run_info['d_video_size']

                    # Delete training recordings
                    if self.delete_training_videos_var.get() and run_info['training_video_dir'].exists():
                        shutil.rmtree(run_info['training_video_dir'])
                        run_freed += run_info['training_video_size']
                    
                    deleted_count += 1
                    freed_space += run_freed
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete {run_info['run_id']}: {e}")
        
        # Show result
        messagebox.showinfo(
            "Cleanup Complete",
            f"Deleted {deleted_count} training runs.\n"
            f"Freed {self._format_size(freed_space)} of disk space."
        )
        
        # Refresh the list
        self.selected_runs.clear()
        self._scan_storage()
    
    def show(self):
        """Show the dialog and wait for it to close."""
        self.dialog.wait_window()
        return self.result


def show_storage_cleanup_dialog(parent, project_root: Path):
    """Show the storage cleanup dialog."""
    dialog = StorageCleanupDialog(parent, project_root)
    return dialog.show()
