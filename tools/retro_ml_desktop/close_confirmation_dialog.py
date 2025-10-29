"""
Close confirmation dialog for handling running training processes.
"""

import customtkinter as ctk
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class ProcessInfo:
    """Information about a running process."""
    id: str
    name: str
    status: str
    progress: float = 0.0

class CloseConfirmationDialog(ctk.CTkToplevel):
    """Dialog to confirm application close when training processes are running."""
    
    def __init__(self, parent, running_processes: List[ProcessInfo]):
        super().__init__(parent)
        
        self.running_processes = running_processes
        self.result: Optional[str] = None
        
        # Configure window
        self.title("Training Processes Running")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center on parent
        self.center_on_parent(parent)
        
        self.setup_ui()
        
    def center_on_parent(self, parent):
        """Center the dialog on the parent window."""
        self.update_idletasks()
        
        # Get parent position and size
        parent_x = parent.winfo_x()
        parent_y = parent.winfo_y()
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        
        # Calculate center position
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        """Setup the dialog UI."""
        # Main frame
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Warning icon and title
        title_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        title_frame.pack(fill="x", pady=(0, 20))
        
        warning_label = ctk.CTkLabel(
            title_frame,
            text="⚠️",
            font=ctk.CTkFont(size=32)
        )
        warning_label.pack(side="left", padx=(0, 10))
        
        title_label = ctk.CTkLabel(
            title_frame,
            text="Training Processes Are Running",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        title_label.pack(side="left")
        
        # Description
        desc_label = ctk.CTkLabel(
            main_frame,
            text="You have active training processes. What would you like to do?",
            font=ctk.CTkFont(size=12),
            wraplength=400
        )
        desc_label.pack(pady=(0, 20))
        
        # Process list
        if self.running_processes:
            process_frame = ctk.CTkFrame(main_frame)
            process_frame.pack(fill="x", pady=(0, 20))
            
            process_title = ctk.CTkLabel(
                process_frame,
                text=f"Running Processes ({len(self.running_processes)}):",
                font=ctk.CTkFont(size=12, weight="bold")
            )
            process_title.pack(anchor="w", padx=10, pady=(10, 5))
            
            # Scrollable frame for processes
            scrollable_frame = ctk.CTkScrollableFrame(process_frame, height=100)
            scrollable_frame.pack(fill="x", padx=10, pady=(0, 10))
            
            for process in self.running_processes:
                process_info = ctk.CTkLabel(
                    scrollable_frame,
                    text=f"• {process.name} ({process.status}) - {process.progress:.1f}%",
                    font=ctk.CTkFont(size=11),
                    anchor="w"
                )
                process_info.pack(fill="x", pady=2)
        
        # Button frame
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(20, 0))
        
        # Pause button
        pause_btn = ctk.CTkButton(
            button_frame,
            text="🔄 Pause Training\n(Resume Later)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2B5CE6",
            hover_color="#1E4BC7",
            height=50,
            command=self.pause_training
        )
        pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        # Stop button
        stop_btn = ctk.CTkButton(
            button_frame,
            text="🛑 Stop Training\n(Terminate All)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#E74C3C",
            hover_color="#C0392B",
            height=50,
            command=self.stop_training
        )
        stop_btn.pack(side="left", fill="x", expand=True, padx=5)
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            button_frame,
            text="❌ Cancel\n(Keep Running)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#95A5A6",
            hover_color="#7F8C8D",
            height=50,
            command=self.cancel_close
        )
        cancel_btn.pack(side="left", fill="x", expand=True, padx=(5, 0))
        
        # Info text
        info_label = ctk.CTkLabel(
            main_frame,
            text="💡 Paused training can be resumed when you restart the application",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        info_label.pack(pady=(10, 0))
    
    def pause_training(self):
        """Pause all training and close application."""
        self.result = "pause"
        self.destroy()
    
    def stop_training(self):
        """Stop all training and close application."""
        self.result = "stop"
        self.destroy()
    
    def cancel_close(self):
        """Cancel closing the application."""
        self.result = "cancel"
        self.destroy()
    
    def get_result(self) -> Optional[str]:
        """Get the user's choice."""
        return self.result


def show_close_confirmation(parent, running_processes: List[ProcessInfo]) -> Optional[str]:
    """Show close confirmation dialog and return user choice."""
    dialog = CloseConfirmationDialog(parent, running_processes)
    parent.wait_window(dialog)
    return dialog.get_result()
