"""
Retro ML Desktop - Main application entry point.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import yaml
import threading
from pathlib import Path
from typing import Dict, List, Optional

# Add the project root to the path so we can import our modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.retro_ml_desktop.monitor import SystemMonitor, SystemMetrics, get_gpu_status_message
from tools.retro_ml_desktop.docker_manager import DockerManager, ContainerInfo, ResourceLimits, generate_run_id


class RetroMLDesktop:
    """Main application class for Retro ML Desktop."""
    
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize components
        self.system_monitor = SystemMonitor(update_interval=2.0)
        self.docker_manager = None
        self.presets = {}
        self.games = []
        self.algorithms = []
        
        # UI state
        self.outputs_dir = str(project_root / "outputs")
        self.datasets_dir = str(project_root / "datasets")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Retro ML Desktop - Container Manager")
        self.root.geometry("1200x800")
        
        # Load presets
        self._load_presets()
        
        # Initialize Docker manager
        self._init_docker_manager()
        
        # Create UI
        self._create_ui()
        
        # Start monitoring
        self.system_monitor.add_callback(self._update_dashboard)
        self.system_monitor.start_monitoring()
        
        # Refresh containers initially
        self._refresh_containers()
    
    def _load_presets(self):
        """Load training presets from YAML file."""
        presets_file = Path(__file__).parent / "training_presets.yaml"
        try:
            with open(presets_file, 'r') as f:
                data = yaml.safe_load(f)
                self.presets = data.get('presets', {})
                self.games = data.get('games', [])
                self.algorithms = data.get('algorithms', ['ppo', 'dqn'])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load presets: {e}")
            self.presets = {}
            self.games = []
            self.algorithms = ['ppo', 'dqn']
    
    def _init_docker_manager(self):
        """Initialize Docker manager with error handling."""
        try:
            self.docker_manager = DockerManager()
        except Exception as e:
            messagebox.showerror(
                "Docker Error",
                f"Failed to connect to Docker daemon:\n{e}\n\n"
                "Please ensure Docker is running and accessible."
            )
            self.docker_manager = None
    
    def _create_ui(self):
        """Create the main user interface."""
        # Create main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create sidebar for folder selection
        self._create_sidebar(main_frame)
        
        # Create main content area with tabs
        self._create_main_content(main_frame)
    
    def _create_sidebar(self, parent):
        """Create the sidebar with folder selection."""
        sidebar = ctk.CTkFrame(parent, width=250)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)
        
        # Title
        title = ctk.CTkLabel(sidebar, text="Folders", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=(20, 10))
        
        # Outputs directory
        ctk.CTkLabel(sidebar, text="Outputs Directory:").pack(pady=(10, 5))
        self.outputs_var = tk.StringVar(value=self.outputs_dir)
        outputs_frame = ctk.CTkFrame(sidebar)
        outputs_frame.pack(fill="x", padx=10, pady=5)
        
        outputs_entry = ctk.CTkEntry(outputs_frame, textvariable=self.outputs_var, width=180)
        outputs_entry.pack(side="left", padx=5, pady=5)
        
        outputs_btn = ctk.CTkButton(
            outputs_frame, text="...", width=30,
            command=lambda: self._select_folder(self.outputs_var, "Select Outputs Directory")
        )
        outputs_btn.pack(side="right", padx=5, pady=5)
        
        # Datasets directory
        ctk.CTkLabel(sidebar, text="Datasets Directory:").pack(pady=(10, 5))
        self.datasets_var = tk.StringVar(value=self.datasets_dir)
        datasets_frame = ctk.CTkFrame(sidebar)
        datasets_frame.pack(fill="x", padx=10, pady=5)
        
        datasets_entry = ctk.CTkEntry(datasets_frame, textvariable=self.datasets_var, width=180)
        datasets_entry.pack(side="left", padx=5, pady=5)
        
        datasets_btn = ctk.CTkButton(
            datasets_frame, text="...", width=30,
            command=lambda: self._select_folder(self.datasets_var, "Select Datasets Directory")
        )
        datasets_btn.pack(side="right", padx=5, pady=5)
        
        # Start Training button
        start_btn = ctk.CTkButton(
            sidebar, text="Start Training", font=ctk.CTkFont(size=14, weight="bold"),
            height=40, command=self._show_start_training_dialog
        )
        start_btn.pack(pady=20, padx=10, fill="x")
        
        # Docker status
        ctk.CTkLabel(sidebar, text="Docker Status:", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 5))
        self.docker_status_label = ctk.CTkLabel(sidebar, text="Checking...")
        self.docker_status_label.pack(pady=5)
        
        # GPU status
        ctk.CTkLabel(sidebar, text="GPU Status:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        gpu_status = get_gpu_status_message()
        self.gpu_status_label = ctk.CTkLabel(sidebar, text=gpu_status)
        self.gpu_status_label.pack(pady=5)
        
        # Update Docker status
        self._update_docker_status()
    
    def _create_main_content(self, parent):
        """Create the main content area with tabs."""
        # Create tabview
        self.tabview = ctk.CTkTabview(parent)
        self.tabview.pack(side="right", fill="both", expand=True)
        
        # Create tabs
        self.dashboard_tab = self.tabview.add("Dashboard")
        self.containers_tab = self.tabview.add("Containers")
        self.logs_tab = self.tabview.add("Logs")
        
        # Setup each tab
        self._setup_dashboard_tab()
        self._setup_containers_tab()
        self._setup_logs_tab()
    
    def _setup_dashboard_tab(self):
        """Setup the dashboard tab with system metrics."""
        # CPU section
        cpu_frame = ctk.CTkFrame(self.dashboard_tab)
        cpu_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(cpu_frame, text="CPU", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.cpu_label = ctk.CTkLabel(cpu_frame, text="Loading...")
        self.cpu_label.pack(pady=5)
        
        # Memory section
        memory_frame = ctk.CTkFrame(self.dashboard_tab)
        memory_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(memory_frame, text="Memory", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.memory_label = ctk.CTkLabel(memory_frame, text="Loading...")
        self.memory_label.pack(pady=5)
        
        # GPU section
        gpu_frame = ctk.CTkFrame(self.dashboard_tab)
        gpu_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(gpu_frame, text="GPUs", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        self.gpu_label = ctk.CTkLabel(gpu_frame, text="Loading...")
        self.gpu_label.pack(pady=5)
    
    def _setup_containers_tab(self):
        """Setup the containers tab with container list."""
        # Control buttons
        controls_frame = ctk.CTkFrame(self.containers_tab)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        refresh_btn = ctk.CTkButton(controls_frame, text="Refresh", command=self._refresh_containers)
        refresh_btn.pack(side="left", padx=5, pady=5)
        
        stop_btn = ctk.CTkButton(controls_frame, text="Stop Selected", command=self._stop_selected_container)
        stop_btn.pack(side="left", padx=5, pady=5)
        
        remove_btn = ctk.CTkButton(controls_frame, text="Remove Selected", command=self._remove_selected_container)
        remove_btn.pack(side="left", padx=5, pady=5)
        
        # Container list (using tkinter Treeview for table)
        list_frame = ctk.CTkFrame(self.containers_tab)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create Treeview
        columns = ("Name", "Image", "Status", "Created")
        self.container_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # Configure columns
        for col in columns:
            self.container_tree.heading(col, text=col)
            self.container_tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.container_tree.yview)
        self.container_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.container_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _setup_logs_tab(self):
        """Setup the logs tab with log viewer."""
        # Log text area
        self.log_text = ctk.CTkTextbox(self.logs_tab, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Clear logs button
        clear_btn = ctk.CTkButton(self.logs_tab, text="Clear Logs", command=self._clear_logs)
        clear_btn.pack(pady=5)
    
    def _select_folder(self, var: tk.StringVar, title: str):
        """Open folder selection dialog."""
        folder = filedialog.askdirectory(title=title, initialdir=var.get())
        if folder:
            var.set(folder)
    
    def _update_dashboard(self, metrics: SystemMetrics):
        """Update dashboard with new metrics."""
        def update_ui():
            # Update CPU
            cpu_text = (
                f"Usage: {metrics.cpu.percent:.1f}%\n"
                f"Frequency: {metrics.cpu.frequency:.0f} MHz\n"
                f"Cores: {metrics.cpu.logical_cores} logical, {metrics.cpu.physical_cores} physical"
            )
            self.cpu_label.configure(text=cpu_text)
            
            # Update Memory
            memory_text = (
                f"Used: {metrics.memory.used_gb:.1f} GB / {metrics.memory.total_gb:.1f} GB\n"
                f"Usage: {metrics.memory.percent:.1f}%\n"
                f"Available: {metrics.memory.available_gb:.1f} GB"
            )
            self.memory_label.configure(text=memory_text)
            
            # Update GPU
            if metrics.gpus:
                gpu_lines = []
                for gpu in metrics.gpus:
                    gpu_lines.append(
                        f"GPU {gpu.id}: {gpu.name}\n"
                        f"  Load: {gpu.load_percent:.1f}%\n"
                        f"  Memory: {gpu.memory_used_mb:.0f} MB / {gpu.memory_total_mb:.0f} MB ({gpu.memory_percent:.1f}%)\n"
                        f"  Temperature: {gpu.temperature:.0f}°C"
                    )
                gpu_text = "\n\n".join(gpu_lines)
            else:
                gpu_text = get_gpu_status_message()
            
            self.gpu_label.configure(text=gpu_text)
        
        # Schedule UI update on main thread
        self.root.after(0, update_ui)
    
    def _update_docker_status(self):
        """Update Docker connection status."""
        if self.docker_manager and self.docker_manager.is_connected():
            self.docker_status_label.configure(text="Connected", text_color="green")
        else:
            self.docker_status_label.configure(text="Disconnected", text_color="red")
    
    def _refresh_containers(self):
        """Refresh the container list."""
        if not self.docker_manager:
            return

        # Clear existing items
        for item in self.container_tree.get_children():
            self.container_tree.delete(item)

        # Get containers
        containers = self.docker_manager.get_training_containers()

        # Add containers to tree
        for container in containers:
            created_str = container.created.strftime("%Y-%m-%d %H:%M")
            self.container_tree.insert("", "end", values=(
                container.name,
                container.image,
                container.status,
                created_str
            ), tags=(container.id,))

    def _stop_selected_container(self):
        """Stop the selected container."""
        selection = self.container_tree.selection()
        if not selection or not self.docker_manager:
            return

        item = selection[0]
        container_id = self.container_tree.item(item)["tags"][0]

        if messagebox.askyesno("Confirm", "Stop selected container?"):
            success = self.docker_manager.stop_container(container_id)
            if success:
                self._refresh_containers()
            else:
                messagebox.showerror("Error", "Failed to stop container")

    def _remove_selected_container(self):
        """Remove the selected container."""
        selection = self.container_tree.selection()
        if not selection or not self.docker_manager:
            return

        item = selection[0]
        container_id = self.container_tree.item(item)["tags"][0]

        if messagebox.askyesno("Confirm", "Remove selected container?"):
            success = self.docker_manager.remove_container(container_id, force=True)
            if success:
                self._refresh_containers()
            else:
                messagebox.showerror("Error", "Failed to remove container")

    def _clear_logs(self):
        """Clear the log display."""
        self.log_text.delete("1.0", "end")

    def _append_log(self, message: str):
        """Append a message to the log display."""
        def update_ui():
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")

        self.root.after(0, update_ui)

    def _show_start_training_dialog(self):
        """Show the start training dialog."""
        if not self.docker_manager:
            messagebox.showerror("Error", "Docker not connected")
            return

        dialog = StartTrainingDialog(self.root, self.presets, self.games, self.algorithms)
        result = dialog.show()

        if result:
            self._start_training_container(result)

    def _start_training_container(self, config: Dict):
        """Start a new training container with the given configuration."""
        try:
            # Get preset
            preset = self.presets[config['preset']]

            # Generate run ID
            run_id = config.get('run_id') or generate_run_id()

            # Prepare command
            command = preset['command'].format(
                game=config['game'],
                algo=config['algorithm'],
                run_id=run_id,
                omp_threads=config.get('omp_threads', 2)
            )

            # Prepare volumes
            volumes = []
            for volume in preset.get('volumes', []):
                expanded_volume = volume.format(
                    host_outputs=self.outputs_var.get(),
                    host_datasets=self.datasets_var.get()
                )
                volumes.append(expanded_volume)

            # Prepare environment
            environment = []
            for env in preset.get('env', []):
                expanded_env = env.format(
                    omp_threads=config.get('omp_threads', 2)
                )
                environment.append(expanded_env)

            # Prepare resource limits
            resources = ResourceLimits(
                cpus=config.get('cpus'),
                nano_cpus=config.get('nano_cpus'),
                mem_limit=config.get('mem_limit'),
                shm_size=config.get('shm_size'),
                gpu=config.get('gpu', 'auto')
            )

            # Create container
            container_name = f"retro-ml-{config['game']}-{config['algorithm']}-{run_id}"
            container_id = self.docker_manager.create_container(
                image=preset['image'],
                command=command,
                name=container_name,
                workdir=preset.get('workdir', '/workspace'),
                volumes=volumes,
                environment=environment,
                resources=resources,
                run_id=run_id
            )

            # Start log streaming
            self.docker_manager.start_log_stream(container_id, self._append_log)

            # Refresh container list
            self._refresh_containers()

            # Switch to logs tab
            self.tabview.set("Logs")

            messagebox.showinfo("Success", f"Training container started: {container_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

    def run(self):
        """Start the application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanup
            self.system_monitor.stop_monitoring()


class StartTrainingDialog:
    """Dialog for configuring and starting a new training run."""

    def __init__(self, parent, presets: Dict, games: List[str], algorithms: List[str]):
        self.parent = parent
        self.presets = presets
        self.games = games
        self.algorithms = algorithms
        self.result = None

        # Create dialog window
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Start Training")
        self.dialog.geometry("500x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Center the dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (600 // 2)
        self.dialog.geometry(f"500x600+{x}+{y}")

        self._create_dialog_ui()

    def _create_dialog_ui(self):
        """Create the dialog UI."""
        # Main frame
        main_frame = ctk.CTkScrollableFrame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Preset selection
        ctk.CTkLabel(main_frame, text="Preset:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        self.preset_var = tk.StringVar(value=list(self.presets.keys())[0] if self.presets else "")
        preset_combo = ctk.CTkComboBox(main_frame, variable=self.preset_var, values=list(self.presets.keys()))
        preset_combo.pack(fill="x", pady=(0, 10))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        # Game selection
        ctk.CTkLabel(main_frame, text="Game:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        self.game_var = tk.StringVar(value=self.games[0] if self.games else "")
        game_combo = ctk.CTkComboBox(main_frame, variable=self.game_var, values=self.games)
        game_combo.pack(fill="x", pady=(0, 10))

        # Algorithm selection
        ctk.CTkLabel(main_frame, text="Algorithm:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        self.algo_var = tk.StringVar(value=self.algorithms[0] if self.algorithms else "")
        algo_combo = ctk.CTkComboBox(main_frame, variable=self.algo_var, values=self.algorithms)
        algo_combo.pack(fill="x", pady=(0, 10))

        # Run ID
        ctk.CTkLabel(main_frame, text="Run ID:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(0, 5))
        self.run_id_var = tk.StringVar(value=generate_run_id())
        run_id_entry = ctk.CTkEntry(main_frame, textvariable=self.run_id_var)
        run_id_entry.pack(fill="x", pady=(0, 10))

        # Resource controls
        ctk.CTkLabel(main_frame, text="Resource Limits:", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 5))

        # CPUs
        ctk.CTkLabel(main_frame, text="CPUs:").pack(anchor="w", pady=(0, 5))
        self.cpus_var = tk.StringVar(value="6")
        cpus_entry = ctk.CTkEntry(main_frame, textvariable=self.cpus_var)
        cpus_entry.pack(fill="x", pady=(0, 10))

        # Memory limit
        ctk.CTkLabel(main_frame, text="Memory Limit:").pack(anchor="w", pady=(0, 5))
        self.mem_limit_var = tk.StringVar(value="16g")
        mem_entry = ctk.CTkEntry(main_frame, textvariable=self.mem_limit_var)
        mem_entry.pack(fill="x", pady=(0, 10))

        # Shared memory
        ctk.CTkLabel(main_frame, text="Shared Memory:").pack(anchor="w", pady=(0, 5))
        self.shm_size_var = tk.StringVar(value="2g")
        shm_entry = ctk.CTkEntry(main_frame, textvariable=self.shm_size_var)
        shm_entry.pack(fill="x", pady=(0, 10))

        # GPU
        ctk.CTkLabel(main_frame, text="GPU:").pack(anchor="w", pady=(0, 5))
        self.gpu_var = tk.StringVar(value="auto")
        gpu_combo = ctk.CTkComboBox(main_frame, variable=self.gpu_var, values=["auto", "none", "0", "1", "0,1"])
        gpu_combo.pack(fill="x", pady=(0, 10))

        # OMP threads
        ctk.CTkLabel(main_frame, text="OMP Threads:").pack(anchor="w", pady=(0, 5))
        self.omp_threads_var = tk.StringVar(value="2")
        omp_entry = ctk.CTkEntry(main_frame, textvariable=self.omp_threads_var)
        omp_entry.pack(fill="x", pady=(0, 10))

        # Buttons
        button_frame = ctk.CTkFrame(self.dialog)
        button_frame.pack(fill="x", padx=20, pady=10)

        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=self._cancel)
        cancel_btn.pack(side="right", padx=5)

        start_btn = ctk.CTkButton(button_frame, text="Start Training", command=self._start)
        start_btn.pack(side="right", padx=5)

        # Load default values from preset
        self._on_preset_changed()

    def _on_preset_changed(self, event=None):
        """Update default values when preset changes."""
        preset_name = self.preset_var.get()
        if preset_name in self.presets:
            preset = self.presets[preset_name]
            defaults = preset.get('default_resources', {})

            if 'cpus' in defaults:
                self.cpus_var.set(str(defaults['cpus']))
            if 'mem_limit' in defaults:
                self.mem_limit_var.set(defaults['mem_limit'])
            if 'shm_size' in defaults:
                self.shm_size_var.set(defaults['shm_size'])
            if 'gpu' in preset:
                self.gpu_var.set(preset['gpu'])

    def _start(self):
        """Start training with current configuration."""
        try:
            self.result = {
                'preset': self.preset_var.get(),
                'game': self.game_var.get(),
                'algorithm': self.algo_var.get(),
                'run_id': self.run_id_var.get(),
                'cpus': float(self.cpus_var.get()) if self.cpus_var.get() else None,
                'mem_limit': self.mem_limit_var.get() if self.mem_limit_var.get() else None,
                'shm_size': self.shm_size_var.get() if self.shm_size_var.get() else None,
                'gpu': self.gpu_var.get(),
                'omp_threads': int(self.omp_threads_var.get()) if self.omp_threads_var.get() else 2
            }
            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def _cancel(self):
        """Cancel the dialog."""
        self.result = None
        self.dialog.destroy()

    def show(self):
        """Show the dialog and return the result."""
        self.dialog.wait_window()
        return self.result


if __name__ == "__main__":
    app = RetroMLDesktop()
    app.run()
