"""
Hour-long ML Analytics Video Callback for continuous training recording.
Creates 1-hour videos that can be merged into 10-hour epic learning journeys.
"""

import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import cv2
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv




class HourVideoCallback(BaseCallback):
    """
    Callback that records hour-long ML analytics videos during training.
    Each video shows 1 hour of continuous neural network learning with
    enhanced visualizations that can be merged into epic 10-hour journeys.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        hours_to_record: int = 10,
        fps: int = 30,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.config = config
        self.hours_to_record = hours_to_record
        self.fps = fps
        self.total_timesteps = config['train']['total_timesteps']
        
        # Video output settings
        self.output_dir = Path(config['paths']['videos_milestones'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Hour-long video configuration
        self.hour_duration_seconds = 3600  # 1 hour = 3600 seconds
        self.frames_per_hour = self.hour_duration_seconds * self.fps  # 108,000 frames per hour
        
        # Current recording state
        self.current_hour = 1
        self.current_video_writer = None
        self.current_video_path = None
        self.hour_start_time = None
        self.hour_frame_count = 0
        

        
        if self.verbose >= 1:
            print(f"[HourVideo] HourVideoCallback initialized:")
            print(f"  - Output directory: {self.output_dir.absolute()}")
            print(f"  - Hours to record: {hours_to_record}")
            print(f"  - Video resolution: 960x540 @ {fps} FPS")
            print(f"  - Frames per hour: {self.frames_per_hour:,}")
            print(f"  - Total frames for {hours_to_record} hours: {self.frames_per_hour * hours_to_record:,}")

    def _on_training_start(self) -> None:
        """Called when training starts."""
        if self.verbose >= 1:
            print(f"[HourVideo] Starting hour-long video recording...")
        self._start_new_hour_video()

    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.current_video_writer is None:
            return True
            
        # Check if we should record this frame (sample every N steps for target FPS)
        steps_per_frame = max(1, self.model.n_envs // self.fps)
        if self.num_timesteps % steps_per_frame != 0:
            return True
            
        # Record frame for current hour video
        self._record_frame()
        
        # Check if hour is complete
        if self.hour_frame_count >= self.frames_per_hour:
            self._finish_current_hour()
            
            # Start next hour if we haven't reached the limit
            if self.current_hour <= self.hours_to_record:
                self._start_new_hour_video()
            
        return True

    def _on_training_end(self) -> None:
        """Called when training ends."""
        try:
            if self.current_video_writer is not None:
                self._finish_current_hour()

            if self.verbose >= 1:
                print(f"[HourVideo] Training ended. Recorded {self.current_hour - 1} hour(s) of video.")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[HourVideo] [WARNING] Error during training end cleanup: {e}")

    def _start_new_hour_video(self) -> None:
        """Start recording a new hour-long video."""
        if self.current_hour > self.hours_to_record:
            return
            
        # Create video filename
        self.current_video_path = self.output_dir / f"hour_{self.current_hour:02d}_neural_learning.mp4"
        
        # Initialize FFmpeg process for video encoding
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '960x540',  # Resolution
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),  # Frame rate
            '-i', '-',  # Input from stdin
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            str(self.current_video_path)
        ]
        
        try:
            self.current_video_writer = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.hour_start_time = time.time()
            self.hour_frame_count = 0
            
            if self.verbose >= 1:
                print(f"[HourVideo] Started recording Hour {self.current_hour}: {self.current_video_path.name}")
                
        except Exception as e:
            print(f"[HourVideo] Error starting video recording: {e}")
            self.current_video_writer = None

    def _record_frame(self) -> None:
        """Record a single frame to the current hour video."""
        if self.current_video_writer is None:
            return
            
        try:
            # Get current environment state
            env = self.training_env
            if hasattr(env, 'envs'):
                # Use first environment for recording
                game_env = env.envs[0]
            else:
                game_env = env
                
            # Get game frame
            if hasattr(game_env, 'render'):
                game_frame = game_env.render(mode='rgb_array')
            else:
                # Fallback: create a black frame
                game_frame = np.zeros((210, 160, 3), dtype=np.uint8)
            
            # Extract ML analytics
            analytics = self._extract_ml_analytics()
            
            # Create enhanced frame with neural network visualization
            enhanced_frame = self._create_enhanced_frame(game_frame, analytics)
            
            # Write frame to video
            self.current_video_writer.stdin.write(enhanced_frame.tobytes())
            self.hour_frame_count += 1
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[HourVideo] Error recording frame: {e}")

    def _finish_current_hour(self) -> None:
        """Finish the current hour video."""
        if self.current_video_writer is None:
            return
            
        try:
            # Close FFmpeg process
            self.current_video_writer.stdin.close()
            self.current_video_writer.wait()
            
            duration = time.time() - self.hour_start_time if self.hour_start_time else 0
            
            if self.verbose >= 1:
                print(f"[OK] Hour {self.current_hour} video completed!")
                print(f"[FILE] LOCATION: {self.current_video_path.absolute()}")
                print(f"[STATS] DETAILS:")
                print(f"   - Duration: {self.hour_frame_count / self.fps:.1f}s ({self.hour_frame_count:,} frames)")
                print(f"   - Recording time: {duration:.1f}s")
                print(f"[VIDEO] TO VIEW: {self.current_video_path.absolute()}")
                print("=" * 60)
            
            self.current_hour += 1
            self.current_video_writer = None
            self.current_video_path = None
            
        except Exception as e:
            print(f"[HourVideo] Error finishing video: {e}")

    def _extract_ml_analytics(self) -> Dict[str, Any]:
        """Extract ML analytics from the current model state, including real layer activations."""
        analytics = {
            'step': self.num_timesteps,
            'hour': self.current_hour,
            'frame_in_hour': self.hour_frame_count,
            'progress_pct': (self.num_timesteps / self.total_timesteps) * 100,
            'action_probs': None,
            'value_estimate': 0.0,
            'learning_rate': 0.00025,
            'episode_reward': 0.0,
            'layer_activations': {}  # Store real layer activations
        }

        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'predict'):
                # Get current observation from environment
                env = self.training_env
                if hasattr(env, 'buf_obs'):
                    obs = env.buf_obs
                    if obs is not None and len(obs) > 0:
                        # Use first environment's observation
                        single_obs = obs[0:1]  # Shape: (1, ...)

                        # Extract layer activations using forward hooks
                        import torch
                        activations = {}
                        hooks = []

                        def get_activation(name):
                            def hook(module, input, output):
                                # Store mean activation across spatial dimensions
                                if isinstance(output, torch.Tensor):
                                    # For conv layers: average across spatial dimensions
                                    if len(output.shape) == 4:  # (batch, channels, height, width)
                                        activations[name] = output[0].mean(dim=(1, 2)).cpu().numpy()
                                    # For linear layers: just use the values
                                    elif len(output.shape) == 2:  # (batch, features)
                                        activations[name] = output[0].cpu().numpy()
                            return hook

                        # Register hooks for feature extractor layers
                        if hasattr(self.model.policy, 'features_extractor'):
                            features = self.model.policy.features_extractor
                            if hasattr(features, 'cnn'):
                                # For NatureCNN architecture
                                cnn = features.cnn
                                if len(cnn) > 0:
                                    hooks.append(cnn[0].register_forward_hook(get_activation('conv1')))
                                if len(cnn) > 2:
                                    hooks.append(cnn[2].register_forward_hook(get_activation('conv2')))
                                if len(cnn) > 4:
                                    hooks.append(cnn[4].register_forward_hook(get_activation('conv3')))

                            # Hook for the linear layer after CNN
                            if hasattr(features, 'linear'):
                                hooks.append(features.linear.register_forward_hook(get_activation('dense')))

                        # Get action probabilities and value estimate
                        with self.model.policy.set_training_mode(False):
                            actions, values, log_probs = self.model.policy(single_obs)

                            if hasattr(self.model.policy, 'action_dist'):
                                action_dist = self.model.policy.action_dist
                                if hasattr(action_dist, 'probs'):
                                    analytics['action_probs'] = action_dist.probs[0].detach().cpu().numpy()

                            if values is not None:
                                analytics['value_estimate'] = float(values[0].detach().cpu().numpy())

                        # Remove hooks
                        for hook in hooks:
                            hook.remove()

                        # Store activations in analytics
                        analytics['layer_activations'] = activations

        except Exception as e:
            if self.verbose >= 2:
                print(f"[HourVideo] Analytics extraction error: {e}")

        return analytics

    def _create_enhanced_frame(self, game_frame: np.ndarray, analytics: Dict[str, Any]) -> np.ndarray:
        """Create enhanced frame with neural network visualization."""
        # Resize game frame to fit right side
        game_resized = cv2.resize(game_frame, (480, 540))
        
        # Create analytics panel (left side)
        analytics_panel = np.zeros((540, 480, 3), dtype=np.uint8)
        
        # Draw neural network and analytics
        self._draw_analytics_panel(analytics_panel, analytics)
        self._draw_neural_network(analytics_panel, analytics)
        
        # Combine panels
        enhanced_frame = np.hstack([analytics_panel, game_resized])
        
        # Convert BGR to RGB for FFmpeg
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
        
        return enhanced_frame

    def _draw_analytics_panel(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw ML analytics information on the panel."""
        # Colors
        white = (255, 255, 255)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)

        # Font
        font = cv2.FONT_HERSHEY_SIMPLEX
        mono_font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        cv2.putText(panel, "BREAKOUT - DQN Neural Activity Viewer", (10, 30), font, 0.5, white, 1)

        # Subtitle with training parameters
        hour = analytics.get('hour', 1)
        progress = analytics.get('progress_pct', 0)
        cv2.putText(panel, f"Hour {hour} | Progress: {progress:.1f}% | lr=2.5e-4", (10, 50), font, 0.35, (180, 180, 180), 1)

        # Training metrics
        y_pos = 80
        line_height = 20

        step = analytics.get('step', 0)
        frame_in_hour = analytics.get('frame_in_hour', 0)

        cv2.putText(panel, f"Step:       {step:,}", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height
        cv2.putText(panel, f"Hour Frame: {frame_in_hour:,}", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height
        cv2.putText(panel, f"Progress:   {progress:.2f}%", (10, int(y_pos)), mono_font, 0.4, white, 1)
        y_pos += line_height * 1.5

        # Action probabilities
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs')

        if action_probs is not None and len(action_probs) >= 4:
            cv2.putText(panel, "Policy Distribution:", (10, int(y_pos)), mono_font, 0.4, (200, 200, 200), 1)
            y_pos += line_height * 0.8

            for i, (name, prob) in enumerate(zip(action_names, action_probs[:4])):
                color = yellow if prob == max(action_probs[:4]) else (150, 150, 150)
                cv2.putText(panel, f"  {name:5s}: {prob:.3f}", (10, int(y_pos)), mono_font, 0.35, color, 1)
                y_pos += line_height * 0.8

        y_pos += line_height * 0.5

        # Value estimate
        value = analytics.get('value_estimate', 0.0)
        cv2.putText(panel, f"Value Est:  {value:.3f}", (10, int(y_pos)), mono_font, 0.4, green, 1)

    def _draw_neural_network(self, panel: np.ndarray, analytics: Dict[str, Any]) -> None:
        """Draw neural network visualization with REAL layer activations from the model."""
        # Network area
        net_x, net_y = 20, 280
        net_width, net_height = 440, 240

        # Layer configuration (realistic CNN for Atari)
        layers = [
            {'name': 'Input', 'shape': '84×84×4', 'nodes': 8, 'color': (0, 255, 0), 'key': None},      # Green
            {'name': 'Conv1', 'shape': '32@20×20', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv1'},   # Blue
            {'name': 'Conv2', 'shape': '64@9×9', 'nodes': 6, 'color': (255, 100, 0), 'key': 'conv2'},     # Blue
            {'name': 'Dense', 'shape': '512', 'nodes': 8, 'color': (100, 100, 255), 'key': 'dense'},      # Light blue
            {'name': 'Output', 'shape': '4', 'nodes': 4, 'color': (0, 165, 255), 'key': 'output'}          # Orange
        ]

        # Get real layer activations
        layer_activations = analytics.get('layer_activations', {})

        # Calculate positions
        layer_spacing = net_width // (len(layers) - 1)
        layer_positions = []

        for i, layer in enumerate(layers):
            x = net_x + i * layer_spacing
            nodes = layer['nodes']
            node_spacing = net_height // (nodes + 1)

            positions = []
            for j in range(nodes):
                y = net_y + (j + 1) * node_spacing
                positions.append((x, y))

            layer_positions.append(positions)

        # Get node activations for each layer
        step = analytics.get('step', 0)
        layer_node_activations = []

        for layer_idx, layer in enumerate(layers):
            layer_key = layer.get('key')
            num_nodes = layer['nodes']

            if layer_idx == 0:  # Input layer - use constant moderate activation
                activations = [0.5] * num_nodes
            elif layer_idx == len(layers) - 1:  # Output layer - use action probabilities
                action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])
                activations = [action_probs[i] if i < len(action_probs) else 0.25 for i in range(num_nodes)]
            elif layer_key and layer_key in layer_activations:
                # Use REAL activations from the network
                real_activations = layer_activations[layer_key]
                # Normalize to 0-1 range and sample nodes
                if len(real_activations) > 0:
                    # Take absolute value and normalize
                    abs_activations = np.abs(real_activations)
                    max_val = abs_activations.max() if abs_activations.max() > 0 else 1.0
                    normalized = abs_activations / max_val

                    # Sample evenly across the activation array to get values for our display nodes
                    indices = np.linspace(0, len(normalized) - 1, num_nodes, dtype=int)
                    activations = [float(normalized[i]) for i in indices]
                else:
                    activations = [0.5] * num_nodes
            else:
                # Fallback to moderate activation if no real data available
                activations = [0.5] * num_nodes

            layer_node_activations.append(activations)

        # Draw connections based on real activations
        for i in range(len(layer_positions) - 1):
            current_layer = layer_positions[i]
            next_layer = layer_positions[i + 1]
            current_activations = layer_node_activations[i]
            next_activations = layer_node_activations[i + 1]

            for j, start_pos in enumerate(current_layer):
                for k, end_pos in enumerate(next_layer):
                    # Connection strength based on REAL activations of connected nodes
                    current_act = current_activations[j] if j < len(current_activations) else 0.5
                    next_act = next_activations[k] if k < len(next_activations) else 0.5
                    strength = (current_act + next_act) / 2.0

                    if strength > 0.3:  # Only draw connections with sufficient activation
                        alpha = int(strength * 200)
                        color = (alpha // 3, alpha // 3, alpha)
                        thickness = 1 if strength > 0.6 else 1
                        cv2.line(panel, start_pos, end_pos, color, thickness)

        # Draw nodes with REAL activations
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action_probs = analytics.get('action_probs', [0.25, 0.25, 0.25, 0.25])

        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            activations = layer_node_activations[layer_idx]

            for node_idx, pos in enumerate(positions):
                # Get REAL activation for this node
                activation = activations[node_idx] if node_idx < len(activations) else 0.5

                # Node color based on REAL activation
                base_color = layer['color']
                brightness = int(activation * 255)
                node_color = tuple(int(c * brightness / 255) for c in base_color)

                # Draw node
                radius = 8 if layer_idx == len(layers) - 1 else 6
                cv2.circle(panel, pos, radius, node_color, -1)
                cv2.circle(panel, pos, radius, (255, 255, 255), 1)

                # Highlight chosen action
                if layer_idx == len(layers) - 1 and action_probs is not None:
                    if node_idx < len(action_probs) and action_probs[node_idx] == max(action_probs):
                        # Pulsing effect
                        pulse = int(128 + 127 * np.sin(step * 0.1))
                        cv2.circle(panel, pos, radius + 2, (pulse, pulse, pulse), 2)

        # Layer labels
        for layer_idx, (layer, positions) in enumerate(zip(layers, layer_positions)):
            if positions:
                label_x = positions[0][0]
                label_y = net_y + net_height + 20

                # Layer name
                cv2.putText(panel, layer['name'], (label_x - 20, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                # Layer shape
                cv2.putText(panel, layer['shape'], (label_x - 20, label_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        # Output action labels
        if len(layer_positions) > 0:
            output_positions = layer_positions[-1]
            for i, (pos, name) in enumerate(zip(output_positions, action_names)):
                if i < len(output_positions):
                    cv2.putText(panel, name, (pos[0] + 15, pos[1] + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
