# Container Management System Specification

## Overview

The Container Management System provides orchestration for multiple ML training sessions, ensuring proper resource allocation, isolation, and monitoring.

## Core Components

### 1. ContainerManager

**Purpose**: Central orchestrator for all training containers

**Key Responsibilities**:
- Create and manage training containers
- Allocate resources (CPU, GPU, memory)
- Monitor container health and status
- Handle container lifecycle events

**API Methods**:
```python
class ContainerManager:
    def create_container(self, config: TrainingConfig) -> Container
    def start_container(self, container_id: str) -> bool
    def stop_container(self, container_id: str) -> bool
    def delete_container(self, container_id: str) -> bool
    def list_containers(self) -> List[Container]
    def get_container_status(self, container_id: str) -> ContainerStatus
    def allocate_resources(self, container_id: str, resources: ResourceSpec) -> bool
```

### 2. Container Model

**Container States**:
- `CREATED`: Container created but not started
- `STARTING`: Container initialization in progress
- `RUNNING`: Active training session
- `PAUSED`: Training paused (can be resumed)
- `STOPPING`: Graceful shutdown in progress
- `STOPPED`: Container stopped (can be restarted)
- `ERROR`: Container encountered an error
- `DELETED`: Container removed from system

**Container Configuration**:
```python
@dataclass
class TrainingConfig:
    game: str                    # e.g., "breakout", "pong"
    algorithm: str               # e.g., "ppo", "dqn"
    total_timesteps: int
    vec_envs: int
    learning_rate: float
    checkpoint_every_sec: int
    video_recording: bool
    resource_limits: ResourceSpec
    
@dataclass
class ResourceSpec:
    cpu_cores: float             # e.g., 2.0 cores
    memory_gb: float             # e.g., 4.0 GB
    gpu_memory_gb: Optional[float]  # e.g., 2.0 GB VRAM
    disk_space_gb: float         # e.g., 10.0 GB
```

### 3. Resource Allocation System

**Resource Types**:
- **CPU**: Cores allocated per container
- **Memory**: RAM allocation with limits
- **GPU**: VRAM allocation and compute units
- **Disk**: Storage space for models/videos
- **Network**: Bandwidth for streaming (future)

**Allocation Strategy**:
```python
class ResourceAllocator:
    def __init__(self):
        self.total_resources = self._detect_system_resources()
        self.allocated_resources = {}
        self.available_resources = self.total_resources.copy()
    
    def can_allocate(self, resources: ResourceSpec) -> bool
    def allocate(self, container_id: str, resources: ResourceSpec) -> bool
    def deallocate(self, container_id: str) -> bool
    def get_utilization(self) -> ResourceUtilization
```

**Resource Monitoring**:
- Real-time usage tracking per container
- System-wide resource utilization
- Alerts for resource conflicts
- Automatic scaling recommendations

### 4. Process Management

**Process Isolation**:
- Each container runs in separate Python process
- Environment variable isolation
- File system sandboxing
- Resource limit enforcement

**Process Communication**:
```python
class ProcessManager:
    def spawn_training_process(self, config: TrainingConfig) -> Process
    def monitor_process_health(self, process_id: str) -> ProcessHealth
    def send_command(self, process_id: str, command: Command) -> bool
    def get_process_metrics(self, process_id: str) -> ProcessMetrics
```

**Commands**:
- `PAUSE`: Pause training (save checkpoint)
- `RESUME`: Resume training from checkpoint
- `STOP`: Graceful shutdown
- `KILL`: Force termination
- `CHECKPOINT`: Save current state
- `STATUS`: Get current status

### 5. Session Tracking

**Training Session Metadata**:
```python
@dataclass
class TrainingSession:
    session_id: str
    container_id: str
    game: str
    algorithm: str
    start_time: datetime
    end_time: Optional[datetime]
    status: SessionStatus
    config: TrainingConfig
    metrics: SessionMetrics
    checkpoints: List[Checkpoint]
    videos: List[VideoFile]
```

**Session Metrics**:
- Training progress (timesteps, episodes)
- Performance metrics (reward, loss)
- Resource usage over time
- Video generation status

### 6. Container Networking

**Port Management**:
- TensorBoard ports (6006+)
- Video streaming ports (8000+)
- API communication ports
- WebSocket connections

**Service Discovery**:
- Container registry with endpoints
- Health check endpoints
- Metric collection endpoints

## Integration Points

### 1. Existing Training System
```python
# Wrapper for existing train.py
class ContainerizedTrainer:
    def __init__(self, config: TrainingConfig, container_id: str):
        self.config = config
        self.container_id = container_id
        self.resource_monitor = ResourceMonitor(container_id)
        
    def run_training(self):
        # Set resource limits
        self._apply_resource_limits()
        
        # Initialize existing training pipeline
        from training.train import main as train_main
        
        # Run with monitoring
        with self.resource_monitor:
            train_main(self.config.to_yaml())
```

### 2. Video Streaming Integration
```python
class ContainerVideoStreamer:
    def __init__(self, container_id: str):
        self.container_id = container_id
        self.stream_port = self._allocate_stream_port()
        
    def start_stream(self):
        # Use existing stream_eval.py
        from tools.stream.stream_eval import ContinuousEvaluator
        evaluator = ContinuousEvaluator(
            config=self.config,
            output_port=self.stream_port
        )
        evaluator.start()
```

### 3. Resource Monitoring Integration
```python
class ContainerResourceMonitor:
    def __init__(self, container_id: str, process_id: int):
        self.container_id = container_id
        self.process_id = process_id
        
    def get_metrics(self) -> ResourceMetrics:
        return ResourceMetrics(
            cpu_usage=psutil.Process(self.process_id).cpu_percent(),
            memory_usage=psutil.Process(self.process_id).memory_info(),
            gpu_usage=torch.cuda.utilization() if torch.cuda.is_available() else 0,
            gpu_memory=torch.cuda.memory_usage() if torch.cuda.is_available() else 0
        )
```

## Database Schema

### Container Table
```sql
CREATE TABLE containers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    game TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    stopped_at TIMESTAMP,
    config_json TEXT,
    resource_spec_json TEXT
);
```

### Sessions Table
```sql
CREATE TABLE training_sessions (
    id TEXT PRIMARY KEY,
    container_id TEXT REFERENCES containers(id),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_timesteps INTEGER,
    final_reward REAL,
    status TEXT,
    metrics_json TEXT
);
```

### Resource Usage Table
```sql
CREATE TABLE resource_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    container_id TEXT REFERENCES containers(id),
    timestamp TIMESTAMP,
    cpu_usage REAL,
    memory_usage REAL,
    gpu_usage REAL,
    gpu_memory REAL
);
```

## Error Handling

### Container Failures
- Automatic restart with exponential backoff
- Checkpoint recovery on restart
- Resource cleanup on failure
- Error logging and notification

### Resource Conflicts
- Pre-allocation validation
- Dynamic resource adjustment
- Priority-based allocation
- Graceful degradation

### System Overload
- Container prioritization
- Automatic pausing of low-priority containers
- Resource usage alerts
- Emergency shutdown procedures

## Security Considerations

### Process Isolation
- Separate user accounts per container
- File system permissions
- Resource limit enforcement
- Network isolation

### API Security
- Authentication tokens
- Rate limiting
- Input validation
- Audit logging

## Performance Optimization

### Resource Efficiency
- Lazy loading of ML models
- Shared memory for common data
- Efficient checkpoint storage
- Optimized video encoding

### Scalability
- Horizontal scaling support
- Load balancing
- Resource pooling
- Caching strategies
