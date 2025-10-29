# ML Container Management UI - Architecture Plan

## Overview

This document outlines the architecture for a web-based UI system to manage multiple ML training containers, track resource usage, and provide real-time monitoring of training sessions.

## Technology Stack Selection

### Backend: FastAPI + Python
**Rationale:**
- Seamless integration with existing Python ML pipeline
- Excellent async support for real-time features
- Automatic OpenAPI documentation
- Native support for WebSockets
- Leverages existing dependencies (PyTorch, Stable-Baselines3)

### Frontend: React + Next.js
**Rationale:**
- Modern, responsive UI with excellent real-time capabilities
- Server-side rendering for better performance
- Built-in API routes for backend integration
- Rich ecosystem for charts and dashboards
- TypeScript support for better maintainability

### Database: SQLite + Redis
**Rationale:**
- SQLite: Lightweight, file-based storage for training metadata
- Redis: In-memory cache for real-time metrics and session state

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                      │
├─────────────────────────────────────────────────────────────┤
│  Dashboard │ Container Mgmt │ Config UI │ Video Streams    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend API (FastAPI)                    │
├─────────────────────────────────────────────────────────────┤
│  Container API │ Resource Monitor │ Training API │ Stream  │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Container Orchestration                     │
├─────────────────────────────────────────────────────────────┤
│  Process Manager │ Resource Allocator │ Session Tracker   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              ML Training Containers                        │
├─────────────────────────────────────────────────────────────┤
│  Container 1  │  Container 2  │  Container 3  │  ...       │
│  (Breakout)   │  (Pong)       │  (Tetris)     │            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Container Management System
- **ContainerManager**: Orchestrates multiple training sessions
- **ResourceAllocator**: Manages CPU/GPU/memory allocation
- **ProcessMonitor**: Tracks container health and performance
- **SessionTracker**: Maintains training session metadata

### 2. Resource Monitoring
- **SystemMonitor**: Real-time system resource tracking
- **GPUMonitor**: CUDA/GPU utilization and memory
- **ProcessTracker**: Per-container resource usage
- **MetricsCollector**: Aggregates and stores metrics

### 3. Training Session API
- **SessionAPI**: CRUD operations for training sessions
- **ConfigAPI**: Manage training configurations
- **ProgressAPI**: Track training progress and metrics
- **VideoAPI**: Access generated videos and streams

### 4. Frontend Dashboard
- **ContainerGrid**: Visual overview of active containers
- **ResourceCharts**: Real-time resource usage graphs
- **TrainingProgress**: Progress bars and metrics
- **VideoViewer**: Live streams and recorded videos

## Directory Structure

```
ui/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── containers.py
│   │   │   ├── resources.py
│   │   │   ├── training.py
│   │   │   └── websocket.py
│   │   ├── core/
│   │   │   ├── container_manager.py
│   │   │   ├── resource_monitor.py
│   │   │   └── config.py
│   │   ├── models/
│   │   │   ├── container.py
│   │   │   ├── session.py
│   │   │   └── metrics.py
│   │   └── main.py
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── components/
│   │   ├── Dashboard/
│   │   ├── ContainerManager/
│   │   ├── ResourceMonitor/
│   │   └── VideoStreamer/
│   ├── pages/
│   │   ├── index.tsx
│   │   ├── containers/
│   │   └── api/
│   ├── hooks/
│   ├── utils/
│   ├── package.json
│   └── next.config.js
└── docker-compose.yml
```

## Integration with Existing System

### 1. Training Pipeline Integration
- Wrap existing `training/train.py` in container management
- Extend `conf/config.yaml` for multi-container scenarios
- Integrate with existing video streaming (`tools/stream/`)

### 2. Resource Monitoring Integration
- Leverage PyTorch's CUDA monitoring capabilities
- Extend existing TensorBoard integration
- Use existing checkpoint management system

### 3. Video System Integration
- Integrate with `tools/stream/stream_eval.py`
- Extend `tools/video_tools/` for UI consumption
- Support multiple concurrent video streams

## Key Features

### 1. Container Management
- Create/start/stop/delete training containers
- Resource allocation per container
- Container health monitoring
- Automatic restart on failure

### 2. Resource Tracking
- Real-time CPU/GPU/memory usage
- Resource allocation visualization
- Conflict detection and prevention
- Performance optimization suggestions

### 3. Training Monitoring
- Live training progress tracking
- TensorBoard integration
- Checkpoint management
- Performance metrics dashboard

### 4. Video Integration
- Live training video streams
- Generated video gallery
- Multi-container video grid
- Video download and sharing

## Security Considerations

- API authentication and authorization
- Container isolation and sandboxing
- Resource limit enforcement
- Secure file access controls

## Performance Considerations

- Async operations for non-blocking UI
- Efficient WebSocket connections
- Resource usage optimization
- Caching strategies for metrics

## Next Steps

1. Set up basic FastAPI backend structure
2. Implement core container management
3. Create resource monitoring system
4. Build initial React dashboard
5. Integrate with existing ML pipeline
