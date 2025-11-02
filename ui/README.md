# ML Container Management UI

A web-based interface for managing multiple ML training containers with real-time resource monitoring and training progress tracking.

## Features

- **Container Management**: Create, start, stop, and monitor multiple ML training sessions
- **Resource Monitoring**: Real-time CPU, GPU, memory, and disk usage tracking
- **Training Progress**: Live training metrics and progress visualization
- **Video Streaming**: Integrated video streams showing agent learning progress
- **Configuration Management**: Easy setup and management of training parameters
- **Multi-Game Support**: Support for multiple Atari games (Breakout, Pong, Tetris, etc.)

## Architecture

### Backend (FastAPI)
- RESTful API for container management
- WebSocket connections for real-time updates
- Resource monitoring and allocation
- Integration with existing ML training pipeline

### Frontend (Next.js + React)
- Responsive dashboard interface
- Real-time charts and metrics
- Container management controls
- Video streaming integration

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- FFmpeg (for video processing)
- CUDA-capable GPU (recommended)

### Backend Setup
```bash
cd ui/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup
```bash
cd ui/frontend
npm install
npm run dev
```

### Access the UI
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Project Structure

```
ui/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core business logic
│   │   ├── models/         # Data models
│   │   └── main.py         # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # Next.js frontend
│   ├── components/         # React components
│   ├── pages/             # Next.js pages
│   ├── hooks/             # Custom React hooks
│   ├── utils/             # Utility functions
│   ├── package.json
│   └── next.config.js
├── docker-compose.yml      # Docker orchestration
└── README.md
```

## API Endpoints

### Container Management
- `GET /api/containers` - List all containers
- `POST /api/containers` - Create new container
- `GET /api/containers/{id}` - Get container details
- `POST /api/containers/{id}/start` - Start container
- `POST /api/containers/{id}/stop` - Stop container
- `DELETE /api/containers/{id}` - Delete container

### Resource Monitoring
- `GET /api/resources/system` - System resource overview
- `GET /api/resources/containers` - Per-container resource usage
- `WebSocket /ws/resources` - Real-time resource updates

### Training Sessions
- `GET /api/training/sessions` - List training sessions
- `GET /api/training/sessions/{id}` - Get session details
- `GET /api/training/sessions/{id}/metrics` - Get training metrics
- `WebSocket /ws/training/{id}` - Real-time training updates

### Video Streaming
- `GET /api/videos/streams` - List active video streams
- `GET /api/videos/recordings` - List recorded videos
- `WebSocket /ws/video/{container_id}` - Live video stream

## Configuration

### Backend Configuration
Environment variables in `backend/.env`:
```
DATABASE_URL=sqlite:///./containers.db
REDIS_URL=redis://localhost:6379
ML_MODELS_PATH=../models
VIDEO_OUTPUT_PATH=../video
LOG_LEVEL=INFO
```

### Frontend Configuration
Environment variables in `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Development

### Running Tests
```bash
# Backend tests
cd ui/backend
pytest

# Frontend tests
cd ui/frontend
npm test
```

### Code Quality
```bash
# Backend linting
cd ui/backend
black . && flake8 . && mypy .

# Frontend linting
cd ui/frontend
npm run lint
```

## Docker Deployment

### Build and Run
```bash
docker-compose up --build
```

### Production Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Integration with Existing ML Pipeline

The UI system integrates seamlessly with the existing ML training pipeline:

1. **Training Scripts**: Wraps existing `training/train.py` in containerized processes
2. **Configuration**: Extends `conf/config.yaml` for multi-container scenarios
3. **Video Streaming**: Integrates with `tools/stream/stream_eval.py`
4. **Monitoring**: Uses existing TensorBoard and checkpoint systems

## Resource Management

### Allocation Strategy
- Automatic resource detection and allocation
- Configurable limits per container
- Conflict detection and prevention
- Dynamic resource adjustment

### Monitoring
- Real-time CPU, GPU, memory tracking
- Per-container resource usage
- System-wide utilization metrics
- Performance optimization suggestions

## Security

- API authentication and authorization
- Container process isolation
- Resource limit enforcement
- Secure file access controls

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
