import axios from 'axios';

// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle common errors
    if (error.response?.status === 401) {
      // Handle unauthorized
      console.error('Unauthorized access');
    } else if (error.response?.status >= 500) {
      // Handle server errors
      console.error('Server error:', error.response.data);
    }
    return Promise.reject(error);
  }
);

// SWR fetcher function
export const fetcher = (url: string) => api.get(url).then(res => res.data);

// API endpoints
export const endpoints = {
  // Health and info
  health: '/health',
  root: '/',
  
  // Containers
  containers: '/api/containers',
  container: (id: string) => `/api/containers/${id}`,
  containerStart: (id: string) => `/api/containers/${id}/start`,
  containerStop: (id: string) => `/api/containers/${id}/stop`,
  containerMetrics: (id: string) => `/api/containers/${id}/metrics`,
  containerLogs: (id: string) => `/api/containers/${id}/logs`,
  systemResources: '/api/containers/system/resources',
  
  // Resources
  resourcesSystem: '/api/resources/system',
  resourcesSystemHistory: '/api/resources/system/history',
  resourcesContainers: '/api/resources/containers',
  resourcesContainer: (id: string) => `/api/resources/containers/${id}`,
  resourcesContainerHistory: (id: string) => `/api/resources/containers/${id}/history`,
  resourcesSummary: '/api/resources/summary',
  resourcesAlerts: '/api/resources/alerts',
  monitoringStart: '/api/resources/monitoring/start',
  monitoringStop: '/api/resources/monitoring/stop',
  monitoringStatus: '/api/resources/monitoring/status',
  
  // Training
  trainingSessions: '/api/training/sessions',
  trainingSession: (id: string) => `/api/training/sessions/${id}`,
  trainingMetrics: (id: string) => `/api/training/sessions/${id}/metrics`,
  trainingLogs: (id: string) => `/api/training/sessions/${id}/logs`,
  trainingPause: (id: string) => `/api/training/sessions/${id}/pause`,
  trainingResume: (id: string) => `/api/training/sessions/${id}/resume`,
  trainingSummary: '/api/training/summary',
  trainingGames: '/api/training/games',
  trainingAlgorithms: '/api/training/algorithms',
};

// WebSocket URLs
export const wsEndpoints = {
  resources: `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/resources`,
  containers: `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/containers`,
  training: (id: string) => `${process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'}/ws/training/${id}`,
};

// Type definitions
export interface Container {
  id: string;
  name: string;
  config: TrainingConfig;
  status: ContainerStatus;
  created_at: string;
  started_at?: string;
  stopped_at?: string;
  process_id?: number;
  working_directory?: string;
  log_file?: string;
  error_message?: string;
  current_timesteps: number;
  current_episodes: number;
  current_reward: number;
  checkpoint_count: number;
  video_count: number;
}

export interface TrainingConfig {
  game: string;
  algorithm: string;
  total_timesteps: number;
  vec_envs: number;
  learning_rate: number;
  checkpoint_every_sec: number;
  video_recording: boolean;
  fast_mode?: boolean;
  resource_limits: ResourceSpec;
  batch_size: number;
  n_steps: number;
  gamma: number;
  gae_lambda: number;
  clip_range: number;
  ent_coef: number;
  vf_coef: number;
}

export interface ResourceSpec {
  cpu_cores: number;
  memory_gb: number;
  gpu_memory_gb?: number;
  disk_space_gb: number;
}

export type ContainerStatus = 
  | 'created'
  | 'starting'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'stopped'
  | 'error'
  | 'deleted';

export interface SystemMetrics {
  timestamp: string;
  cpu_percent: number;
  memory_percent: number;
  memory_used_gb: number;
  memory_total_gb: number;
  disk_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  gpu_count: number;
  gpu_utilization: number[];
  gpu_memory_used: number[];
  gpu_memory_total: number[];
  gpu_temperature: number[];
}

export interface ProcessMetrics {
  pid: number;
  name: string;
  cpu_percent: number;
  memory_percent: number;
  memory_mb: number;
  gpu_memory_mb: number;
  status: string;
}

export interface TrainingProgress {
  container_id: string;
  container_name: string;
  game: string;
  algorithm: string;
  status: string;
  progress_percentage: number;
  current_timesteps: number;
  total_timesteps: number;
  current_episodes: number;
  current_reward: number;
  runtime_duration?: number;
  estimated_completion?: string;
}

export interface GameConfig {
  game: string;
  display_name: string;
  description: string;
  default_algorithm: string;
  supported_algorithms: string[];
  default_timesteps: number;
  estimated_training_time: string;
}

// API functions
export const containerAPI = {
  list: () => api.get<Container[]>(endpoints.containers),
  get: (id: string) => api.get<Container>(endpoints.container(id)),
  create: (data: { name: string; config: TrainingConfig }) => 
    api.post<{ container_id: string }>(endpoints.containers, data),
  start: (id: string) => api.post(endpoints.containerStart(id)),
  stop: (id: string) => api.post(endpoints.containerStop(id)),
  delete: (id: string) => api.delete(endpoints.container(id)),
  metrics: (id: string) => api.get(endpoints.containerMetrics(id)),
  logs: (id: string, params?: { lines?: number; follow?: boolean }) => 
    api.get(endpoints.containerLogs(id), { params }),
};

export const resourceAPI = {
  system: () => api.get<SystemMetrics>(endpoints.resourcesSystem),
  systemHistory: (params?: { hours?: number }) => 
    api.get<SystemMetrics[]>(endpoints.resourcesSystemHistory, { params }),
  containers: () => api.get(endpoints.resourcesContainers),
  container: (id: string) => api.get(endpoints.resourcesContainer(id)),
  summary: () => api.get(endpoints.resourcesSummary),
  alerts: () => api.get(endpoints.resourcesAlerts),
};

export const trainingAPI = {
  sessions: () => api.get<TrainingProgress[]>(endpoints.trainingSessions),
  session: (id: string) => api.get<TrainingProgress>(endpoints.trainingSession(id)),
  metrics: (id: string) => api.get(endpoints.trainingMetrics(id)),
  logs: (id: string, params?: { lines?: number; follow?: boolean }) => 
    api.get(endpoints.trainingLogs(id), { params }),
  pause: (id: string) => api.post(endpoints.trainingPause(id)),
  resume: (id: string) => api.post(endpoints.trainingResume(id)),
  summary: () => api.get(endpoints.trainingSummary),
  games: () => api.get<GameConfig[]>(endpoints.trainingGames),
  algorithms: () => api.get(endpoints.trainingAlgorithms),
};
