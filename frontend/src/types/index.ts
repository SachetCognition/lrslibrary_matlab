export interface AlgorithmInfo {
  method_id: string;
  algorithm_id: string;
  name: string;
  speed_class: number;
  is_tensor: boolean;
}

export interface JobRequest {
  method_id: string;
  algorithm_id: string;
  video_id: string;
  params?: Record<string, unknown>;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'complete' | 'failed';
  progress: number;
  cputime?: number;
  result_urls?: Record<string, string>;
  error?: string;
  nframes?: number;
}

export interface VideoMetadata {
  video_id: string;
  width: number;
  height: number;
  nframes: number;
  fps: number;
}
