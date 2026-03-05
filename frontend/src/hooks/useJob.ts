import { useMutation, useQuery } from '@tanstack/react-query';
import apiClient from '../api/client';
import type { JobRequest, JobStatus } from '../types';

async function submitJob(request: JobRequest): Promise<JobStatus> {
  const response = await apiClient.post<JobStatus>('/jobs', request);
  return response.data;
}

async function fetchJobStatus(jobId: string): Promise<JobStatus> {
  const response = await apiClient.get<JobStatus>(`/jobs/${jobId}`);
  return response.data;
}

export function useSubmitJob() {
  return useMutation({
    mutationFn: submitJob,
  });
}

export function useJobStatus(jobId: string | null) {
  return useQuery({
    queryKey: ['job', jobId],
    queryFn: () => fetchJobStatus(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === 'complete' || status === 'failed') return false;
      return 1000; // Poll every second while pending/running
    },
  });
}
