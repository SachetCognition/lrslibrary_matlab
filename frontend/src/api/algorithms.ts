import apiClient from './client';
import type { AlgorithmInfo } from '../types';

export async function fetchAlgorithms(): Promise<AlgorithmInfo[]> {
  const response = await apiClient.get<AlgorithmInfo[]>('/algorithms');
  return response.data;
}

export async function fetchAlgorithmsByMethod(
  methodId: string
): Promise<AlgorithmInfo[]> {
  const response = await apiClient.get<AlgorithmInfo[]>(
    `/algorithms/${methodId}`
  );
  return response.data;
}
