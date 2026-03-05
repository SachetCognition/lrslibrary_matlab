import { useQuery } from '@tanstack/react-query';
import { fetchAlgorithms, fetchAlgorithmsByMethod } from '../api/algorithms';

export function useAlgorithms() {
  return useQuery({
    queryKey: ['algorithms'],
    queryFn: fetchAlgorithms,
  });
}

export function useAlgorithmsByMethod(methodId: string | null) {
  return useQuery({
    queryKey: ['algorithms', methodId],
    queryFn: () => fetchAlgorithmsByMethod(methodId!),
    enabled: !!methodId,
  });
}
