import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ResultViewer from '../../src/components/ResultViewer';

const mockJobStatus = {
  job_id: 'test-job-123',
  status: 'complete' as const,
  progress: 1.0,
  cputime: 0.5,
  result_urls: {
    L: '/api/jobs/test-job-123/results/L',
    S: '/api/jobs/test-job-123/results/S',
    O: '/api/jobs/test-job-123/results/O',
  },
};

// Mock the useJobStatus hook
vi.mock('../../src/hooks/useJob', () => ({
  useJobStatus: () => ({
    data: mockJobStatus,
    isLoading: false,
  }),
}));

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>
  );
}

describe('ResultViewer', () => {
  it('renders 4 panels with correct labels', () => {
    renderWithProviders(<ResultViewer jobId="test-job-123" />);

    expect(screen.getByText('Input')).toBeDefined();
    expect(screen.getByText('Low-Rank (L)')).toBeDefined();
    expect(screen.getByText('Sparse (S)')).toBeDefined();
    expect(screen.getByText('Outliers (O)')).toBeDefined();
  });

  it('renders decomposition results title', () => {
    renderWithProviders(<ResultViewer jobId="test-job-123" />);

    expect(screen.getByText('Decomposition Results')).toBeDefined();
  });

  it('shows frame scrubber', () => {
    renderWithProviders(<ResultViewer jobId="test-job-123" />);

    expect(screen.getByText('Frame: 0')).toBeDefined();
  });
});
