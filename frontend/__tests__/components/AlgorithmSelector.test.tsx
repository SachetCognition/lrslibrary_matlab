import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AlgorithmSelector from '../../src/components/AlgorithmSelector';

// Mock the useAlgorithmsByMethod hook
vi.mock('../../src/hooks/useAlgorithms', () => ({
  useAlgorithmsByMethod: () => ({
    data: [
      {
        method_id: 'RPCA',
        algorithm_id: 'FPCP',
        name: 'Fast PCP (Rodriguez and Wohlberg, 2013)',
        speed_class: 1,
        is_tensor: false,
      },
      {
        method_id: 'RPCA',
        algorithm_id: 'IALM',
        name: 'Inexact ALM (Lin et al. 2009)',
        speed_class: 1,
        is_tensor: false,
      },
    ],
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

describe('AlgorithmSelector', () => {
  it('renders filtered algorithms for selected method', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <AlgorithmSelector methodId="RPCA" onSelect={onSelect} />
    );

    expect(screen.getByText('FPCP')).toBeDefined();
    expect(screen.getByText('IALM')).toBeDefined();
  });

  it('shows speed class badges', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <AlgorithmSelector methodId="RPCA" onSelect={onSelect} />
    );

    const speedBadges = screen.getAllByText('Speed: 1');
    expect(speedBadges.length).toBe(2);
  });

  it('calls onSelect on click', () => {
    const onSelect = vi.fn();
    renderWithProviders(
      <AlgorithmSelector methodId="RPCA" onSelect={onSelect} />
    );

    fireEvent.click(screen.getByText('FPCP'));
    expect(onSelect).toHaveBeenCalledWith('FPCP');
  });
});
