import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MantineProvider } from '@mantine/core';
import MethodSelector from '../../src/components/MethodSelector';

// Mock the useAlgorithms hook
vi.mock('../../src/hooks/useAlgorithms', () => ({
  useAlgorithms: () => ({
    data: [
      {
        method_id: 'RPCA',
        algorithm_id: 'FPCP',
        name: 'Fast PCP',
        speed_class: 1,
        is_tensor: false,
      },
      {
        method_id: 'RPCA',
        algorithm_id: 'IALM',
        name: 'Inexact ALM',
        speed_class: 1,
        is_tensor: false,
      },
      {
        method_id: 'MC',
        algorithm_id: 'TEST',
        name: 'Test MC',
        speed_class: 2,
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
    <QueryClientProvider client={queryClient}>
      <MantineProvider>{ui}</MantineProvider>
    </QueryClientProvider>
  );
}

describe('MethodSelector', () => {
  it('renders all method categories', () => {
    const onSelect = vi.fn();
    renderWithProviders(<MethodSelector onSelect={onSelect} />);

    expect(screen.getByText('RPCA')).toBeDefined();
    expect(screen.getByText('MC')).toBeDefined();
    expect(screen.getByText('NMF')).toBeDefined();
    expect(screen.getByText('TD')).toBeDefined();
    expect(screen.getByText('NTF')).toBeDefined();
    expect(screen.getByText('ST')).toBeDefined();
    expect(screen.getByText('LRR')).toBeDefined();
    expect(screen.getByText('TTD')).toBeDefined();
  });

  it('calls onSelect with correct method_id on click', () => {
    const onSelect = vi.fn();
    renderWithProviders(<MethodSelector onSelect={onSelect} />);

    fireEvent.click(screen.getByText('RPCA'));
    expect(onSelect).toHaveBeenCalledWith('RPCA');
  });

  it('shows algorithm count badges', () => {
    const onSelect = vi.fn();
    renderWithProviders(<MethodSelector onSelect={onSelect} />);

    expect(screen.getByText('2 algorithms')).toBeDefined(); // RPCA has 2
    expect(screen.getByText('1 algorithms')).toBeDefined(); // MC has 1
  });
});
