import { Badge, Card, Group, SimpleGrid, Text, Title } from '@mantine/core';
import { useAlgorithms } from '../hooks/useAlgorithms';
import type { AlgorithmInfo } from '../types';

const METHOD_LABELS: Record<string, string> = {
  RPCA: 'Robust PCA',
  MC: 'Matrix Completion',
  NMF: 'Non-negative Matrix Factorization',
  TD: 'Tensor Decomposition',
  NTF: 'Non-negative Tensor Factorization',
  ST: 'Subspace Tracking',
  LRR: 'Low-Rank Representation',
  TTD: 'Three-Term Decomposition',
};

interface MethodSelectorProps {
  onSelect: (methodId: string) => void;
}

export default function MethodSelector({ onSelect }: MethodSelectorProps) {
  const { data: algorithms, isLoading } = useAlgorithms();

  if (isLoading) {
    return <Text>Loading methods...</Text>;
  }

  // Group by method_id and count
  const methodCounts: Record<string, number> = {};
  const allMethods = new Set<string>();
  (algorithms ?? []).forEach((algo: AlgorithmInfo) => {
    allMethods.add(algo.method_id);
    methodCounts[algo.method_id] = (methodCounts[algo.method_id] ?? 0) + 1;
  });

  // Include all 8 categories even if no algorithms registered yet
  const methods = [
    'RPCA',
    'MC',
    'NMF',
    'TD',
    'NTF',
    'ST',
    'LRR',
    'TTD',
  ];

  return (
    <div>
      <Title order={3} mb="md">
        Select Method Category
      </Title>
      <SimpleGrid cols={{ base: 1, sm: 2, md: 4 }}>
        {methods.map((methodId) => (
          <Card
            key={methodId}
            shadow="sm"
            padding="lg"
            radius="md"
            withBorder
            style={{ cursor: 'pointer' }}
            onClick={() => onSelect(methodId)}
          >
            <Group justify="space-between" mb="xs">
              <Text fw={500}>{methodId}</Text>
              <Badge color="blue" variant="light">
                {methodCounts[methodId] ?? 0} algorithms
              </Badge>
            </Group>
            <Text size="sm" c="dimmed">
              {METHOD_LABELS[methodId] ?? methodId}
            </Text>
          </Card>
        ))}
      </SimpleGrid>
    </div>
  );
}
