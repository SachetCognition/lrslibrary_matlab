import { Badge, Card, Group, SimpleGrid, Text, Title } from '@mantine/core';
import { useAlgorithmsByMethod } from '../hooks/useAlgorithms';

const SPEED_COLORS: Record<number, string> = {
  1: 'green',
  2: 'lime',
  3: 'yellow',
  4: 'orange',
  5: 'red',
};

interface AlgorithmSelectorProps {
  methodId: string;
  onSelect: (algorithmId: string) => void;
}

export default function AlgorithmSelector({
  methodId,
  onSelect,
}: AlgorithmSelectorProps) {
  const { data: algorithms, isLoading } = useAlgorithmsByMethod(methodId);

  if (isLoading) {
    return <Text>Loading algorithms...</Text>;
  }

  if (!algorithms || algorithms.length === 0) {
    return <Text c="dimmed">No algorithms available for {methodId} yet.</Text>;
  }

  return (
    <div>
      <Title order={3} mb="md">
        Select Algorithm ({methodId})
      </Title>
      <SimpleGrid cols={{ base: 1, sm: 2, md: 3 }}>
        {algorithms.map((algo) => (
          <Card
            key={algo.algorithm_id}
            shadow="sm"
            padding="lg"
            radius="md"
            withBorder
            style={{ cursor: 'pointer' }}
            onClick={() => onSelect(algo.algorithm_id)}
          >
            <Group justify="space-between" mb="xs">
              <Text fw={500}>{algo.algorithm_id}</Text>
              <Badge
                color={SPEED_COLORS[algo.speed_class] ?? 'gray'}
                variant="light"
              >
                Speed: {algo.speed_class}
              </Badge>
            </Group>
            <Text size="sm" c="dimmed">
              {algo.name}
            </Text>
            {algo.is_tensor && (
              <Badge color="violet" variant="light" mt="xs">
                Tensor
              </Badge>
            )}
          </Card>
        ))}
      </SimpleGrid>
    </div>
  );
}
