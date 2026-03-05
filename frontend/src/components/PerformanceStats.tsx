import { Card, Group, Text, Title } from '@mantine/core';

interface PerformanceStatsProps {
  cputime: number | null;
  algorithmName: string;
  matrixDimensions: string;
}

export default function PerformanceStats({
  cputime,
  algorithmName,
  matrixDimensions,
}: PerformanceStatsProps) {
  return (
    <Card shadow="sm" padding="lg" radius="md" withBorder>
      <Title order={4} mb="sm">
        Performance Statistics
      </Title>
      <Group gap="xl">
        <div>
          <Text size="sm" c="dimmed">
            Algorithm
          </Text>
          <Text fw={500}>{algorithmName}</Text>
        </div>
        <div>
          <Text size="sm" c="dimmed">
            CPU Time
          </Text>
          <Text fw={500}>
            {cputime !== null ? `${cputime.toFixed(3)}s` : 'N/A'}
          </Text>
        </div>
        <div>
          <Text size="sm" c="dimmed">
            Matrix Dimensions
          </Text>
          <Text fw={500}>{matrixDimensions}</Text>
        </div>
      </Group>
    </Card>
  );
}
